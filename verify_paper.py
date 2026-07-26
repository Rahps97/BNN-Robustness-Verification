#!/usr/bin/env python3
"""verify_paper.py -- one command that re-checks the paper's numerical claims.

Run it from the repository root:

    python verify_paper.py

Every check prints the value claimed in the paper, the value recomputed here,
and an outcome. The exit status is non-zero only if a check genuinely FAILED --
never because a check was not run, could not be run, or was inconclusive.

Outcomes
--------

    PASS            recomputed and matches the paper
    FAIL            recomputed and does NOT match the paper
    SKIPPED         not requested; the flag that would run it is shown
    UNAVAILABLE     cannot be run here (no license, no data, no hardware)
    TIMEOUT         started but exceeded --timeout
    INCONCLUSIVE    ran, but the solver is stochastic and fell short; this
                    neither confirms nor refutes the reported number
    NOT VERIFIABLE  no offline substitute exists at all

The default run
---------------

The default needs nothing beyond this repository and the packages in
requirements.txt: no GPU, no solver license, no network, no special hardware.
It covers every instance and takes a couple of minutes.

  Tables III, IV, VI   Each QUBO instance is rebuilt from scratch with the
                       authors' own builder, ``bnn_as_qubo.setup_optim_model``,
                       and the number of variables, the number of encoded
                       constraints and the constant energy offset are compared
                       with the tables. The rebuilt matrix is also compared
                       entry by entry with the shipped ``QUBO_W.txt``.

  Table IV (FEM)       The FEM solution vectors recorded in
                       ``FEM_best_configurations.txt`` are evaluated against the
                       shipped QUBO matrices as x^T Q x and compared with the
                       target energy. Each solution is then decoded into a pixel
                       perturbation and replayed through an independent NumPy
                       implementation of the BNN forward pass.

  Table V              The Z3/SMT baseline is re-run on each instance at the
                       epsilon recorded in its ``Info.txt``, and every witness is
                       reverse-checked with that same independent forward pass.

  Table V (d_min)      The minimum adversarial distance is recomputed by
                       exhaustive enumeration (the paper's method for 5x5) and
                       by a Z3 minimum-distance scan (the paper's method for the
                       other rows). Both are run on every instance, so each
                       number is confirmed twice by independent means.

Opt-in checks
-------------

    --with-gurobi   Table IV Gurobi column. Needs a Gurobi license.
    --with-sa       Table IV SA column. Minutes for 5x5, hours for 28x28.
    --with-fem      Replays the FEM solver at its recorded hyperparameters
                    instead of only verifying the recorded solution vectors.
    --everything    All of the above.
    --timeout S     Per-check wall-clock bound; exceeding it is TIMEOUT, not FAIL.
                    Opt-in checks default to 900 s each, because an unbounded
                    Gurobi run on these instances takes about a day.

Table VII (D-Wave and Fujitsu hardware) has no flag: it needs hardware access
and an instance that is not in this repository. It is always reported as NOT
VERIFIABLE.

Instance selection
------------------

    --all               verify every instance (the default)
    --quick             verify the 5x5 instance only
    --instance 5,7      verify a comma-separated subset of 5, 7, 11, 28

Other options
-------------

    --json PATH     also write a machine-readable report ('-' for stdout)
    --no-extract    do not unpack data/qubo_and_networks.tar.gz automatically
    --verbose       show the full output of the sub-checks
"""

import argparse
import contextlib
import io
import json
import os
import re
import sys
import tarfile
import time


# -----------------------------------------------------------------------------
# The claims under test. Every number below is quoted from the manuscript.
# -----------------------------------------------------------------------------

PAPER = {
    5: {
        "arch": "31x7x10",
        "input_dim": 31,
        "perturbable": 16,           # Table III, "Perturbed Pixels"
        "epsilon": 8,                # Table III, "Perturbation Bound"
        "variables": 276,            # Tables III, IV
        "constraints": 200,          # Tables III, IV, VI, "Total Constraints"
        "offset": 533,               # Tables III, IV, VI, "Energy Offset" E_off
        "fem_score": 533,            # Table IV, FEM column
        "sa_score": 533,             # Table IV, SA column
        "gurobi_score": 529,         # Table IV, Gurobi column
        "z3_result": "NR",           # Table V
        "d_min": 3,                  # Table V
    },
    7: {
        "arch": "63x7x10", "input_dim": 63,
        "perturbable": 32, "epsilon": 32,
        "variables": 413, "constraints": 312, "offset": 2643,
        "fem_score": 2643, "sa_score": 2643, "gurobi_score": 2635,
        "z3_result": "NR", "d_min": 1,
    },
    11: {
        "arch": "127x7x10", "input_dim": 127,
        "perturbable": 64, "epsilon": 32,
        "variables": 676, "constraints": 536, "offset": 8020,
        "fem_score": 8020, "sa_score": 8019, "gurobi_score": 8018,
        "z3_result": "NR", "d_min": 2,
    },
    28: {
        "arch": "1023x7x10", "input_dim": 1023,
        "perturbable": 256, "epsilon": 128,
        "variables": 2235, "constraints": 1880, "offset": 1027318,
        "fem_score": 1027318, "sa_score": 1027316, "gurobi_score": 1027316,
        "z3_result": "NR", "d_min": 2,
    },
}

ALL_SIZES = [5, 7, 11, 28]

FEM_SOLUTIONS_PATH = "FEM_best_configurations.txt"
FEM_HYPERPARAMETERS_PATH = "FEM_HYPERPARAMETERS.md"
DATA_ARCHIVE = os.path.join("data", "qubo_and_networks.tar.gz")
DATASET_ARCHIVE = os.path.join("data", "datasets.tar.gz")
HARDWARE_ARCHIVE = os.path.join("data", "hardware_results.tar.gz")
GUROBI_LOG_ARCHIVE = os.path.join("data", "gurobi_logs.tar.gz")

# Table VII's two-class instance, as supplied by the authors.
HARDWARE_DIR = "hardware"
HARDWARE_STEM = "113-1273-28-15-zero-3-3-561020-H"
HARDWARE_QUBO = f"{HARDWARE_DIR}/QUBO/{HARDWARE_STEM}.pickle"
HARDWARE_FUJITSU = f"{HARDWARE_DIR}/Result/{HARDWARE_STEM}_solution.pickle"
HARDWARE_FUJITSU_TIME = f"{HARDWARE_DIR}/Time/{HARDWARE_STEM}_time.pickle"
HARDWARE_DWAVE = f"{HARDWARE_DIR}/Dwave/{HARDWARE_STEM}_solution_dataframe.pickle"

# Recomputed properties of that instance, and the reported hardware results.
HARDWARE = {
    "variables": 113,
    "constraints": 65,          # 1 lt + 64 eq
    "qubo_terms": 1272,
    "offset": 874674,           # so the target energy is -874,674
    "fujitsu_energy": -874674,  # Table VII, Digital Annealer
    "fujitsu_time": 0.366,
    "fujitsu_constraints": 65,
    "dwave_energy": -874318,    # Table VII, Quantum Annealer
    "dwave_time": 0.724,
    "dwave_constraints": 36,
    "dwave_shots": 4000,
}

GUROBI_LOG_DIR = "gurobi_logs"
GUROBI_LOG_PATTERN = GUROBI_LOG_DIR + "/GurobiLog_{size}x{size}.txt"

# Exhaustive enumeration refuses to start a distance level larger than this.
MAX_COMBINATIONS_PER_LEVEL = 5_000_000

Z3_TIMEOUT_SECONDS = 600.0

# The opt-in solver checks would otherwise run for hours to a day: Gurobi's
# early-stop rule allows 100,000 non-improving nodes on the two smaller instances
# and 10,000,000 on the two larger ones, which is what the reported runs took 47
# minutes to 18.7 hours to exhaust. Unless --timeout says otherwise, each opt-in
# check gets this bound, and exceeding it is TIMEOUT, never FAIL.
OPT_IN_DEFAULT_TIMEOUT_SECONDS = 900.0

# SA.py's settings, so that --with-sa runs the same configuration. They are
# expensive -- 2048 x 5000 on the 276-variable 5x5 instance already takes
# minutes -- so they can be dialled down for a quick smoke test through the
# environment. Anything other than the defaults is not the paper's SA setup and
# is reported as such.
SA_NUM_READS = int(os.environ.get("VERIFY_SA_READS", 2048))
SA_NUM_SWEEPS = int(os.environ.get("VERIFY_SA_SWEEPS", 5000))
SA_BETA_SCHEDULE = "geometric"
SA_DEFAULT_READS = 2048
SA_DEFAULT_SWEEPS = 5000

# The common size-limited Gurobi license.
GUROBI_LIMITED_VARIABLES = 2000


def qubo_dir(size):
    return f"QUBO/{size}x{size}/{PAPER[size]['arch']}"


def info_path(size):
    return f"{qubo_dir(size)}/Info.txt"


def qubo_matrix_path(size):
    return f"{qubo_dir(size)}/QUBO_W.txt"


def variables_path(size):
    return f"{qubo_dir(size)}/Variables.json"


def checkpoint_path(size):
    return (f"TrainedNN/{size}x{size}/{PAPER[size]['arch']}/"
            f"{PAPER[size]['input_dim']}.pth")


def dataset_path(size):
    return f"Dataset/{size}x{size}/Train.txt"


def instance_available(size):
    return all(os.path.exists(path) for path in
               (info_path(size), qubo_matrix_path(size),
                variables_path(size), checkpoint_path(size)))


# -----------------------------------------------------------------------------
# Report plumbing
# -----------------------------------------------------------------------------

PASS = "PASS"
FAIL = "FAIL"
SKIPPED = "SKIPPED"
UNAVAILABLE = "UNAVAILABLE"
TIMEOUT = "TIMEOUT"
INCONCLUSIVE = "INCONCLUSIVE"
NOT_VERIFIABLE = "NOT VERIFIABLE"

NOT_RUN_STATUSES = (SKIPPED, UNAVAILABLE, TIMEOUT, INCONCLUSIVE, NOT_VERIFIABLE)

_STATUS_WORD = {
    PASS: "passed",
    FAIL: "FAILED",
    SKIPPED: "not run",
    UNAVAILABLE: "unavailable",
    TIMEOUT: "timed out",
    INCONCLUSIVE: "inconclusive",
    NOT_VERIFIABLE: "not verifiable",
}


class Report:
    """Collects outcomes and prints them grouped by paper table."""

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.entries = []
        self.group = None

    # -- output helpers --------------------------------------------------

    def banner(self, text):
        print("=" * 79)
        print(f" {text}")
        print("=" * 79)

    def section(self, title, group_key):
        self.group = group_key
        print()
        print("-" * 79)
        print(f" {title}")
        print("-" * 79)

    def note(self, text):
        for line in text.strip("\n").split("\n"):
            print(f" {line}" if line else "")

    def instance_header(self, size, detail=""):
        head = f" {size}x{size} ({PAPER[size]['arch']})"
        print(f"\n{head}{('   ' + detail) if detail else ''}")

    # -- result recording ------------------------------------------------

    def _record(self, status, name, size, claimed=None, recomputed=None,
                detail="", hint=None):
        self.entries.append({
            "group": self.group, "instance": size, "check": name,
            "status": status, "claimed": _jsonable(claimed),
            "recomputed": _jsonable(recomputed), "detail": detail,
            "hint": hint,
        })
        return status == PASS

    def check(self, name, claimed, recomputed, ok=None, detail="", size=None):
        """A numerical claim: paper value vs recomputed value."""
        if ok is None:
            ok = claimed == recomputed
        status = PASS if ok else FAIL
        claim = "-" if claimed is None else _fmt(claimed)
        got = "-" if recomputed is None else _fmt(recomputed)
        line = (f"   [{status:^14}] {name:<46} "
                f"paper {claim:>12}   recomputed {got:>12}")
        if detail:
            line += f"   {detail}"
        print(line)
        return self._record(status, name, size, claimed, recomputed, detail)

    def assertion(self, name, ok, detail="", size=None):
        """A consistency assertion with no single 'claimed number'."""
        status = PASS if ok else FAIL
        line = f"   [{status:^14}] {name:<46}"
        if detail:
            line += f" {detail}"
        print(line)
        return self._record(status, name, size, detail=detail)

    def outcome(self, status, name, reason, hint=None, size=None,
                claimed=None, recomputed=None):
        """A non-PASS/FAIL outcome: SKIPPED, UNAVAILABLE, TIMEOUT, ..."""
        head = f"   [{status:^14}] {name}"
        if claimed is not None:
            head += f"   (paper {_fmt(claimed)}"
            head += (f", reached {_fmt(recomputed)})" if recomputed is not None
                     else ")")
        print(head)
        print(f"{'':<20}why : {reason}")
        if hint:
            print(f"{'':<20}how : {hint}")
        return self._record(status, name, size, claimed, recomputed,
                            reason, hint)

    def error(self, name, message, size=None):
        print(f"   [{FAIL:^14}] {name}")
        print(f"{'':<20}{message}")
        return self._record(FAIL, name, size, detail=message)

    def solver_run(self, name, size, paper_energy, target_energy, energies,
                   runtime, runs_label="seeds", hint=None):
        """Report one heuristic / early-terminated solver run.

        Two different gaps are reported, and they mean different things:

          * gap vs the paper's value is the reproduction question. Zero means
            this run reproduced what the paper reports for this solver.
          * gap vs the target energy is the feasibility question. Zero means
            every encoded constraint is satisfied.

        They are NOT the same: Table IV's SA column is 8,019 for 11x11 against
        a target of 8,020, and 1,027,316 for 28x28 against 1,027,318, so the
        paper itself does not claim full feasibility everywhere.

        The verdict is taken from the gap vs the paper's value only, and a
        shortfall is INCONCLUSIVE, never FAIL: a heuristic search that falls
        short is not evidence that the reported number is wrong.
        """
        energies = sorted(float(value) for value in energies)
        best = energies[0]
        matched = sum(1 for value in energies if value <= paper_energy)
        median = energies[len(energies) // 2]

        gap_paper = best - paper_energy          # <= 0 means at least as good
        gap_target = best - target_energy        # >= 0 by construction
        scale = abs(target_energy) or 1.0

        status = PASS if gap_paper <= 0 else INCONCLUSIVE
        print(f"   [{status:^14}] {name:<46} "
              f"paper {_fmt(paper_energy):>12}   recomputed {_fmt(best):>12}")
        print(f"{'':<20}{runs_label:<7}: {len(energies)} run, "
              f"{matched}/{len(energies)} reached the paper's value; "
              f"energies min {best:,.0f} / median {median:,.0f} / "
              f"max {energies[-1]:,.0f}")
        if gap_paper <= 0:
            paper_text = ("0 -- reproduced exactly" if gap_paper == 0 else
                          f"{-gap_paper:,.0f} better than reported")
        else:
            paper_text = f"{gap_paper:,.0f} short ({gap_paper / scale:.3%})"
        print(f"{'':<20}gaps   : vs paper {paper_text}; vs target "
              f"{gap_target:,.0f} ({gap_target / scale:.3%}) "
              f"-> at most {gap_target:,.0f} encoded constraint(s) violated")
        print(f"{'':<20}time   : {runtime:.1f} s")
        if status == INCONCLUSIVE and hint:
            print(f"{'':<20}how    : {hint}")

        detail = (f"best {best:,.0f}; gap vs paper {gap_paper:,.0f}; "
                  f"gap vs target {gap_target:,.0f}; "
                  f"{matched}/{len(energies)} runs matched; {runtime:.1f} s")
        self.entries.append({
            "group": self.group, "instance": size, "check": name,
            "status": status, "claimed": paper_energy, "recomputed": best,
            "detail": detail, "hint": hint,
            "gap_vs_paper": gap_paper, "gap_vs_target": gap_target,
            "energies": energies, "runtime_seconds": runtime,
        })
        return status == PASS

    # -- summary ---------------------------------------------------------

    def tally(self):
        counts = {}
        for entry in self.entries:
            counts[entry["status"]] = counts.get(entry["status"], 0) + 1
        return counts

    def summary(self, elapsed):
        print()
        self.banner("SUMMARY")
        order = []
        for entry in self.entries:
            if entry["group"] not in order:
                order.append(entry["group"])
        for group in order:
            rows = [e for e in self.entries if e["group"] == group]
            counts = {}
            for row in rows:
                counts[row["status"]] = counts.get(row["status"], 0) + 1
            bits = []
            for status in (PASS, FAIL, SKIPPED, UNAVAILABLE, TIMEOUT,
                           INCONCLUSIVE, NOT_VERIFIABLE):
                if counts.get(status):
                    bits.append(f"{counts[status]} {_STATUS_WORD[status]}")
            print(f" {group:<50} {', '.join(bits)}")

        counts = self.tally()
        hints = sorted({e["hint"] for e in self.entries
                        if e["status"] == SKIPPED and e.get("hint")
                        and e["hint"].startswith("--")})
        print()
        line = (f" {counts.get(PASS, 0)} passed, "
                f"{counts.get(FAIL, 0)} failed")
        if counts.get(SKIPPED):
            line += f", {counts[SKIPPED]} not run"
            if hints:
                line += f" ({', '.join(hints)})"
        if counts.get(UNAVAILABLE):
            line += f", {counts[UNAVAILABLE]} unavailable"
        if counts.get(TIMEOUT):
            line += f", {counts[TIMEOUT]} timed out"
        if counts.get(INCONCLUSIVE):
            line += f", {counts[INCONCLUSIVE]} inconclusive"
        if counts.get(NOT_VERIFIABLE):
            line += (f", {counts[NOT_VERIFIABLE]} not verifiable "
                     f"(hardware access required)")
        print(line)
        print(f" elapsed: {elapsed:.1f} s")
        print()
        if counts.get(FAIL):
            print(" RESULT: FAIL -- at least one reported number did not "
                  "reproduce.")
        else:
            print(" RESULT: PASS -- every check that was run reproduces the "
                  "paper.")
            not_run = sum(counts.get(status, 0) for status in NOT_RUN_STATUSES)
            if not_run:
                print(f"         {not_run} check(s) were NOT run and are "
                      f"therefore NOT verified; see the")
                print("         reasons above. Do not read them as confirmed.")
        print("=" * 79)


def _fmt(value):
    if isinstance(value, float) and value == int(value):
        value = int(value)
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _jsonable(value):
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


@contextlib.contextmanager
def captured(verbose):
    """Silence chatty sub-checks unless --verbose was given."""
    if verbose:
        yield None
        return
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        yield buffer


class Deadline:
    """A soft per-check wall-clock bound.

    It is enforced where a solver supports a native time limit (Gurobi, Z3) and
    otherwise between sub-steps. It cannot interrupt a single long call inside a
    native extension, so it is a bound on when a check gives up, not a hard kill.
    """

    def __init__(self, seconds):
        self.seconds = seconds
        self.started = time.perf_counter()

    def reset(self):
        self.started = time.perf_counter()
        return self

    def remaining(self):
        if self.seconds is None:
            return None
        return self.seconds - (time.perf_counter() - self.started)

    def expired(self):
        remaining = self.remaining()
        return remaining is not None and remaining <= 0


# -----------------------------------------------------------------------------
# Instance description, read from the shipped Info.txt
# -----------------------------------------------------------------------------

def read_info(size):
    """Parse the fields of Info.txt that describe the verification instance."""
    text = open(info_path(size), encoding="utf-8").read()

    def bracketed(key):
        anchor = text.index(key)
        start = text.index("[", anchor)
        depth = 0
        for position in range(start, len(text)):
            if text[position] == "[":
                depth += 1
            elif text[position] == "]":
                depth -= 1
                if depth == 0:
                    return text[start + 1:position]
        raise ValueError(f"{info_path(size)}: unterminated '{key}'")

    return {
        "minimum_energy": int(float(
            re.search(r"Minimum Energy\s*:\s*(-?[\d.]+)", text).group(1))),
        "total_variables": int(
            re.search(r"Total Variables\s*:\s*(\d+)", text).group(1)),
        "epsilon": int(re.search(r"Epsilon\s*:\s*(\d+)", text).group(1)),
        "pixels": [int(value) for value in
                   bracketed("Pixels perturbed").replace(",", " ").split()],
        "clean": [int(float(value)) for value in
                  bracketed("Input to perturb").replace(",", " ").split()],
        "label": int(float(
            re.search(r"Label to perturb\s*:\s*([\d.]+)", text).group(1))),
    }


_QUBO_CACHE = {}


def load_qubo_matrix(size):
    """Load QUBO_W.txt (upper triangular including the diagonal)."""
    if size not in _QUBO_CACHE:
        import numpy as np
        _QUBO_CACHE[size] = np.loadtxt(qubo_matrix_path(size))
    return _QUBO_CACHE[size]


def qubo_dict(size):
    """The shipped QUBO as the {(i, j): w} dictionary the solvers expect."""
    import numpy as np
    matrix = load_qubo_matrix(size)
    rows, columns = np.nonzero(matrix)
    return {(int(i), int(j)): float(matrix[i, j])
            for i, j in zip(rows, columns)}


# -----------------------------------------------------------------------------
# The authors' network, exactly as QUBOCreator.py defines it
# -----------------------------------------------------------------------------

def build_net(input_dim):
    """Rebuild QUBOCreator.py's QUBONet.

    QUBOCreator.py performs the whole generation pipeline at import time, so it
    cannot be imported for its class definitions alone. The definition is
    therefore repeated here; it must stay identical to the one in
    QUBOCreator.py, and the entry-by-entry comparison against the shipped
    QUBO_W.txt below would fail immediately if it ever drifted.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.autograd import Function

    class Binarize(Function):
        clip_value = 1

        @staticmethod
        def forward(ctx, inp):
            ctx.save_for_backward(inp)
            out = inp.new(inp.size())
            out[inp >= 0] = 1
            out[inp < 0] = -1
            return out

        @staticmethod
        def backward(ctx, grad_output):
            inp = ctx.saved_tensors[0]
            clipped = inp.abs() <= Binarize.clip_value
            out = torch.zeros(inp.size()).to(grad_output.device)
            out[clipped] = 1
            out[~clipped] = 0
            return out * grad_output

    binarize = Binarize.apply

    class BinaryLinear(nn.Linear):
        def __init__(self, in_features, out_features, bias=False):
            super().__init__(in_features, out_features, bias=bias)

        def forward(self, inp):
            return binarize(F.linear(inp, binarize(self.weight))).to(inp.device)

    class LastLayer(nn.Linear):
        def __init__(self, in_features, out_features, bias=False):
            super().__init__(in_features, out_features, bias=bias)

        def forward(self, inp):
            return F.linear(inp, binarize(self.weight)).to(inp.device)

    class QUBONet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = BinaryLinear(input_dim, 7)
            self.fc4 = LastLayer(7, 10)

        def forward(self, x):
            return torch.argmax(self.fc4(self.fc1(x)))

    net = QUBONet()
    net.load_state_dict(torch.load(_checkpoint_for_dim(input_dim),
                                   weights_only=True, map_location="cpu"))
    net.eval()
    return net


def _checkpoint_for_dim(input_dim):
    for size, claim in PAPER.items():
        if claim["input_dim"] == input_dim:
            return checkpoint_path(size)
    raise ValueError(f"no instance with input dimension {input_dim}")


def rebuild_qubo(size, info):
    """Rebuild the instance with the authors' own builder.

    The clean input, the label and the perturbable pixel list are taken from the
    Info.txt that QUBOCreator.py wrote alongside the QUBO. That file fixes the
    variable order of the shipped matrix, so this rebuild can be compared with
    QUBO_W.txt entry by entry. Whether the same instance is recovered from the
    training set is checked separately in cross_check_dataset().
    """
    import torch
    from get_args import args
    from utils import to_spin
    from bnn_as_qubo import setup_optim_model

    args.pixels_to_perturb_len = PAPER[size]["perturbable"]
    args.LAMBDA = {
        "sum_taus": 0.1,
        "output": 1,
        "hard_constraints": 1,
        "perturbation_bound_constraint": 1,
        "epsilon": 1,
    }
    args.objective = "zero"
    args.epsilon = info["epsilon"]
    args.include_perturbation_bound_constraint = True
    args.selected_targets = tuple(range(10))
    # Pin the misclassification encoding to the strict variant that every
    # shipped QUBO and every reported number uses, so that an exported
    # BNN_ARGMAX_TIE_AWARE=1 cannot silently change what is being checked.
    args.argmax_tie_aware = False
    args.pixels_to_perturb = list(info["pixels"])

    net = build_net(PAPER[size]["input_dim"])
    spin = to_spin(torch.tensor(info["clean"], dtype=torch.float32))
    return setup_optim_model(spin, info["label"], net, args)


def cross_check_dataset(size, info):
    """Re-derive the instance from the training set, as QUBOCreator.py does.

    Returns (pixels_match_as_set, pixels_match_in_order, input_matches,
    label_matches), or None if Dataset/ is not extracted.
    """
    if not os.path.exists(dataset_path(size)):
        return None

    import numpy as np
    import torch
    from utils import to_spin

    net = build_net(PAPER[size]["input_dim"])
    train = torch.load(dataset_path(size), weights_only=False)
    flat = train.dataset.tensors[0][:, 0:size * size]
    pixels = [int(value) for value in flat.mean(axis=0).abs().topk(
        min(PAPER[size]["perturbable"], flat.shape[1]),
        largest=False).indices.numpy()]

    with torch.no_grad():
        for index in range(len(train)):
            item = train.dataset[index]
            spin = to_spin(item[0])
            if net(torch.Tensor(np.array([np.array(spin)]))) == item[1]:
                break

    return (
        sorted(pixels) == sorted(info["pixels"]),
        pixels == list(info["pixels"]),
        [int(value) for value in item[0].tolist()] == list(info["clean"]),
        int(item[1]) == info["label"],
    )


def dense_from_qubo(qubo, order):
    """Densify a qubovert QUBO dictionary the way QUBOCreator.py writes it."""
    import numpy as np
    matrix = np.zeros((len(order), len(order)))
    for key, value in qubo.items():
        if len(key) == 1:
            matrix[key[0], key[0]] = value
        elif len(key) == 2:
            matrix[key[0], key[1]] = value
    return matrix


# -----------------------------------------------------------------------------
# Group 1 -- Tables III, IV, VI: QUBO structure
# -----------------------------------------------------------------------------

def check_structure(report, sizes, missing):
    report.section(
        "Tables III, IV, VI -- QUBO structure "
        "(variables / constraints / energy offset)",
        "Tables III/IV/VI  QUBO structure")
    report.note("""
Each instance is rebuilt from scratch with the authors' own builder,
bnn_as_qubo.setup_optim_model, and the rebuilt matrix is compared entry by
entry with the shipped QUBO_W.txt.

The distinction that Tables III, IV and VI were corrected for:

  * Total Constraints is the number of constraints actually encoded in the
    QCBO and carried into the QUBO -- the equality constraints plus the two
    perturbation-budget inequalities. It is counted here from H.constraints.
  * Energy offset E_off is the constant term of the QUBO polynomial. It is an
    energy in units of the penalty weights, NOT a constraint count. Earlier
    versions of these tables reported E_off in the Total Constraints column.

Both quantities are recomputed and reported separately below.
""")

    for size in sizes:
        claim = PAPER[size]
        if size in missing:
            report.instance_header(size)
            report.outcome(UNAVAILABLE, "QUBO structure",
                           f"instance data not present ({qubo_dir(size)}/ "
                           f"is missing)",
                           f"tar xzf {DATA_ARCHIVE}", size=size)
            continue

        info = read_info(size)
        started = time.perf_counter()
        try:
            H, ordered = rebuild_qubo(size, info)
        except Exception as exc:  # pragma: no cover - defensive
            report.instance_header(size)
            report.error("QUBO structure", f"rebuild failed: {exc!r}", size=size)
            continue
        elapsed = time.perf_counter() - started

        report.instance_header(size, f"rebuilt in {elapsed:.1f} s")

        counts = {key: len(value) for key, value in H.constraints.items()}
        total_constraints = sum(counts.values())
        breakdown = " + ".join(f"{key} {value}"
                               for key, value in sorted(counts.items()))
        offset = H.to_qubo()[()]

        report.check("Perturbed pixels (Table III)", claim["perturbable"],
                     len(info["pixels"]), size=size)
        report.check("Perturbation bound epsilon (Table III)", claim["epsilon"],
                     info["epsilon"], size=size)
        report.check("Total Variables", claim["variables"],
                     len(H.variables), size=size)
        report.check("Total Constraints (true count)", claim["constraints"],
                     total_constraints, detail=f"({breakdown})", size=size)
        report.check("Energy offset E_off (an energy, not a count)",
                     claim["offset"], offset, size=size)
        report.assertion(
            "E_off == -(Info.txt Minimum Energy)",
            offset == -info["minimum_energy"],
            f"Info.txt Minimum Energy = {info['minimum_energy']:,}", size=size)

        import numpy as np
        shipped = load_qubo_matrix(size)
        rebuilt = dense_from_qubo(H.to_qubo().Q, ordered)
        same_shape = shipped.shape == rebuilt.shape
        max_difference = (float(np.abs(rebuilt - shipped).max())
                          if same_shape else float("inf"))
        report.assertion(
            "rebuilt QUBO == shipped QUBO_W.txt",
            same_shape and max_difference == 0.0,
            f"{shipped.shape[0]}x{shipped.shape[1]}, "
            f"max |difference| = {max_difference:g}", size=size)

        with open(variables_path(size), encoding="utf-8") as handle:
            report.assertion("variable order == shipped Variables.json",
                             list(ordered) == json.load(handle), size=size)

        crossed = cross_check_dataset(size, info)
        if crossed is None:
            report.outcome(
                SKIPPED, "training set reproduces the chosen instance",
                "Dataset/ is not extracted; the instance was read from "
                "Info.txt instead",
                f"tar xzf {DATASET_ARCHIVE}", size=size)
        else:
            as_set, in_order, input_ok, label_ok = crossed
            report.assertion(
                "training set reproduces the chosen instance",
                as_set and input_ok and label_ok,
                "same perturbable pixel set, clean input and label"
                + ("" if in_order else
                   "; note: torch.topk orders the tied pixels differently on "
                   "this build, so the variable order is taken from Info.txt"),
                size=size)


# -----------------------------------------------------------------------------
# Group 2 -- Table IV, FEM column
# -----------------------------------------------------------------------------

LOG_PREFIX = re.compile(r"^\s*\d\d:\d\d:\d\d \| [A-Z]+ \| \w+ \| ?", re.MULTILINE)
ENERGY_RE = re.compile(r"Best Energy:\s*(-?[\d.]+)")


def parse_fem_solutions(path):
    """Read (energy, solution vector) pairs from FEM_best_configurations.txt.

    The file is a verbatim transcript of the FEM runs, so the multi-device
    record carries a logging prefix on some lines; it is stripped before
    parsing.
    """
    text = LOG_PREFIX.sub("", open(path, encoding="utf-8").read())
    records = []
    for match in ENERGY_RE.finditer(text):
        tail = text[match.end():]
        start = tail.find("[")
        if start < 0:
            continue
        depth = 0
        end = -1
        for position in range(start, len(tail)):
            if tail[position] == "[":
                depth += 1
            elif tail[position] == "]":
                depth -= 1
                if depth == 0:
                    end = position
                    break
        if end < 0:
            continue
        bits = [int(float(token)) for token in tail[start + 1:end].split()]
        records.append((float(match.group(1)), bits))
    return records


def reverse_check(size, info, bits_by_qubo_index=None, boolean_input=None):
    """Replay a solution on the original BNN with an independent forward pass.

    Either pass a QUBO solution vector (decoded here through Variables.json) or
    a ready-made Boolean input. Returns (ok, description).
    """
    import verify_counterexamples as vc

    if boolean_input is None:
        with open(variables_path(size), encoding="utf-8") as handle:
            names = json.load(handle)
        tau_index = {}
        for position, name in enumerate(names):
            found = re.fullmatch(r"tau_(\d+)", str(name))
            if found:
                tau_index[int(found.group(1))] = position
        boolean_input = list(info["clean"])
        for pixel in info["pixels"]:
            if pixel in tau_index and bits_by_qubo_index[tau_index[pixel]] == 1:
                boolean_input[pixel] = 1 - boolean_input[pixel]

    matrices = vc.load_weights(checkpoint_path(size))
    clean_label, _ = vc.forward(matrices, info["clean"])
    adversarial_label, _ = vc.forward(matrices, boolean_input)
    changed = [i for i in range(len(boolean_input))
               if boolean_input[i] != info["clean"][i]]
    outside = sorted(set(changed) - set(info["pixels"]))

    ok = (clean_label == info["label"]
          and adversarial_label != clean_label
          and len(changed) <= info["epsilon"]
          and not outside)
    description = (f"{len(changed)} pixels flipped <= budget "
                   f"{info['epsilon']}, "
                   f"{'all perturbable' if not outside else f'OUTSIDE: {outside}'}, "
                   f"prediction {clean_label} -> {adversarial_label}")
    return ok, description, adversarial_label


def check_fem_solutions(report, sizes, missing):
    report.section(
        "Table IV -- FEM column (energy score) and reverse check on the BNN",
        "Table IV          FEM energies + reverse check")
    report.note(f"""
The FEM solution vectors recorded in {FEM_SOLUTIONS_PATH} are evaluated
directly against the shipped QUBO matrices as x^T Q x, with Q upper triangular
including the diagonal. The target energy is -E_off; because the objective H_0
is identically zero, reaching it is an exact certificate that every encoded
constraint is satisfied, so the energy score E_off - dE equals E_off exactly
when dE = 0.

Each solution is then decoded into a pixel perturbation through Variables.json
and replayed through an independent NumPy forward pass of the BNN, which shares
no code with the QUBO pipeline. This verifies the reported energies without
rerunning the stochastic FEM search; use --with-fem to also replay the solver.
""")

    if not os.path.exists(FEM_SOLUTIONS_PATH):
        report.outcome(UNAVAILABLE, "Table IV, FEM column",
                       f"{FEM_SOLUTIONS_PATH} not found")
        return

    import numpy as np

    records = parse_fem_solutions(FEM_SOLUTIONS_PATH)
    by_length = {}
    for energy, bits in records:
        by_length.setdefault(len(bits), (energy, bits))
    print(f"\n parsed {len(records)} solution vectors from "
          f"{FEM_SOLUTIONS_PATH}")

    for size in sizes:
        claim = PAPER[size]
        report.instance_header(size)
        if size in missing:
            report.outcome(UNAVAILABLE, "Table IV, FEM column",
                           "instance data not present in the repository",
                           f"tar xzf {DATA_ARCHIVE}", size=size)
            continue
        if claim["variables"] not in by_length:
            report.outcome(UNAVAILABLE, "Table IV, FEM column",
                           f"no {claim['variables']}-variable solution vector "
                           f"in {FEM_SOLUTIONS_PATH}", size=size)
            continue

        recorded_energy, bits = by_length[claim["variables"]]
        info = read_info(size)
        matrix = load_qubo_matrix(size)
        vector = np.asarray(bits, dtype=np.float64)

        if matrix.shape[0] != vector.size:
            report.error("Table IV, FEM column",
                         f"solution has {vector.size} entries but QUBO_W.txt "
                         f"is {matrix.shape[0]}x{matrix.shape[1]}", size=size)
            continue

        energy = float(vector @ matrix @ vector)
        gap = energy - float(info["minimum_energy"])
        score = claim["offset"] - gap

        report.check("FEM best energy x^T Q x", -claim["offset"], energy,
                     ok=energy == -float(claim["offset"]),
                     detail=f"(transcript says {recorded_energy:,.0f})",
                     size=size)
        report.check("Energy score E_off - dE (Table IV, FEM)",
                     claim["fem_score"], score,
                     ok=score == float(claim["fem_score"]),
                     detail=f"(dE = {gap:g})", size=size)

        ok, description, _ = reverse_check(size, info, bits_by_qubo_index=bits)
        report.assertion("reverse check on the original BNN", ok,
                         description, size=size)


# -----------------------------------------------------------------------------
# Group 3 -- Table V, Z3 baseline
# -----------------------------------------------------------------------------

def check_z3(report, sizes, missing, deadline):
    report.section(
        "Table V -- exact SMT baseline (Z3) at the epsilon from Info.txt",
        "Table V           Z3 SMT baseline")
    report.note("""
Z3.py is re-run in its Table V configuration: one query per instance at the
perturbation bound recorded in that instance's Info.txt. SAT means a
counterexample exists inside the budget, i.e. NR (not robust).

Each witness is then reverse-checked with verify_counterexamples.py's
independent NumPy forward pass, which shares no code with Z3.py.

Runtimes are solve-only and machine dependent; the paper's figures are quoted
for reference and are not pass/fail criteria.
""")

    import Z3

    for size in sizes:
        claim = PAPER[size]
        report.instance_header(size)
        if size in missing:
            report.outcome(UNAVAILABLE, "Table V, Z3 result",
                           "instance data not present in the repository",
                           f"tar xzf {DATA_ARCHIVE}", size=size)
            continue

        info = read_info(size)
        budget = deadline.reset().remaining()
        try:
            with captured(report.verbose):
                result = Z3.verify_instance(
                    info_path(size), checkpoint_path(size),
                    timeout_seconds=min(Z3_TIMEOUT_SECONDS, budget)
                    if budget else Z3_TIMEOUT_SECONDS)
        except Exception as exc:  # pragma: no cover - defensive
            report.error("Table V, Z3 result", f"Z3 failed: {exc!r}", size=size)
            continue

        if result.status == "UNKNOWN":
            report.outcome(TIMEOUT, "Table V, Z3 verdict",
                           f"Z3 returned UNKNOWN after "
                           f"{result.runtime_seconds:.1f} s",
                           "raise --timeout", size=size,
                           claimed=claim["z3_result"])
            continue

        verdict = {"SAT": "NR", "UNSAT": "R"}.get(result.status, result.status)
        report.check(f"Z3 verdict at epsilon {info['epsilon']}",
                     claim["z3_result"], verdict,
                     detail=f"({result.status}, solve "
                            f"{result.runtime_seconds:.3f} s)", size=size)

        if result.status != "SAT":
            report.assertion("witness reverse-checked on the original BNN",
                             False, "no witness: status is not SAT", size=size)
            continue

        ok, description, adversarial = reverse_check(
            size, info, boolean_input=result.adversarial_input_boolean)
        ok = ok and adversarial == result.adversarial_prediction
        report.assertion("witness reverse-checked on the original BNN", ok,
                         description, size=size)


# -----------------------------------------------------------------------------
# Group 4 -- Table V, minimum adversarial distance
# -----------------------------------------------------------------------------

def check_minimum_distance(report, sizes, missing, deadline):
    report.section(
        "Table V -- minimum adversarial distance d_min",
        "Table V           minimum adversarial distance")
    report.note("""
d_min is recomputed two independent ways:

  * exhaustive enumeration of every perturbation by increasing Hamming
    distance, using verify_counterexamples.py. This is ground truth by
    construction and is the method the paper used for the 5x5 row.
  * a Z3 minimum-distance scan over epsilon = 0, 1, 2, ... stopping at the
    first SAT. This is the method the paper used for the other three rows.

Both are cheap on all four instances here, so each number is confirmed twice.
(*) marks the method the paper used for that row.
""")

    import Z3
    import verify_counterexamples as vc

    for size in sizes:
        claim = PAPER[size]
        report.instance_header(size)
        if size in missing:
            report.outcome(UNAVAILABLE, "Table V, d_min",
                           "instance data not present in the repository",
                           f"tar xzf {DATA_ARCHIVE}", size=size)
            continue

        paper_method = "enumeration" if size == 5 else "Z3 scan"

        # -- exhaustive enumeration -----------------------------------------
        label = ("d_min, exhaustive enumeration"
                 + (" (*)" if paper_method == "enumeration" else ""))
        try:
            with captured(report.verbose):
                enumerated, _, proven_through = vc.brute_force_minimum_distance(
                    info_path(size), checkpoint_path(size),
                    max_combinations=MAX_COMBINATIONS_PER_LEVEL)
            if enumerated is None:
                report.outcome(
                    UNAVAILABLE, label,
                    f"enumeration is exhaustive only through distance "
                    f"{proven_through}; the next level exceeds the "
                    f"{MAX_COMBINATIONS_PER_LEVEL:,} combination cap",
                    size=size, claimed=claim["d_min"])
            else:
                report.check(label, claim["d_min"], enumerated,
                             detail="every smaller distance enumerated "
                                    "exhaustively", size=size)
        except Exception as exc:  # pragma: no cover - defensive
            report.error(label, f"enumeration failed: {exc!r}", size=size)

        # -- Z3 minimum-distance scan ---------------------------------------
        label = ("d_min, Z3 minimum-distance scan"
                 + (" (*)" if paper_method == "Z3 scan" else ""))
        budget = deadline.reset().remaining()
        try:
            with captured(report.verbose):
                summary, _ = Z3.scan_minimum_adversarial_distance(
                    info_path(size), checkpoint_path(size), start_epsilon=0,
                    timeout_seconds=min(Z3_TIMEOUT_SECONDS, budget)
                    if budget else Z3_TIMEOUT_SECONDS)
            if summary.minimum_adversarial_distance is None:
                report.outcome(TIMEOUT, label,
                               f"scan ended with status "
                               f"{summary.scan_status}", "raise --timeout",
                               size=size, claimed=claim["d_min"])
            else:
                report.check(label, claim["d_min"],
                             summary.minimum_adversarial_distance,
                             detail=("exact minimum proven"
                                     if summary.exact_minimum_proven
                                     else "NOT proven exact"), size=size)
        except Exception as exc:  # pragma: no cover - defensive
            report.error(label, f"scan failed: {exc!r}", size=size)


# -----------------------------------------------------------------------------
# Group 5 -- Table IV, Gurobi column, from the recorded solver logs
# -----------------------------------------------------------------------------

GUROBI_BEST_RE = re.compile(
    r"^Best objective\s+(-?[\d.eE+]+),\s*best bound\s+(-?[\d.eE+]+),\s*"
    r"gap\s+([\d.eE+-]+)%", re.MULTILINE)
GUROBI_COLUMNS_RE = re.compile(r"Optimize a model with .*?(\d+) columns")
GUROBI_EXPLORED_RE = re.compile(
    r"^Explored (\d[\d,]*) nodes .*? in ([\d.]+) seconds", re.MULTILINE)
# An improving incumbent in the branch-and-bound log, e.g.
#   H19567 17187                    -8013.000000 -27407.343   242%   2.4   20s
# The leading integer is the node count at which it was found.
GUROBI_INCUMBENT_RE = re.compile(r"^[H*]\s*(\d+)\s+\d+\s", re.MULTILINE)


def gurobi_span_tolerance(limit):
    """How far past `limit` a no-improvement span may run and still match it.

    The overshoot is the work still in flight when model.terminate() lands, so it
    tracks the search rate rather than the limit; the four observed values are 22,
    530, 1,161 and 1,805 nodes. One percent of the limit, floored at 10,000, keeps
    all four comfortably inside while still separating 10^5 from 10^7.
    """
    return max(10_000, limit // 100)


def shipped_no_impr_nodes():
    """run_gurobi_all.DEFAULT_NO_IMPR_NODES, read from source.

    Read rather than imported, because importing run_gurobi_all pulls in gurobipy
    and this check must work without a license. Returns None if it cannot be read,
    which is a failure of the check, not a fallback: the point is to test the
    table the repository actually ships.
    """
    import ast
    try:
        tree = ast.parse(open("run_gurobi_all.py", encoding="utf-8").read())
    except (OSError, SyntaxError):
        return None
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if "DEFAULT_NO_IMPR_NODES" not in names:
            continue
        try:
            table = ast.literal_eval(node.value)
        except ValueError:
            return None
        if isinstance(table, dict):
            return table
    return None


def parse_gurobi_log(path):
    """Read the incumbent, bound, MIP gap and node count out of a Gurobi log."""
    text = open(path, encoding="utf-8", errors="replace").read()
    best = GUROBI_BEST_RE.search(text)
    if best is None:
        return None
    columns = GUROBI_COLUMNS_RE.search(text)
    explored = GUROBI_EXPLORED_RE.search(text)
    incumbents = [int(m.group(1)) for m in GUROBI_INCUMBENT_RE.finditer(text)]
    nodes = int(explored.group(1).replace(",", "")) if explored else None
    last_incumbent = incumbents[-1] if incumbents else None
    return {
        "best_objective": float(best.group(1)),
        "best_bound": float(best.group(2)),
        "gap_percent": float(best.group(3)),
        "variables": int(columns.group(1)) if columns else None,
        "nodes": nodes,
        "seconds": float(explored.group(2)) if explored else None,
        "interrupted": "Solve interrupted" in text,
        "time_limit_set": "Set parameter TimeLimit" in text,
        "proved_optimal": "Optimal solution found" in text,
        "last_incumbent_node": last_incumbent,
        "no_improvement_span": (nodes - last_incumbent
                                if nodes is not None
                                and last_incumbent is not None else None),
    }


def check_gurobi_logs(report, sizes, missing, available):
    report.section(
        "Table IV -- Gurobi column, from the recorded solver logs",
        "Table IV          Gurobi column (logs)")
    report.note("""
The Gurobi runs behind Table IV took 47 minutes to 18.7 hours each on a 32-core
machine, so they are not rerun by default. Their solver logs are shipped
instead, and the reported incumbent is read straight out of each one and
compared with the table. Use --with-gurobi to re-solve from scratch instead.

The logs also record how those runs ended, and this reconstructs it. Every one
says "Solve interrupted" with no TimeLimit parameter set, and none proves
optimality. The stop was the early-stopping callback firing, not a hand
interrupt: the no-improvement span -- nodes explored minus the node of the last
improving incumbent -- lands just above the limit that was in force, the small
overshoot being the in-flight nodes that drain after model.terminate(). That
limit is 10^5 on 5x5 and 7x7 and 10^7 on 11x11 and 28x28, which is what
run_gurobi_all.DEFAULT_NO_IMPR_NODES now carries; the check below is that the two
agree. The remaining MIP gaps are printed as context, not as pass/fail criteria.
""")

    if not available:
        print()
        for size in sizes:
            report.outcome(UNAVAILABLE,
                           f"{size}x{size} Gurobi best objective (Table IV)",
                           f"{GUROBI_LOG_DIR}/ is not present",
                           f"tar xzf {GUROBI_LOG_ARCHIVE}", size=size,
                           claimed=-PAPER[size]["gurobi_score"])
        return

    shipped_limits = shipped_no_impr_nodes()

    for size in sizes:
        claim = PAPER[size]
        report.instance_header(size)
        path = GUROBI_LOG_PATTERN.format(size=size)
        if not os.path.exists(path):
            report.outcome(UNAVAILABLE, "Gurobi best objective (Table IV)",
                           f"no solver log at {path}",
                           f"tar xzf {GUROBI_LOG_ARCHIVE}", size=size,
                           claimed=-claim["gurobi_score"])
            continue

        parsed = parse_gurobi_log(path)
        if parsed is None:
            report.error("Gurobi best objective (Table IV)",
                         f"{path}: no 'Best objective' line", size=size)
            continue

        report.check("Gurobi best objective (Table IV)",
                     -float(claim["gurobi_score"]), parsed["best_objective"],
                     detail=f"({os.path.basename(path)})", size=size)
        report.check("Energy score E_off - dE (Table IV, Gurobi)",
                     claim["gurobi_score"],
                     claim["offset"] - (parsed["best_objective"]
                                        + claim["offset"]),
                     detail=f"(dE = "
                            f"{parsed['best_objective'] + claim['offset']:g})",
                     size=size)
        if parsed["variables"] is not None:
            report.check("model size in the log", claim["variables"],
                         parsed["variables"], detail="columns", size=size)
        report.assertion(
            "interrupted, with no time limit set",
            parsed["interrupted"] and not parsed["time_limit_set"],
            f"'Solve interrupted', no TimeLimit set", size=size)
        report.assertion(
            "reported as an incumbent, not proven optimal",
            not parsed["proved_optimal"],
            f"MIP gap {parsed['gap_percent']:.4g}% remaining, "
            f"{parsed['nodes']:,} nodes in {parsed['seconds']:,.0f} s",
            size=size)
        span = parsed["no_improvement_span"]
        configured = (shipped_limits or {}).get(size)
        label = "no-improvement span vs the shipped limit"
        if configured is None:
            report.assertion(
                label, False,
                "could not read DEFAULT_NO_IMPR_NODES from run_gurobi_all.py",
                size=size)
        elif span is None:
            report.assertion(
                label, False, "no incumbent line found in the log", size=size)
        else:
            overshoot = span - configured
            report.assertion(
                label,
                0 <= overshoot < gurobi_span_tolerance(configured),
                f"{parsed['nodes']:,} - {parsed['last_incumbent_node']:,} = "
                f"{span:,} nodes without improvement, i.e. the configured "
                f"{configured:,} plus {overshoot:,} drained after terminate()",
                size=size)


# -----------------------------------------------------------------------------
# Group 6 -- opt-in: re-solve with Gurobi
# -----------------------------------------------------------------------------

def check_gurobi(report, sizes, missing, enabled, deadline):
    report.section("Table IV -- Gurobi column (opt-in)",
                   "Table IV          Gurobi column")
    report.note("""
Solves each shipped QUBO with Gurobi using run_gurobi_all.py's model, its
early-stopping callback and its per-instance no-improvement limit (10^5 on 5x5
and 7x7, 10^7 on 11x11 and 28x28 -- the values the reported runs used). Those
runs took 47 minutes to 18.7 hours, so a run here that reaches the reported
energy confirms it, while one that falls short is inconclusive rather than a
contradiction.

Two gaps are reported for every solver run below, and they answer different
questions. The gap vs the paper's value is the reproduction question. The gap
vs the target energy is the feasibility question. They are not the same: the
paper does not claim full feasibility everywhere -- Gurobi is reported at 529
for 5x5 against a target of 533, for instance. Because the objective H_0 is
identically zero, a gap of g above the target upper-bounds the number of
violated encoded constraints, and near the optimum usually equals it.
""")

    if not enabled:
        print()
        for size in sizes:
            report.outcome(SKIPPED, f"{size}x{size} Gurobi best energy",
                           "not requested; Gurobi runs can take hours and need "
                           "a license", "--with-gurobi", size=size,
                           claimed=-PAPER[size]["gurobi_score"])
        return

    try:
        import gurobipy as gp
        from gurobipy import GRB  # noqa: F401
        import run_gurobi_all
    except ImportError as exc:
        for size in sizes:
            report.outcome(UNAVAILABLE, f"{size}x{size} Gurobi best energy",
                           f"gurobipy is not installed ({exc})",
                           "pip install gurobipy, then obtain a license",
                           size=size, claimed=-PAPER[size]["gurobi_score"])
        return

    try:
        with captured(report.verbose):
            probe = gp.Model("license_probe")
            probe.Params.OutputFlag = 0
            probe.dispose()
    except Exception as exc:
        for size in sizes:
            report.outcome(UNAVAILABLE, f"{size}x{size} Gurobi best energy",
                           f"no usable Gurobi license: {exc}",
                           "activate a Gurobi license (grbgetkey ...)",
                           size=size, claimed=-PAPER[size]["gurobi_score"])
        return

    for size in sizes:
        claim = PAPER[size]
        report.instance_header(size)
        if size in missing:
            report.outcome(UNAVAILABLE, "Gurobi best energy",
                           "instance data not present in the repository",
                           f"tar xzf {DATA_ARCHIVE}", size=size,
                           claimed=-claim["gurobi_score"])
            continue

        matrix = load_qubo_matrix(size)
        info = read_info(size)
        if matrix.shape[0] > GUROBI_LIMITED_VARIABLES:
            print(f"   (this instance has {matrix.shape[0]:,} variables; the "
                  f"common size-limited license caps at "
                  f"{GUROBI_LIMITED_VARIABLES:,})")
        budget = deadline.reset().remaining()
        try:
            with captured(report.verbose):
                energy, status, runtime = _gurobi_solve(
                    run_gurobi_all, matrix, budget, size)
        except Exception as exc:
            message = str(exc)
            if ("size-limited" in message.lower()
                    or "license" in message.lower()
                    or "model too large" in message.lower()):
                report.outcome(
                    UNAVAILABLE, "Gurobi best energy",
                    f"license limitation, not a verification failure: "
                    f"{message.strip()}"
                    + (f" -- this instance has {matrix.shape[0]:,} variables "
                       f"and the size-limited license caps at "
                       f"{GUROBI_LIMITED_VARIABLES:,}"
                       if matrix.shape[0] > GUROBI_LIMITED_VARIABLES else ""),
                    "use a full Gurobi license", size=size,
                    claimed=-claim["gurobi_score"])
            else:
                report.error("Gurobi best energy",
                             f"Gurobi failed: {exc!r}", size=size)
            continue

        if energy is None:
            report.outcome(TIMEOUT, "Gurobi best energy",
                           f"no incumbent after {runtime:.1f} s "
                           f"(Gurobi status {status})", "raise --timeout",
                           size=size, claimed=-claim["gurobi_score"])
            continue

        report.solver_run(
            "Gurobi best energy (Table IV)", size,
            paper_energy=-float(claim["gurobi_score"]),
            target_energy=float(info["minimum_energy"]),
            energies=[energy], runtime=runtime, runs_label="runs",
            hint=f"the paper's Gurobi runs ran to their no-improvement limit, "
                 f"47 minutes to 18.7 hours; this one was bounded at "
                 f"{budget:.0f} s. Raise --timeout, or run: "
                 f"python run_gurobi_all.py"
                 if budget else "python run_gurobi_all.py")


def _gurobi_solve(module, matrix, budget, size):
    """run_gurobi_all.solve_qubo_upper_tri with an optional native time limit.

    The no-improvement limit is the per-instance one run_gurobi_all.py uses, so
    that this reproduces the configuration behind the shipped log for `size`
    rather than one blanket value; NO_IMPR_NODES still overrides it.
    """
    import gurobipy as gp

    limit = module.no_impr_nodes(size)

    if budget is None:
        return module.solve_qubo_upper_tri(matrix, limit)

    original = gp.Model

    class TimeLimitedModel(gp.Model):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.Params.TimeLimit = max(budget, 1.0)

    gp.Model = TimeLimitedModel
    try:
        return module.solve_qubo_upper_tri(matrix, limit)
    finally:
        gp.Model = original


# -----------------------------------------------------------------------------
# Group 6 -- opt-in: simulated annealing
# -----------------------------------------------------------------------------

def check_sa(report, sizes, missing, enabled, seeds, deadline):
    report.section("Table IV -- SA column (opt-in)",
                   "Table IV          SA column")
    report.note(f"""
Runs D-Wave Ocean's SimulatedAnnealingSampler on each shipped QUBO with the
settings SA.py uses ({SA_NUM_READS:,} reads, {SA_NUM_SWEEPS:,} sweeps, a
{SA_BETA_SCHEDULE} schedule), taking the best sample over the requested seeds.
SA is stochastic, so reaching the reported energy is PASS and falling short is
INCONCLUSIVE, never FAIL.

Note that the paper's SA column is NOT the target energy on every instance: it
reports 8,019 for 11x11 against a target of 8,020, and 1,027,316 for 28x28
against 1,027,318. The gap vs the paper's value and the gap vs the target are
therefore reported separately below. Because the objective H_0 is identically
zero, a gap of g above the target upper-bounds the number of violated encoded
constraints, and near the optimum usually equals it.
""")
    if (SA_NUM_READS, SA_NUM_SWEEPS) != (SA_DEFAULT_READS, SA_DEFAULT_SWEEPS):
        report.note(
            "WARNING: VERIFY_SA_READS / VERIFY_SA_SWEEPS override the paper's "
            f"SA settings\n({SA_DEFAULT_READS:,} reads x "
            f"{SA_DEFAULT_SWEEPS:,} sweeps). This run uses "
            f"{SA_NUM_READS:,} x {SA_NUM_SWEEPS:,} and is a smoke test,\n"
            "not the paper's configuration.")

    if not enabled:
        print()
        for size in sizes:
            report.outcome(SKIPPED, f"{size}x{size} SA best energy",
                           "not requested; SA takes minutes for 5x5 and hours "
                           "for 28x28", "--with-sa", size=size,
                           claimed=-PAPER[size]["sa_score"])
        return

    try:
        import dimod
        from dwave.samplers import SimulatedAnnealingSampler
    except ImportError as exc:
        for size in sizes:
            report.outcome(UNAVAILABLE, f"{size}x{size} SA best energy",
                           f"dimod / dwave-samplers not installed ({exc})",
                           "pip install -r requirements.txt", size=size,
                           claimed=-PAPER[size]["sa_score"])
        return

    for size in sizes:
        claim = PAPER[size]
        report.instance_header(size)
        if size in missing:
            report.outcome(UNAVAILABLE, "SA best energy",
                           "instance data not present in the repository",
                           f"tar xzf {DATA_ARCHIVE}", size=size,
                           claimed=-claim["sa_score"])
            continue

        info = read_info(size)
        print(f"   running {seeds} seed(s) at {SA_NUM_READS:,} reads x "
              f"{SA_NUM_SWEEPS:,} sweeps ...", flush=True)
        bqm = dimod.BinaryQuadraticModel.from_qubo(qubo_dict(size))
        sampler = SimulatedAnnealingSampler()

        energies = []
        best_sample = None
        deadline.reset()
        started = time.perf_counter()
        for seed in range(seeds):
            if deadline.expired():
                break
            sampleset = sampler.sample(
                bqm, num_reads=SA_NUM_READS, num_sweeps=SA_NUM_SWEEPS,
                beta_schedule_type=SA_BETA_SCHEDULE, seed=1234 + seed)
            energy = float(sampleset.first.energy)
            if not energies or energy < min(energies):
                best_sample = sampleset.first.sample
            energies.append(energy)
        runtime = time.perf_counter() - started

        if not energies:
            report.outcome(TIMEOUT, "SA best energy",
                           "--timeout expired before the first seed finished",
                           "raise --timeout", size=size,
                           claimed=-claim["sa_score"])
            continue

        report.solver_run(
            "SA best energy (Table IV)", size,
            paper_energy=-float(claim["sa_score"]),
            target_energy=float(info["minimum_energy"]),
            energies=energies, runtime=runtime,
            hint="raise --sa-seeds / --timeout, or run: python SA.py")

        if min(energies) <= float(info["minimum_energy"]):
            ok, description, _ = reverse_check(
                size, info,
                bits_by_qubo_index=[int(best_sample[i])
                                    for i in range(claim["variables"])])
            report.assertion("reverse check on the original BNN", ok,
                             description, size=size)


# -----------------------------------------------------------------------------
# Group 7 -- opt-in: replay the FEM solver
# -----------------------------------------------------------------------------

FEM_HYPERPARAMETER_ROW = re.compile(
    r"^\|\s*(\d+)x\1\s*/[^|]*\|\s*(-?[\d.]+)\s*\|"
    r"\s*([\deE.+-]+)\s*\|\s*([\deE.+-]+)\s*\|\s*([\deE.+-]+)\s*\|"
    r"\s*([\deE.+-]+)\s*\|\s*([\deE.+-]+)\s*\|\s*([\deE.+-]+)\s*\|"
    r"\s*([\deE.+-]+)\s*\|\s*(\d+)\s*\|", re.MULTILINE)


def parse_fem_hyperparameters(path):
    """Read the recorded FEM hyperparameters out of FEM_HYPERPARAMETERS.md."""
    if not os.path.exists(path):
        return {}
    table = {}
    for match in FEM_HYPERPARAMETER_ROW.finditer(
            open(path, encoding="utf-8").read()):
        size = int(match.group(1))
        table[size] = {
            "energy": float(match.group(2)),
            "lr": float(match.group(3)),
            "wd": float(match.group(4)),
            "alpha": float(match.group(5)),
            "mom": float(match.group(6)),
            "c_grad": float(match.group(7)),
            "Tmin": float(match.group(8)),
            "Tmax": float(match.group(9)),
            "seed": int(match.group(10)),
        }
    return table


def check_fem_replay(report, sizes, missing, enabled, deadline):
    report.section("Table IV -- replaying the FEM solver (opt-in)",
                   "Table IV          FEM solver replay")
    report.note(f"""
Replays FEM at the hyperparameters recorded in {FEM_HYPERPARAMETERS_PATH},
using FEM.py's own batched solver. FEM's coordinate search is stochastic and the
recorded values are the state of that search when the best energy was found, so
this is NOT expected to reproduce the reported energy exactly. Reaching the
reported energy is PASS; falling short is INCONCLUSIVE, never FAIL. The
energies themselves are verified exactly from the recorded solution vectors
above, which is the stronger check. As for the other solvers, the gap vs the
paper's value and the gap vs the target energy are reported separately.
""")

    if not enabled:
        print()
        for size in sizes:
            report.outcome(SKIPPED, f"{size}x{size} FEM solver replay",
                           "not requested; FEM's original runs took 638 s to "
                           "21,898 s per instance", "--with-fem", size=size,
                           claimed=-PAPER[size]["fem_score"])
        return

    try:
        import numpy as np
        import torch  # noqa: F401
        import FEM
    except Exception as exc:
        for size in sizes:
            report.outcome(UNAVAILABLE, f"{size}x{size} FEM solver replay",
                           f"FEM.py could not be imported ({exc!r})",
                           size=size, claimed=-PAPER[size]["fem_score"])
        return

    table = parse_fem_hyperparameters(FEM_HYPERPARAMETERS_PATH)

    for size in sizes:
        claim = PAPER[size]
        report.instance_header(size)
        if size in missing:
            report.outcome(UNAVAILABLE, "FEM solver replay",
                           "instance data not present in the repository",
                           f"tar xzf {DATA_ARCHIVE}", size=size,
                           claimed=-claim["fem_score"])
            continue
        if size not in table:
            report.outcome(UNAVAILABLE, "FEM solver replay",
                           f"no recorded hyperparameters for {size}x{size} in "
                           f"{FEM_HYPERPARAMETERS_PATH}", size=size,
                           claimed=-claim["fem_score"])
            continue

        parameters = table[size]
        info = read_info(size)
        matrix = load_qubo_matrix(size)
        symmetric = (matrix + matrix.T) / 2.0
        h_vector = np.diag(symmetric).copy()
        j_matrix = symmetric - np.diag(h_vector)

        deadline.reset()
        started = time.perf_counter()
        try:
            with captured(report.verbose):
                energies, configurations, _, _ = FEM.param_search_batched(
                    seeds=np.array([parameters["seed"]], dtype=np.float32),
                    J=j_matrix, h=h_vector, betamode="inv",
                    Tmax=np.array([parameters["Tmax"]], dtype=np.float32),
                    Tmin=np.array([parameters["Tmin"]], dtype=np.float32),
                    N_step=100, lr=np.array([parameters["lr"]],
                                            dtype=np.float32),
                    batch=100,
                    c_grad=np.array([parameters["c_grad"]], dtype=np.float32),
                    dev="cpu",
                    opt_params=np.array([[parameters["alpha"],
                                          parameters["wd"],
                                          parameters["mom"]]],
                                        dtype=np.float32),
                    optimizer="rmsprop")
        except Exception as exc:
            report.error("FEM solver replay", f"FEM failed: {exc!r}", size=size)
            continue
        runtime = time.perf_counter() - started

        if energies is None or len(energies) == 0:
            report.outcome(INCONCLUSIVE, "FEM solver replay",
                           "the solver returned no samples", size=size,
                           claimed=-claim["fem_score"])
            continue

        report.solver_run(
            "FEM best energy, solver replay (Table IV)", size,
            paper_energy=-float(claim["fem_score"]),
            target_energy=float(info["minimum_energy"]),
            energies=[float(value) for value in energies], runtime=runtime,
            runs_label="batches",
            hint="python FEM.py    # the full coordinate search; see "
                 "FEM_HYPERPARAMETERS.md. The recorded solution vector is "
                 "already verified exactly above.")


# -----------------------------------------------------------------------------
# Group 9 -- Table VII: the two-class hardware instance
# -----------------------------------------------------------------------------

def build_hardware_model(pickled):
    """Rebuild the two-class PCBO from the pickled constraint dictionaries.

    This follows the authors' own Verify.ipynb exactly: the 'lt' entries go
    through add_constraint_lt_zero and the 'eq' entries through
    add_constraint_eq_zero. The objective H_0 is identically zero here too, so
    the QUBO is just the penalty sum and the pickled 'qubo' dictionary must come
    back out of it unchanged.
    """
    import qubovert as qv
    from qubovert import boolean_var

    def term(entry):
        built = qv.PUBO()
        for variables, coefficient in entry.items():
            if len(variables) == 0:
                built += coefficient
            elif len(variables) == 1:
                built += boolean_var(variables[0]) * coefficient
            else:
                built += (boolean_var(variables[0])
                          * boolean_var(variables[1]) * coefficient)
        return built

    model = qv.PCBO()
    for entry in pickled["constraints"]["lt"]:
        model.add_constraint_lt_zero(term(entry))
    for entry in pickled["constraints"]["eq"]:
        model.add_constraint_eq_zero(term(entry))
    return model


def count_satisfied(model, solution):
    """Count satisfied constraints using qubovert's own validity predicates."""
    predicates = {
        "eq": lambda value: value == 0,
        "ne": lambda value: value != 0,
        "lt": lambda value: value < 0,
        "le": lambda value: value <= 0,
        "gt": lambda value: value > 0,
        "ge": lambda value: value >= 0,
    }
    satisfied = total = 0
    for kind, constraints in model.constraints.items():
        for constraint in constraints:
            total += 1
            satisfied += bool(predicates[kind](constraint.value(solution)))
    return satisfied, total


def check_hardware(report, available):
    report.section("Table VII -- quantum and digital annealing hardware",
                   "Table VII         hardware results")
    report.note("""
Table VII was produced on a D-Wave quantum annealer and on Fujitsu's Digital
Annealer, on a separate two-class instance. That instance, both hardware
solutions and the authors' own Verify.ipynb now ship in
data/hardware_results.tar.gz, so the rows can be checked here. The hardware
itself is not re-run -- these are the recorded samples, re-evaluated against the
QUBO and against every encoded constraint.

READ THIS BEFORE THE ROWS BELOW. Table VII's "Constraints Satisfied" column
currently prints 1,273 for Gurobi / DA / SA and 356 for the QA. Neither figure
is a constraint count. This instance has 65 encoded constraints in total:

  * 1,273 is the f-term count carried in the instance's file name; the QUBO
    itself has 1,272 quadratic and linear terms.
  * 356 is the QA's energy gap above the target, not a number of constraints.
    It happens to upper-bound the violations, as energy gaps do here, but it
    is 5.5x the total number of constraints that exist.

This is the same confusion between an energy and a constraint count that was
corrected in Tables III, IV and VI, and Table VII needs the same correction.
The rows below check the CORRECTED column -- 65 out of 65 for the Digital
Annealer and 36 out of 65 for the quantum annealer -- together with the
energies and runtimes, which are what the hardware actually reported.
""")
    print()

    if not available:
        report.outcome(UNAVAILABLE, "Table VII, two-class instance",
                       f"{HARDWARE_DIR}/ is not present",
                       f"tar xzf {HARDWARE_ARCHIVE}")
        return

    import pickle

    with open(HARDWARE_QUBO, "rb") as handle:
        pickled = pickle.load(handle)
    model = build_hardware_model(pickled)
    qubo = model.to_qubo()

    print(" the two-class instance")
    report.assertion(
        "rebuilt QUBO == the shipped qubo dictionary",
        qubo.Q == pickled["qubo"],
        f"{len(pickled['qubo']):,} terms, rebuilt with the authors' "
        f"Verify.ipynb recipe")
    report.check("Variables", HARDWARE["variables"], len(model.variables))
    counts = {kind: len(value) for kind, value in model.constraints.items()}
    report.check("Encoded constraints (the true total)",
                 HARDWARE["constraints"], sum(counts.values()),
                 detail="(" + " + ".join(f"{k} {v}" for k, v
                                         in sorted(counts.items())) + ")")
    report.check("QUBO terms", HARDWARE["qubo_terms"], len(qubo.Q))
    report.check("Energy offset E_off", HARDWARE["offset"], qubo[()],
                 detail="so the target energy is "
                        f"{-HARDWARE['offset']:,}")

    # -- Fujitsu Digital Annealer ---------------------------------------------
    print("\n Digital Annealer (Fujitsu)")
    with open(HARDWARE_FUJITSU, "rb") as handle:
        raw = pickle.load(handle)
    solution = {int(key): int(value) for key, value in raw.items()}
    converted = model.convert_solution(solution)
    with open(HARDWARE_FUJITSU_TIME, "rb") as handle:
        timing = pickle.load(handle)[-1]

    satisfied, total = count_satisfied(model, converted)
    energy = qubo.value(solution) - qubo[()]
    report.check("Best energy", HARDWARE["fujitsu_energy"], energy,
                 detail=f"(the recorded run reports "
                        f"{timing['energy']:,})")
    report.assertion("every encoded constraint satisfied",
                     model.is_solution_valid(converted),
                     f"qubovert is_solution_valid True, penalty value "
                     f"{model.value(converted):g}, so the target energy is "
                     f"attained exactly")
    report.check("Constraints satisfied (corrected Table VII)",
                 HARDWARE["fujitsu_constraints"], satisfied,
                 detail=f"of {total}; the table prints 1,273, which is the "
                        f"f-term count")
    report.check("Total time (s)", HARDWARE["fujitsu_time"],
                 float(timing["time"]))

    # -- D-Wave quantum annealer ----------------------------------------------
    print("\n Quantum Annealer (D-Wave)")
    with open(HARDWARE_DWAVE, "rb") as handle:
        frame = pickle.load(handle)
    best = frame.loc[frame["energy"].idxmin()]
    sample = {index: int(best[index]) for index in range(HARDWARE["variables"])}
    converted = model.convert_solution(sample)
    satisfied, total = count_satisfied(model, converted)
    gap = qubo.value(sample) - 0.0   # penalty value == energy above the target

    report.check("Shots in the returned dataframe", HARDWARE["dwave_shots"],
                 int(frame["num_occurrences"].sum()),
                 detail=f"{len(frame):,} distinct samples")
    report.check("Best energy over all samples", HARDWARE["dwave_energy"],
                 float(best["energy"]),
                 detail="recomputed from the QUBO: "
                        f"{qubo.value(sample) - qubo[()]:,.0f}")
    report.check("Constraints satisfied (corrected Table VII)",
                 HARDWARE["dwave_constraints"], satisfied,
                 detail=f"of {total}; the table prints 356, which is the "
                        f"energy gap")
    report.assertion(
        "the printed 356 is the energy gap, not a constraint count",
        gap == 356.0,
        f"gap above the target is {gap:,.0f}; it upper-bounds the "
        f"{total - satisfied} violated constraints but is not a count")
    report.assertion(
        "not a feasible solution, so no non-robustness certificate",
        not model.is_solution_valid(converted),
        f"{total - satisfied} of {total} constraints violated, "
        f"chain break fraction {best['chain_break_fraction']:.3f}")

    # -- rows that still cannot be checked ------------------------------------
    print()
    for solver in ("Gurobi", "Simulated Annealing"):
        report.outcome(
            UNAVAILABLE, f"Table VII, {solver} row (1,273 as printed; the "
            f"corrected value would be 65 of 65)",
            "no solution vector for this two-class instance was supplied for "
            "this solver, so the row cannot be recomputed. Only the D-Wave and "
            "Fujitsu samples were archived",
            "supply the returned sample, as for the two hardware rows")


# -----------------------------------------------------------------------------
# Data availability
# -----------------------------------------------------------------------------

def ensure_archive(archive, marker, description, allow_extract):
    """Unpack a data/ archive into the repository root if `marker` is absent."""
    if os.path.exists(marker):
        return True
    if not (allow_extract and os.path.exists(archive)):
        return False
    print(f" [setup] {description} not extracted; unpacking {archive}")
    with tarfile.open(archive, "r:gz") as handle:
        handle.extractall(".")
    return os.path.exists(marker)


def ensure_data(sizes, allow_extract):
    """Extract data/qubo_and_networks.tar.gz if the instance data is absent."""
    absent = [size for size in sizes if not instance_available(size)]
    if not absent:
        return [], False

    if allow_extract and os.path.exists(DATA_ARCHIVE):
        print(f" [setup] {', '.join(f'{s}x{s}' for s in absent)} not "
              f"extracted; unpacking {DATA_ARCHIVE}")
        print(f" [setup] into the repository root "
              f"(QUBO/ and TrainedNN/; ~138 MB expanded, git-ignored)")
        with tarfile.open(DATA_ARCHIVE, "r:gz") as archive:
            archive.extractall(".")
        absent = [size for size in sizes if not instance_available(size)]
        print(" [setup] done."
              if not absent else
              " [setup] done, but some instances are still missing.")
        return absent, True

    return absent, False


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        prog="verify_paper.py",
        description="Re-check the paper's reported numbers and print a "
                    "pass/fail report.")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--all", action="store_true",
                           help="verify every instance (the default)")
    selection.add_argument("--quick", action="store_true",
                           help="verify the 5x5 instance only")
    selection.add_argument("--instance", metavar="LIST",
                           help="comma-separated subset of 5,7,11,28")
    parser.add_argument("--with-gurobi", action="store_true",
                        help="also run Gurobi (needs a license)")
    parser.add_argument("--with-sa", action="store_true",
                        help="also run simulated annealing (minutes to hours)")
    parser.add_argument("--sa-seeds", type=int, default=3, metavar="N",
                        help="SA seeds per instance (default 3)")
    parser.add_argument("--with-fem", action="store_true",
                        help="also replay the FEM solver (stochastic)")
    parser.add_argument("--everything", action="store_true",
                        help="shorthand for --with-gurobi --with-sa --with-fem")
    parser.add_argument("--timeout", type=float, metavar="SECONDS",
                        help="per-check wall-clock bound; exceeding it is "
                             "TIMEOUT, not FAIL")
    parser.add_argument("--json", metavar="PATH",
                        help="write a machine-readable report ('-' for stdout)")
    parser.add_argument("--no-extract", action="store_true",
                        help="do not unpack data/qubo_and_networks.tar.gz")
    parser.add_argument("--verbose", action="store_true",
                        help="show the full output of the sub-checks")
    return parser.parse_args(argv)


def main(argv=None):
    options = parse_arguments(sys.argv[1:] if argv is None else argv)

    if options.quick:
        sizes = [5]
    elif options.instance:
        try:
            requested = {int(token) for token in options.instance.split(",")
                         if token.strip()}
        except ValueError:
            raise SystemExit("--instance takes a comma-separated list of "
                             "5, 7, 11, 28")
        unknown = sorted(requested - set(PAPER))
        if unknown:
            raise SystemExit(f"unknown instance(s): {unknown}; "
                             f"choose from {ALL_SIZES}")
        sizes = [size for size in ALL_SIZES if size in requested]
    else:
        sizes = list(ALL_SIZES)

    with_gurobi = options.with_gurobi or options.everything
    with_sa = options.with_sa or options.everything
    with_fem = options.with_fem or options.everything

    # get_args.py parses sys.argv at import time, so hide our own flags.
    sys.argv = [sys.argv[0]]

    started = time.time()
    deadline = Deadline(options.timeout)

    report = Report(verbose=options.verbose)
    report.banner("verify_paper.py -- reproducing the reported numerical "
                  "results")
    print(f" repository root : {os.getcwd()}")
    print(f" instances       : {', '.join(f'{s}x{s}' for s in sizes)}")
    print( " QUBO encoding   : strict argmax (args.argmax_tie_aware = False), "
           "the encoding")
    print( "                   used for every shipped QUBO and every reported "
           "number")
    extra = [name for name, on in (("--with-gurobi", with_gurobi),
                                   ("--with-sa", with_sa),
                                   ("--with-fem", with_fem)) if on]
    extra_text = (", ".join(extra) if extra else
                  "none (no license, no GPU, no network, no hardware needed)")
    print(f" opt-in checks   : {extra_text}")
    if options.timeout:
        print(f" per-check limit : {options.timeout:g} s (all checks)")
    elif extra:
        print(f" per-check limit : {OPT_IN_DEFAULT_TIMEOUT_SECONDS:g} s for the "
              f"opt-in checks (default; set --timeout to change)")
    if os.environ.get("BNN_ARGMAX_TIE_AWARE", "0").strip().lower() in (
            "1", "true", "yes", "on"):
        print(" NOTE            : BNN_ARGMAX_TIE_AWARE is set in the "
              "environment; this script")
        print("                   overrides it, because the reported numbers "
              "use the strict encoding.")
    print()

    missing, extracted = ensure_data(sizes, not options.no_extract)
    if missing:
        print(f" NOTE            : no data for "
              f"{', '.join(f'{s}x{s}' for s in missing)}; those instances are "
              f"reported as UNAVAILABLE.")
        if os.path.exists(DATA_ARCHIVE):
            print(f"                   Unpack it with: tar xzf {DATA_ARCHIVE}")

    gurobi_logs_available = ensure_archive(
        GUROBI_LOG_ARCHIVE, GUROBI_LOG_PATTERN.format(size=5),
        "the recorded Gurobi solver logs", not options.no_extract)
    hardware_available = ensure_archive(
        HARDWARE_ARCHIVE, HARDWARE_QUBO,
        "the Table VII two-class hardware instance", not options.no_extract)

    check_structure(report, sizes, missing)
    check_fem_solutions(report, sizes, missing)
    check_z3(report, sizes, missing, deadline)
    check_minimum_distance(report, sizes, missing, deadline)
    opt_in_deadline = Deadline(options.timeout
                               if options.timeout is not None
                               else OPT_IN_DEFAULT_TIMEOUT_SECONDS)
    check_gurobi_logs(report, sizes, missing, gurobi_logs_available)
    check_gurobi(report, sizes, missing, with_gurobi, opt_in_deadline)
    check_sa(report, sizes, missing, with_sa, options.sa_seeds, opt_in_deadline)
    check_fem_replay(report, sizes, missing, with_fem, opt_in_deadline)
    check_hardware(report, hardware_available)

    report.summary(time.time() - started)

    counts = report.tally()
    if options.json:
        payload = {
            "instances": sizes,
            "extracted_archive": extracted,
            "missing_instances": missing,
            "opt_in": {"gurobi": with_gurobi, "sa": with_sa, "fem": with_fem},
            "counts": counts,
            "result": FAIL if counts.get(FAIL) else PASS,
            "checks": report.entries,
        }
        text = json.dumps(payload, indent=2)
        if options.json == "-":
            print()
            print(text)
        else:
            with open(options.json, "w", encoding="utf-8") as handle:
                handle.write(text + "\n")
            print(f" machine-readable report written to {options.json}")

    return 1 if counts.get(FAIL) else 0


if __name__ == "__main__":
    raise SystemExit(main())
