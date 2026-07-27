#!/usr/bin/env python3
"""width_scaling_sweep.py -- regenerate the hidden-layer width sweep.

This is the script the shipped sweep in data/width_scaling.tar.gz came from,
cleaned up to run anywhere: no absolute paths, no job queue, no assumption about
how many cores the machine has. Run it from the repository root.

    python width_scaling_sweep.py sweep --input-size 5           # 31 inputs
    python width_scaling_sweep.py sweep --input-size 28          # 1023 inputs

WHAT IT MEASURES
----------------
At each hidden width W in {7, 15, 31, 63, 127, 255}, holding the input size, the
perturbable pixel set and the perturbation budget fixed:

  1. a BNN trained at that width, from the same seed and optimizer as
     TrainingNN.py;
  2. the QUBO encoding of the robustness query for that network, built with the
     repository's own bnn_as_qubo.setup_optim_model -- variables, the true
     constraint count from the qubovert PCBO's own .constraints dictionary, the
     nonzero terms and the coefficient range;
  3. the exact SMT baseline, through Z3.py, at the instance's own epsilon, at
     the budget the QUBO actually encodes, and as a scan for the exact minimum
     adversarial distance;
  4. simulated annealing at SA.py's settings -- dwave-samplers, 2,048 reads,
     5,000 sweeps, geometric schedule -- over five independent seeds.

A FULL SWEEP TAKES MANY HOURS, AND YOU DO NOT NEED TO RUN IT
------------------------------------------------------------
Measured on the machine the shipped records came from, one sweep costs roughly:

    31-input series (--input-size 5)     67 CPU-hours, 66 of them annealing
    1023-input series (--input-size 28)  77 CPU-hours, 74 of them annealing

Almost all of that is stage 4. A single annealing seed at W = 255 of the
31-input series ran for 8.9 h, and its own 4-read calibration probe projected
15.4 h before it started. Peak memory reaches about 0.7 GB at W = 255 of the
31-input series and about 4.6 GB at W = 255 of the 1023-input series.

The point of shipping the records is that a reader does not have to pay that.
`python verify_paper.py` rebuilds every width's QUBO from the shipped checkpoint,
re-scores the stored best annealing vectors against it, and re-runs Z3, in about
a minute for the 31-input series. Use this script only to regenerate the sweep
from scratch, or to extend it to another width or input size.

The `sweep` subcommand runs the stages in order in a single process, so it needs
no scheduler. Every stage is also its own subcommand, so anything can be re-run
on its own or fanned out across cores by hand -- one process per width and seed
is embarrassingly parallel, and that is how the shipped run was produced. The
orchestration that did the fanning out was specific to the machine it ran on and
is not part of this repository.

Nothing here writes into the tracked pipeline directories. Everything goes under
--out, which defaults to width_scaling_rerun/ and is git-ignored; the shipped
records unpack to width_scaling/ and are never overwritten.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Published per-input-size settings, copied verbatim from QUBOCreator.py.
INPUT_DIM = {5: 31, 7: 63, 11: 127, 28: 1023}
PERTURB_LEN = {5: 16, 7: 32, 11: 64, 28: 256}
PERTURB_BOUND = {5: 8, 7: 32, 11: 32, 28: 128}

WIDTHS = [7, 15, 31, 63, 127, 255]
SEEDS = [1, 2, 3, 4, 5]

# QUBOCreator.py's penalty weights and objective.
LAMBDA = {
    "sum_taus": 0.1,
    "output": 1,
    "hard_constraints": 1,
    "perturbation_bound_constraint": 1,
    "epsilon": 1,
}

# What the shipped run used, per series. The 31-input series is the
# publication-grade one: 1,000 epochs and an annealing budget high enough that
# nothing was ever truncated. The 1023-input series is corroboration: 250 epochs,
# and from W = 31 upwards the one-hour budget truncated the read count, so its
# energies are bounds on what 2,048 reads would have reached, not measurements
# of it.
SERIES_DEFAULTS = {
    5: {"epochs": 1000, "eval_every": 100, "budget_seconds": 1e9},
    28: {"epochs": 250, "eval_every": 25, "budget_seconds": 3600.0},
}

# The shipped 1023-input run gave the two narrowest widths an unbounded budget
# and capped the rest at an hour per seed, which is why W = 7 and W = 15 there
# are untruncated and the wider ones are not.
UNBOUNDED_WIDTHS_1023 = (7, 15)


def peak_rss_mb() -> float:
    import resource
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kB, macOS reports bytes.
    return (usage / 1024.0 if sys.platform.startswith("linux")
            else usage / (1024.0 ** 2))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str),
                    encoding="utf-8")


def cell_dir(out: str, input_size: int, width: int) -> Path:
    return Path(out) / f"{input_size}x{input_size}" / f"w{width}"


# -----------------------------------------------------------------------------
# The network. Identical to TrainingNN.py and QUBOCreator.py except that the
# hidden width is a parameter. Exactly two named children, so that
# setup_optim_model's `len(list(model.modules())) - 2` bookkeeping is unchanged.
# -----------------------------------------------------------------------------

def build_model_classes():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.autograd import Function

    class Binarize(Function):
        clip_value = 1

        @staticmethod
        def forward(ctx, inp):
            ctx.save_for_backward(inp)
            output = inp.new(inp.size())
            output[inp >= 0] = 1
            output[inp < 0] = -1
            return output

        @staticmethod
        def backward(ctx, grad_output):
            inp = ctx.saved_tensors[0]
            clipped = inp.abs() <= Binarize.clip_value
            output = torch.zeros(inp.size()).to(grad_output.device)
            output[clipped] = 1
            output[~clipped] = 0
            return output * grad_output

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

    class Net(nn.Module):
        """Training net: softmax head, as in TrainingNN.py."""

        def __init__(self, in_dim, width):
            super().__init__()
            self.fc1 = BinaryLinear(in_dim, width)
            self.fc4 = LastLayer(width, 10)

        def forward(self, x):
            return nn.Softmax(dim=1)(self.fc4(self.fc1(x)))

    class QUBONet(nn.Module):
        """Inference net used by QUBOCreator.py: argmax head."""

        def __init__(self, in_dim, width):
            super().__init__()
            self.fc1 = BinaryLinear(in_dim, width)
            self.fc4 = LastLayer(width, 10)

        def forward(self, x):
            import torch as _torch
            return _torch.argmax(self.fc4(self.fc1(x)))

    return Net, QUBONet


def load_dataloaders(input_size: int):
    """The binarized data sets, from data/datasets.tar.gz."""
    import torch

    folder = REPO / f"Dataset/{input_size}x{input_size}"
    train_path, test_path = folder / "Train.txt", folder / "Test.txt"
    if not train_path.exists():
        raise SystemExit(
            f"{train_path} not found. Unpack the data sets first:\n"
            f"    tar xzf data/datasets.tar.gz")
    return (torch.load(train_path, weights_only=False),
            torch.load(test_path, weights_only=False))


# -----------------------------------------------------------------------------
# Stage 1: training
# -----------------------------------------------------------------------------

def cmd_train(a) -> None:
    import torch
    import torch.nn.functional as F
    import torch.optim as optim
    from utils import to_spin

    torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads)

    Net, _ = build_model_classes()
    in_dim = INPUT_DIM[a.input_size]
    train_dl, test_dl = load_dataloaders(a.input_size)

    net = Net(in_dim, a.width)
    optimizer = optim.Adadelta(net.parameters(), lr=a.lr)

    def evaluate(loader):
        net.eval()
        correct = total = 0
        loss_sum = 0.0
        with torch.no_grad():
            for data, target in loader:
                data = to_spin(data)
                target = target.type(torch.LongTensor)
                out = net(data)
                loss_sum += F.nll_loss(out, target, reduction="sum").item()
                pred = out.data.max(1, keepdim=True)[1]
                correct += int(pred.eq(target.data.view_as(pred)).sum())
                total += int(target.numel())
        return correct, total, loss_sum / max(total, 1)

    started = time.perf_counter()
    history = []
    for epoch in range(1, a.epochs + 1):
        net.train()
        for data, target in train_dl:
            data = to_spin(data)
            target = target.type(torch.LongTensor)
            optimizer.zero_grad()
            loss = F.nll_loss(net(data), target)
            loss.backward()
            optimizer.step()
        if epoch % a.eval_every == 0 or epoch == a.epochs:
            correct, total, loss_value = evaluate(test_dl)
            history.append({"epoch": epoch, "test_correct": correct,
                            "test_total": total, "test_loss": loss_value,
                            "test_acc": correct / total})
    elapsed = time.perf_counter() - started

    train_c, train_t, train_loss = evaluate(train_dl)
    test_c, test_t, test_loss = evaluate(test_dl)

    out_dir = cell_dir(a.out, a.input_size, a.width)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), out_dir / "model.pth")
    write_json(out_dir / "training.json", {
        "input_size": a.input_size, "input_dim": in_dim, "width": a.width,
        "epochs": a.epochs, "lr": a.lr, "seed": a.seed,
        "optimizer": "adadelta", "batch_size": 64,
        "train_accuracy": train_c / train_t, "train_correct": train_c,
        "train_total": train_t, "train_loss": train_loss,
        "test_accuracy": test_c / test_t, "test_correct": test_c,
        "test_total": test_t, "test_loss": test_loss,
        "train_wall_seconds": elapsed, "history": history,
    })
    print(f"[train] {a.input_size}x{a.input_size} w={a.width} "
          f"train={train_c / train_t:.4f} test={test_c / test_t:.4f} "
          f"({elapsed:.0f}s)", flush=True)


# -----------------------------------------------------------------------------
# Stage 2: pick the instance every width is verified on, and write its Info.txt
# -----------------------------------------------------------------------------

def perturbable_pixels(train_dl, input_size: int, count: int):
    """QUBOCreator.py's rule: the pixels with the smallest mean magnitude."""
    flat = train_dl.dataset.tensors[0][:, 0:input_size * input_size]
    return list(flat.mean(axis=0).abs().topk(
        min(count, flat.shape[1]), largest=False).indices.numpy())


def cmd_instances(a) -> None:
    """Choose one image classified correctly by EVERY width.

    QUBOCreator.py takes the first image the network classifies correctly, which
    would give a different instance at each width and confound the comparison.
    Requiring the same image everywhere makes the width the only variable.
    """
    import numpy as np
    import torch
    from utils import to_spin

    torch.set_num_threads(1)
    _, QUBONet = build_model_classes()
    in_dim = INPUT_DIM[a.input_size]
    train_dl, _ = load_dataloaders(a.input_size)
    pixels = perturbable_pixels(train_dl, a.input_size,
                               PERTURB_LEN[a.input_size])
    epsilon = PERTURB_BOUND[a.input_size]

    nets = {}
    for width in a.widths:
        checkpoint = cell_dir(a.out, a.input_size, width) / "model.pth"
        if not checkpoint.exists():
            raise SystemExit(f"{checkpoint} not found; run `train` first.")
        net = QUBONet(in_dim, width)
        net.load_state_dict(torch.load(checkpoint, weights_only=True,
                                       map_location="cpu"))
        net.eval()
        nets[width] = net

    chosen = None
    per_width_first = {w: None for w in a.widths}
    scanned = min(len(train_dl.dataset), a.max_scan)
    with torch.no_grad():
        for index in range(scanned):
            x_bool, target = train_dl.dataset[index]
            batch = torch.Tensor(np.array([np.array(to_spin(x_bool))]))
            correct = {}
            for width, net in nets.items():
                ok = int(net(batch)) == int(target)
                correct[width] = ok
                if ok and per_width_first[width] is None:
                    per_width_first[width] = index
            if chosen is None and all(correct.values()):
                chosen = index
            if chosen is not None and all(v is not None
                                          for v in per_width_first.values()):
                break

    if chosen is None:
        raise SystemExit(f"No image in the first {scanned} is classified "
                         f"correctly by every width in {a.widths}.")

    x_bool, target = train_dl.dataset[chosen]
    boolean = [int(round(float(value))) for value in x_bool.tolist()]

    write_json(Path(a.out) / f"{a.input_size}x{a.input_size}" / "instance.json",
               {"input_size": a.input_size, "input_dim": in_dim,
                "chosen_image_index": chosen,
                "first_correct_index_per_width": per_width_first,
                "target_label": int(target), "perturbable_count": len(pixels),
                "epsilon": epsilon, "budget_ratio": epsilon / len(pixels),
                "pixels": [int(p) for p in pixels]})

    tensor_text = "tensor([" + ", ".join(f"{v}." for v in boolean) + "])"
    for width in a.widths:
        folder = cell_dir(a.out, a.input_size, width)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "Info.txt").write_text(
            f"Total Pixels to perturb : {len(pixels)} \n"
            f"Pixels perturbed : {[int(p) for p in pixels]} \n"
            f"Input to perturb : {tensor_text} \n"
            f"Label to perturb : {int(target)} \n"
            f"Epsilon : {epsilon} \n"
            f"Perturbation Bound Constriant included : True \n",
            encoding="utf-8")
    print(f"[instances] {a.input_size}x{a.input_size}: image {chosen}, "
          f"label {int(target)}, {len(pixels)} perturbable, epsilon {epsilon}",
          flush=True)


# -----------------------------------------------------------------------------
# Stage 3: the exact SMT baseline
# -----------------------------------------------------------------------------

def cmd_z3(a) -> None:
    from dataclasses import asdict
    import Z3 as z3mod

    folder = cell_dir(a.out, a.input_size, a.width)
    info_path, checkpoint = folder / "Info.txt", folder / "model.pth"
    payload = {"input_size": a.input_size, "width": a.width}

    # (a) the published budget, sum(changed) <= epsilon.
    result = z3mod.verify_instance(info_path, checkpoint,
                                   timeout_seconds=a.timeout)
    payload["at_epsilon"] = asdict(result)

    # (b) the budget the QUBO actually encodes. add_constraint_lt_zero gives
    #     sum(taus) < epsilon, i.e. <= epsilon - 1, so this is the radius the
    #     QUBO's target energy corresponds to.
    if result.epsilon >= 1:
        payload["at_qubo_budget"] = asdict(z3mod.verify_instance(
            info_path, checkpoint, timeout_seconds=a.timeout,
            epsilon_override=result.epsilon - 1))

    # (c) the exact minimum adversarial distance.
    if a.scan:
        started = time.perf_counter()
        try:
            summary, per_radius = z3mod.scan_minimum_adversarial_distance(
                info_path, checkpoint, start_epsilon=0,
                max_epsilon=a.scan_max or None,
                timeout_seconds=a.scan_timeout)
            payload["scan"] = asdict(summary)
            payload["scan_per_radius"] = [
                {"epsilon": r.epsilon, "status": r.status,
                 "runtime_seconds": r.runtime_seconds,
                 "hamming_distance": r.hamming_distance} for r in per_radius]
        except Exception as exc:  # a scan that dies must not lose the rest
            payload["scan_error"] = repr(exc)
        payload["scan_wall_seconds"] = time.perf_counter() - started

    payload["peak_rss_mb"] = peak_rss_mb()
    write_json(folder / "z3.json", payload)
    print(f"[z3] {a.input_size}x{a.input_size} w={a.width} {result.status} "
          f"in {result.runtime_seconds:.4f}s", flush=True)


# -----------------------------------------------------------------------------
# Stage 4: the QUBO encoding and its size
# -----------------------------------------------------------------------------

def cmd_qubo(a) -> None:
    import numpy as np
    import torch
    from argparse import Namespace
    from utils import to_spin
    from bnn_as_qubo import setup_optim_model

    torch.set_num_threads(1)
    _, QUBONet = build_model_classes()
    in_dim = INPUT_DIM[a.input_size]
    folder = cell_dir(a.out, a.input_size, a.width)
    meta = json.loads(
        (Path(a.out) / f"{a.input_size}x{a.input_size}" /
         "instance.json").read_text())

    train_dl, _ = load_dataloaders(a.input_size)
    x_bool, target = train_dl.dataset[meta["chosen_image_index"]]
    spin = to_spin(x_bool)

    net = QUBONet(in_dim, a.width)
    net.load_state_dict(torch.load(folder / "model.pth", weights_only=True,
                                   map_location="cpu"))
    net.eval()

    args = Namespace(
        LAMBDA=LAMBDA, epsilon=meta["epsilon"], objective="zero",
        include_perturbation_bound_constraint=True,
        pixels_to_perturb=list(meta["pixels"]), argmax_tie_aware=False,
    )

    started = time.perf_counter()
    hamiltonian, ordered_variables = setup_optim_model(spin, target, net, args)
    build_seconds = time.perf_counter() - started

    started = time.perf_counter()
    qubo_model = hamiltonian.to_qubo()
    to_qubo_seconds = time.perf_counter() - started
    Q = qubo_model.Q
    offset = qubo_model[()]

    counts = {kind: len(v) for kind, v in hamiltonian.constraints.items()}
    values = np.fromiter(Q.values(), dtype=float, count=len(Q))
    nonzero = np.abs(values[values != 0])
    linear = sum(1 for key in Q if len(key) == 1)
    quadratic = sum(1 for key in Q if len(key) == 2)

    # QUBO index -> PCBO variable name. ordered_variables is what QUBOCreator.py
    # writes to Variables.json and what verify_paper.py decodes a sample with,
    # so it is the authoritative index order. H.mapping is recorded as an
    # independent cross-check.
    n_vars = len(ordered_variables)
    reverse = {index: str(name) for index, name in enumerate(ordered_variables)}
    pcbo_mapping = {str(k): int(v) for k, v in hamiltonian.mapping.items()}
    mapping_consistent = all(pcbo_mapping.get(str(name)) == index
                             for index, name in enumerate(ordered_variables))

    write_json(folder / "qubo.json", {
        "input_size": a.input_size, "input_dim": in_dim, "width": a.width,
        "pcbo_variables": len(hamiltonian.variables),
        "qubo_variables": n_vars,
        "qubo_num_binary_variables": int(
            getattr(qubo_model, "num_binary_variables", n_vars)),
        "mapping_consistent": bool(mapping_consistent),
        "constraints_total": int(sum(counts.values())),
        "constraints_by_type": counts,
        "qubo_terms_nonzero": int(len(Q)),
        "qubo_linear_terms": linear,
        "qubo_quadratic_terms": quadratic,
        "qubo_density": (2.0 * quadratic / (n_vars ** 2) if n_vars > 1
                         else None),
        "energy_offset": float(offset),
        "target_energy": -float(offset),
        "coef_min": float(values.min()), "coef_max": float(values.max()),
        "coef_abs_min_nonzero": float(nonzero.min()) if nonzero.size else None,
        "coef_abs_max": float(nonzero.max()) if nonzero.size else None,
        "coef_dynamic_range": (float(nonzero.max() / nonzero.min())
                               if nonzero.size else None),
        "build_seconds": build_seconds, "to_qubo_seconds": to_qubo_seconds,
        "peak_rss_mb": peak_rss_mb(),
    })

    # The sparse QUBO, so the annealing stage does not rebuild it, and the
    # index -> variable-name mapping needed to decode a sample. Both are
    # regenerable from model.pth and Info.txt, which is why only the mapping
    # ships in data/width_scaling.tar.gz and the matrix does not.
    rows = np.empty(len(Q), dtype=np.int32)
    cols = np.empty(len(Q), dtype=np.int32)
    vals = np.empty(len(Q), dtype=np.float64)
    for n, (key, value) in enumerate(Q.items()):
        rows[n], cols[n], vals[n] = key[0], key[-1], value
    np.savez_compressed(folder / "qubo.npz", rows=rows, cols=cols, vals=vals,
                        offset=np.float64(offset), n=np.int64(n_vars))
    (folder / "qubo_vars.json").write_text(json.dumps(
        {"reverse_mapping": reverse, "ordered_variables": ordered_variables,
         "pcbo_mapping": pcbo_mapping}), encoding="utf-8")

    # The pickled PCBO lets `validate` check the constraints directly. It is
    # large -- 313 MB across both series -- so it is off by default and is not
    # shipped; --max-pickle-terms turns it back on.
    if a.max_pickle_terms and len(Q) <= a.max_pickle_terms:
        with open(folder / "pcbo.pkl", "wb") as handle:
            pickle.dump(hamiltonian, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[qubo] {a.input_size}x{a.input_size} w={a.width} "
          f"vars={n_vars} constraints={sum(counts.values())} terms={len(Q)} "
          f"({build_seconds:.1f}s build)", flush=True)


# -----------------------------------------------------------------------------
# Stage 5: simulated annealing
# -----------------------------------------------------------------------------

def load_qubo(folder: Path):
    import numpy as np
    z = np.load(folder / "qubo.npz")
    Q = {(int(r), int(c)): float(v)
         for r, c, v in zip(z["rows"], z["cols"], z["vals"])}
    return Q, float(z["offset"]), int(z["n"])


def cmd_sa(a) -> None:
    import dimod
    import numpy as np
    from dwave.samplers import SimulatedAnnealingSampler

    folder = cell_dir(a.out, a.input_size, a.width)
    Q, offset, n = load_qubo(folder)
    target = -offset
    bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
    sampler = SimulatedAnnealingSampler()

    # Calibrate on a few reads, so a full 2,048-read run is never started blind.
    started = time.perf_counter()
    sampler.sample(bqm, num_reads=a.probe_reads, num_sweeps=a.sweeps,
                   beta_schedule_type="geometric", seed=a.seed)
    per_read = (time.perf_counter() - started) / a.probe_reads
    projected = per_read * a.reads

    reads, truncated = a.reads, False
    if projected > a.budget_seconds:
        reads = max(1, int(a.budget_seconds / per_read))
        truncated = True

    started = time.perf_counter()
    sampleset = sampler.sample(bqm, num_reads=reads, num_sweeps=a.sweeps,
                               beta_schedule_type="geometric", seed=a.seed)
    elapsed = time.perf_counter() - started
    energies = np.asarray(sampleset.record.energy, dtype=float)
    best = float(energies.min())
    at_target = np.abs(energies - target) < 1e-6

    write_json(folder / f"sa{a.tag}_seed{a.seed}.json", {
        "input_size": a.input_size, "width": a.width, "seed": a.seed,
        "tag": a.tag, "qubo_terms": len(Q), "qubo_variables": n,
        "requested_reads": a.reads, "executed_reads": reads,
        "sweeps": a.sweeps, "beta_schedule": "geometric",
        "truncated_to_fit_budget": truncated,
        "budget_seconds": a.budget_seconds,
        "seconds_per_read": per_read,
        "projected_full_run_seconds": projected,
        "wall_seconds": elapsed,
        "target_energy": target, "best_energy": best,
        "gap_to_target": best - target,
        "reached_target": bool(abs(best - target) < 1e-6),
        "num_reads_at_target": int(at_target.sum()),
        # Reads are executed in order, so the first read that hits the target
        # gives an honest time-to-first-success for an equal-time comparison
        # against the Z3 solve time.
        "first_read_at_target": (int(np.argmax(at_target)) + 1
                                 if at_target.any() else None),
        "seconds_to_first_target": (
            float((int(np.argmax(at_target)) + 1) * per_read)
            if at_target.any() else None),
        "best_energy_by_read_min": float(np.minimum.accumulate(energies)[-1]),
        "energy_mean": float(energies.mean()),
        "energy_median": float(np.median(energies)),
        "peak_rss_mb": peak_rss_mb(),
    })
    best_sample = sampleset.first.sample
    np.save(folder / f"sa{a.tag}_seed{a.seed}_best.npy",
            np.array([best_sample[i] for i in range(n)], dtype=np.int8))
    print(f"[sa{a.tag}] {a.input_size}x{a.input_size} w={a.width} "
          f"seed={a.seed} reads={reads}/{a.reads} best={best:.1f} "
          f"target={target:.1f} "
          f"reached={abs(best - target) < 1e-6} ({elapsed:.0f}s)", flush=True)


# -----------------------------------------------------------------------------
# Stage 6: replay any stored SA sample through an independent forward pass
# -----------------------------------------------------------------------------

def cmd_validate(a) -> None:
    """Decode the stored SA samples and replay them on the BNN itself.

    Shares no code with the QUBO construction: the forward pass comes from
    Z3.py's NumPy implementation. Constraint satisfaction is also checked with
    qubovert's own predicates when the pickled PCBO is present.
    """
    import numpy as np
    import Z3 as z3mod

    folder = cell_dir(a.out, a.input_size, a.width)
    meta = json.loads((Path(a.out) / f"{a.input_size}x{a.input_size}" /
                       "instance.json").read_text())
    reverse = {int(k): v for k, v in json.loads(
        (folder / "qubo_vars.json").read_text())["reverse_mapping"].items()}
    _, _, n = load_qubo(folder)
    layers = z3mod.load_binary_linear_layers(folder / "model.pth")

    hamiltonian = None
    if (folder / "pcbo.pkl").exists():
        with open(folder / "pcbo.pkl", "rb") as handle:
            hamiltonian = pickle.load(handle)
    predicates = {"eq": lambda v: v == 0, "ne": lambda v: v != 0,
                  "lt": lambda v: v < 0, "le": lambda v: v <= 0,
                  "gt": lambda v: v > 0, "ge": lambda v: v >= 0}

    tau_position = {}
    for index, name in reverse.items():
        found = re.fullmatch(r"tau_(\d+)", name)
        if found:
            tau_position[int(found.group(1))] = index

    clean = clean_input(folder)
    clean_label, _ = z3mod.forward_binary_network(clean, layers)

    rows = []
    for path in sorted(folder.glob("sa_seed*_best.npy")):
        seed = int(re.search(r"seed(\d+)", path.name).group(1))
        bits = np.load(path)
        solution = {reverse[i]: int(bits[i]) for i in range(n)}

        satisfied = total = None
        if hamiltonian is not None:
            satisfied = total = 0
            for kind, constraints in hamiltonian.constraints.items():
                for constraint in constraints:
                    total += 1
                    satisfied += bool(predicates[kind](
                        constraint.value(solution)))

        adversarial = list(clean)
        for pixel in meta["pixels"]:
            if pixel in tau_position and bits[tau_position[pixel]] == 1:
                adversarial[pixel] = 1 - adversarial[pixel]
        adv_label, _ = z3mod.forward_binary_network(adversarial, layers)
        changed = [i for i in range(len(adversarial))
                   if adversarial[i] != clean[i]]
        rows.append({
            "seed": seed,
            "constraints_satisfied": satisfied, "constraints_total": total,
            "all_constraints_satisfied": (satisfied == total
                                          if total is not None else None),
            "clean_label": clean_label, "adversarial_label": adv_label,
            "label_changed": adv_label != clean_label,
            "pixels_flipped": len(changed),
            "within_qubo_budget": len(changed) <= meta["epsilon"] - 1,
            "outside_perturbable": sorted(set(changed) - set(meta["pixels"])),
        })
    write_json(folder / "validation.json", rows)
    for row in rows:
        print(f"[validate] {a.input_size}x{a.input_size} w={a.width} "
              f"seed={row['seed']} constraints "
              f"{row['constraints_satisfied']}/{row['constraints_total']} "
              f"label {row['clean_label']}->{row['adversarial_label']} "
              f"flips={row['pixels_flipped']}", flush=True)


def clean_input(folder: Path):
    """The clean Boolean input, read back out of the cell's own Info.txt."""
    text = (folder / "Info.txt").read_text(encoding="utf-8")
    values = re.search(r"tensor\(\[(.*?)\]\)", text, re.S).group(1)
    return [int(float(value)) for value in values.split(",")]


# -----------------------------------------------------------------------------
# The whole sweep, in one process
# -----------------------------------------------------------------------------

def cmd_sweep(a) -> None:
    from argparse import Namespace

    defaults = SERIES_DEFAULTS.get(
        a.input_size, {"epochs": a.epochs or 250, "eval_every": 25,
                       "budget_seconds": 3600.0})
    epochs = a.epochs or defaults["epochs"]
    started = time.perf_counter()

    print(f"[sweep] {a.input_size}x{a.input_size} "
          f"({INPUT_DIM[a.input_size]} inputs), widths {a.widths}, "
          f"{epochs} epochs, seeds {a.seeds}", flush=True)
    print("[sweep] a full sweep takes many hours; see the module docstring. "
          "The shipped", flush=True)
    print("[sweep] records in data/width_scaling.tar.gz let you check the "
          "results instead.", flush=True)

    for width in a.widths:
        cmd_train(Namespace(input_size=a.input_size, width=width,
                            epochs=epochs, lr=0.01, seed=12345,
                            eval_every=defaults["eval_every"],
                            threads=a.threads, out=a.out))

    cmd_instances(Namespace(input_size=a.input_size, widths=a.widths,
                            max_scan=a.max_scan, out=a.out))

    for width in a.widths:
        cmd_qubo(Namespace(input_size=a.input_size, width=width,
                           max_pickle_terms=a.max_pickle_terms, out=a.out))
        cmd_z3(Namespace(input_size=a.input_size, width=width,
                         timeout=a.z3_timeout, scan=not a.no_scan,
                         scan_max=0, scan_timeout=a.scan_timeout, out=a.out))
        for seed in a.seeds:
            budget = a.budget_seconds
            if budget is None:
                budget = defaults["budget_seconds"]
                if (a.input_size == 28
                        and width in UNBOUNDED_WIDTHS_1023):
                    budget = 1e9
            cmd_sa(Namespace(input_size=a.input_size, width=width, seed=seed,
                             reads=a.reads, sweeps=a.sweeps,
                             probe_reads=a.probe_reads,
                             budget_seconds=budget, tag="", out=a.out))
        cmd_validate(Namespace(input_size=a.input_size, width=width,
                               out=a.out))

    print(f"[sweep] done in {(time.perf_counter() - started) / 3600:.2f} h; "
          f"records under {a.out}/{a.input_size}x{a.input_size}/", flush=True)


# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="width_scaling_sweep.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="width_scaling_rerun",
                        help="where to write the records "
                             "(default: width_scaling_rerun/, git-ignored)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    train = sub.add_parser("train", help="train one width")
    train.add_argument("--input-size", type=int, required=True,
                       choices=sorted(INPUT_DIM))
    train.add_argument("--width", type=int, required=True)
    train.add_argument("--epochs", type=int, default=1000)
    train.add_argument("--lr", type=float, default=0.01)
    train.add_argument("--seed", type=int, default=12345)
    train.add_argument("--eval-every", type=int, default=100)
    train.add_argument("--threads", type=int, default=1)
    train.set_defaults(func=cmd_train)

    instances = sub.add_parser(
        "instances", help="choose the instance every width shares")
    instances.add_argument("--input-size", type=int, required=True,
                           choices=sorted(INPUT_DIM))
    instances.add_argument("--widths", type=int, nargs="+", default=WIDTHS)
    instances.add_argument("--max-scan", type=int, default=2000)
    instances.set_defaults(func=cmd_instances)

    z3 = sub.add_parser("z3", help="the exact SMT baseline at one width")
    z3.add_argument("--input-size", type=int, required=True,
                    choices=sorted(INPUT_DIM))
    z3.add_argument("--width", type=int, required=True)
    z3.add_argument("--timeout", type=float, default=1800.0)
    z3.add_argument("--scan", action="store_true",
                    help="also scan for the exact minimum adversarial distance")
    z3.add_argument("--scan-max", type=int, default=0)
    z3.add_argument("--scan-timeout", type=float, default=600.0)
    z3.set_defaults(func=cmd_z3)

    qubo = sub.add_parser("qubo", help="build the QUBO at one width")
    qubo.add_argument("--input-size", type=int, required=True,
                      choices=sorted(INPUT_DIM))
    qubo.add_argument("--width", type=int, required=True)
    qubo.add_argument("--max-pickle-terms", type=int, default=0,
                      help="also pickle the PCBO when it has at most this many "
                           "terms; 0 (the default) never does. `validate` "
                           "checks constraints directly when it is present, "
                           "and it reaches 159 MB at the widest cell")
    qubo.set_defaults(func=cmd_qubo)

    sa = sub.add_parser("sa", help="anneal one width and seed")
    sa.add_argument("--input-size", type=int, required=True,
                    choices=sorted(INPUT_DIM))
    sa.add_argument("--width", type=int, required=True)
    sa.add_argument("--seed", type=int, required=True)
    sa.add_argument("--reads", type=int, default=2048)
    sa.add_argument("--sweeps", type=int, default=5000)
    sa.add_argument("--probe-reads", type=int, default=4)
    sa.add_argument("--budget-seconds", type=float, default=1e9,
                    help="wall-clock cap per seed. When the calibration probe "
                         "projects a longer run the read count is truncated to "
                         "fit, and the record says so; a truncated cell is a "
                         "bound, not a measurement")
    sa.add_argument("--tag", default="",
                    help="suffix for the output files, e.g. '_calib'")
    sa.set_defaults(func=cmd_sa)

    validate = sub.add_parser(
        "validate", help="replay the stored SA samples on the BNN")
    validate.add_argument("--input-size", type=int, required=True,
                          choices=sorted(INPUT_DIM))
    validate.add_argument("--width", type=int, required=True)
    validate.set_defaults(func=cmd_validate)

    sweep = sub.add_parser(
        "sweep", help="every stage, every width, in one process (many hours)")
    sweep.add_argument("--input-size", type=int, required=True,
                       choices=sorted(INPUT_DIM))
    sweep.add_argument("--widths", type=int, nargs="+", default=WIDTHS)
    sweep.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    sweep.add_argument("--epochs", type=int, default=0,
                       help="0 uses the published setting for this series: "
                            "1,000 epochs at 31 inputs, 250 at 1023")
    sweep.add_argument("--reads", type=int, default=2048)
    sweep.add_argument("--sweeps", type=int, default=5000)
    sweep.add_argument("--probe-reads", type=int, default=4)
    sweep.add_argument("--budget-seconds", type=float, default=None,
                       help="wall-clock cap per annealing seed; the default "
                            "is the one the shipped run used for this series")
    sweep.add_argument("--max-pickle-terms", type=int, default=0)
    sweep.add_argument("--max-scan", type=int, default=2000)
    sweep.add_argument("--z3-timeout", type=float, default=1800.0)
    sweep.add_argument("--scan-timeout", type=float, default=600.0)
    sweep.add_argument("--no-scan", action="store_true",
                       help="skip the minimum-distance scan, which is the "
                            "expensive half of the Z3 stage at 1023 inputs")
    sweep.add_argument("--threads", type=int, default=1)
    sweep.set_defaults(func=cmd_sweep)

    args = parser.parse_args()

    # get_args.py parses sys.argv at import time, so hide our own flags from it.
    sys.argv = [sys.argv[0]]
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    args.func(args)


if __name__ == "__main__":
    main()
