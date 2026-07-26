#!/usr/bin/env python3
"""linear_baseline.py -- the complete baselines of Section IV-B, on the
two-class hardware instance, in its LINEAR constraint form.

Section IV-B compares the QUBO form of the 113-variable two-class instance with
the linear constraint system the QUBO is built from. This script is that
comparison. It reads nothing but

    hardware/QUBO/113-1273-28-15-zero-3-3-561020-H.pickle

which ships in ``data/hardware_results.tar.gz``, plus the two archived solution
vectors next to it. Everything below is recomputed from those files.

Why a separate script
---------------------

The pickle stores the instance twice. ``pickled["qubo"]`` is the penalised QUBO,
which is what every solver in Table VII was given. ``pickled["constraints"]``
holds the 64 equalities and the single perturbation-budget inequality the QUBO
was built from, and every one of those 65 entries is strictly linear: each
dictionary maps a 1-tuple of variable names to a coefficient, plus one 0-tuple
constant. Nothing else in this repository ever builds a model from that form,
so the paragraph's numbers could not be re-derived from the repository. This
script builds it.

What it reports
---------------

  * Gurobi on the linear form, asked first for any feasible perturbation and
    then for the smallest one. Both are reported with the build+solve wall time
    and with Gurobi's own ``Runtime``. Needs a license, so it is behind
    ``--with-gurobi``.
  * Z3 on the same constraint system, likewise for a witness and for the proven
    minimum. Needs only ``z3-solver``.
  * Exhaustive enumeration over all 2^15 perturbation patterns, vectorised in
    NumPy. Needs only NumPy.
  * The number of perturbation variables the two archived solution vectors set,
    which is what the paper's "the digital annealer flips 10 pixels where 4 are
    enough" refers to.

Why enumeration is possible at all, and why it is exact
-------------------------------------------------------

Of the 113 QUBO variables, 15 are the perturbation variables ``tau_*``, 94 are
determined by the 64 equalities once those 15 are fixed, and 4 are slack
ancillas that qubovert adds when it penalises the budget inequality. Only the
first 15 are free, so 2^15 = 32,768 patterns cover the whole search space.

The 94 are recovered by propagating the equalities, and each propagation step is
forced rather than heuristic, which is what makes the enumeration a proof and
not a sample:

  * an equality with exactly one unknown fixes that unknown outright;
  * an equality whose unknown coefficients are, in magnitude, the consecutive
    powers of two 1, 2, ..., 2^(m-1) fixes all m of them, because a binary
    place-value expansion of a given value is unique. Negative coefficients are
    handled by substituting z = 1 - y before reading the bits off.

Both rules are sound in the strict sense: if the propagated value falls outside
{0, 1}, or if the completed vector fails any equality, then no binary completion
of that perturbation pattern exists at all. The code checks both afterwards
rather than assuming them, so a pattern counted as feasible has an explicit
witness and a pattern counted as infeasible has been ruled out, not skipped.

Timings
-------

Wall-clock numbers are printed but are never asserted anywhere, here or in
``verify_paper.py``, for the same reason Table VII's Gurobi and SA runtimes are
not: they are machine dependent. The paper's figures for this paragraph were
measured on an Apple M1 Pro.

Usage
-----

    python linear_baseline.py                 # Z3 + enumeration + flip counts
    python linear_baseline.py --with-gurobi   # also the Gurobi rows
    python linear_baseline.py --repeats 5     # best of N for each timed solver
"""

import argparse
import itertools
import os
import pickle
import time

import numpy as np


HARDWARE_DIR = "hardware"
HARDWARE_STEM = "113-1273-28-15-zero-3-3-561020-H"
INSTANCE = f"{HARDWARE_DIR}/QUBO/{HARDWARE_STEM}.pickle"
FUJITSU_SOLUTION = f"{HARDWARE_DIR}/Result/{HARDWARE_STEM}_solution.pickle"
GUROBI_SOLUTION = f"{HARDWARE_DIR}/Gurobi/{HARDWARE_STEM}_solution.pickle"
HARDWARE_ARCHIVE = os.path.join("data", "hardware_results.tar.gz")

# The paragraph's claims, quoted from Section IV-B of the manuscript.
CLAIMED = {
    "feasible": 30826,
    "d_min": 4,
    "fujitsu_flips": 10,
    "gurobi_flips": 7,
    # Reported timings. Printed for comparison, never asserted.
    "gurobi_seconds": 0.0015,
    "z3_witness_seconds": 0.034,
    "z3_minimum_seconds": 0.038,
    "enumeration_seconds": 0.042,
}


# -----------------------------------------------------------------------------
# The linear form
# -----------------------------------------------------------------------------

class LinearForm:
    """The 65 linear constraints of the two-class instance, as matrices.

    ``Aeq @ x == beq`` and ``Alt @ x <= blt - 1``. The second is the strict
    "< 0" of the pickled inequality: every coefficient there is an integer and
    every variable is binary, so ``< 0`` and ``<= -1`` are the same constraint.
    """

    def __init__(self, pickled):
        equalities = pickled["constraints"]["eq"]
        inequalities = pickled["constraints"]["lt"]
        self.equalities = equalities
        self.inequalities = inequalities

        # Every constraint entry must be linear for any of this to be valid.
        degrees = {len(key) for entry in equalities + inequalities
                   for key in entry}
        if not degrees <= {0, 1}:
            raise ValueError(f"constraint entries are not linear: "
                             f"term degrees {sorted(degrees)}")

        self.names = sorted({key[0] for entry in equalities + inequalities
                             for key in entry if len(key) == 1})
        self.index = {name: i for i, name in enumerate(self.names)}
        self.taus = [name for name in self.names if name.startswith("tau_")]
        self.tau_index = [self.index[name] for name in self.taus]
        self.Aeq, self.beq = self._rows(equalities)
        self.Alt, self.blt = self._rows(inequalities)

    def _rows(self, entries):
        A = np.zeros((len(entries), len(self.names)))
        b = np.zeros(len(entries))
        for row, entry in enumerate(entries):
            for key, coefficient in entry.items():
                if len(key) == 1:
                    A[row, self.index[key[0]]] = coefficient
                else:
                    b[row] -= coefficient        # constant moved to the rhs
        return A, b


def propagate(X, A, b, known):
    """Complete `X` from the equality system `A x = b`, in place.

    `X` is (rows, variables); `known` is a boolean mask over the columns whose
    values are already set. Every column derived here is FORCED by the two rules
    documented in the module docstring, so the completion is unique wherever it
    succeeds. Returns the list of equality rows that could not be used, and
    leaves `known` marking the columns that were determined.

    Shared with bruteforce_5x5_energy.py, which needs the same completion on the
    5x5 instance.
    """
    pending = list(range(A.shape[0]))
    while pending:
        still = []
        for row in pending:
            support = np.nonzero(A[row])[0]
            unknown = [j for j in support if not known[j]]
            settled = [j for j in support if known[j]]
            if not unknown:
                continue
            rhs = b[row] - X[:, settled] @ A[row, settled]
            if len(unknown) == 1:
                X[:, unknown[0]] = np.round(
                    rhs / A[row, unknown[0]]).astype(np.int64)
                known[unknown[0]] = True
                continue
            coefficients = A[row, unknown]
            magnitudes = np.abs(coefficients)
            powers = [2.0 ** i for i in range(len(unknown))]
            if sorted(magnitudes.tolist()) == powers:
                # y with a negative coefficient enters as z = 1 - y, which moves
                # |coefficient| to the right-hand side.
                shifted = (rhs + magnitudes[coefficients < 0].sum()
                           ).astype(np.int64)
                order = np.argsort(magnitudes)
                for bit, column in enumerate(np.asarray(unknown)[order]):
                    z = (shifted >> bit) & 1
                    X[:, column] = z if coefficients[order[bit]] > 0 else 1 - z
                    known[column] = True
            else:
                still.append(row)
        if still == pending:
            return still
        pending = still
    return []


# -----------------------------------------------------------------------------
# Exhaustive enumeration over the 2^15 perturbation patterns
# -----------------------------------------------------------------------------

def enumerate_patterns(form):
    """Every feasible perturbation of the instance, by exhaustive completion.

    Returns a dict with the feasible count, d_min, the distance histogram and
    the wall time. See the module docstring for why this is exact.
    """
    width = len(form.taus)
    total = 1 << width
    grid = ((np.arange(total)[:, None] >> np.arange(width)) & 1).astype(np.int64)

    started = time.perf_counter()
    X = np.zeros((total, len(form.names)), dtype=np.int64)
    X[:, form.tau_index] = grid
    known = np.zeros(len(form.names), dtype=bool)
    known[form.tau_index] = True
    stuck = propagate(X, form.Aeq, form.beq, known)
    if stuck or not known.all():
        undetermined = [form.names[i] for i in np.nonzero(~known)[0]]
        raise RuntimeError(
            f"the equalities did not determine every variable: "
            f"{len(stuck)} equality/equalities unused, undetermined "
            f"{undetermined}")

    binary = ((X >= 0) & (X <= 1)).all(1)
    satisfies_equalities = (X @ form.Aeq.T == form.beq).all(1)
    within_budget = (X @ form.Alt.T <= form.blt - 1).all(1)
    feasible = binary & satisfies_equalities & within_budget
    elapsed = time.perf_counter() - started

    distances = grid[feasible].sum(1)
    return {
        "patterns": total,
        "feasible": int(feasible.sum()),
        "d_min": int(distances.min()) if len(distances) else None,
        "histogram": np.bincount(distances, minlength=width + 1).tolist(),
        "seconds": elapsed,
        "mask": feasible,
        "grid": grid,
    }


# -----------------------------------------------------------------------------
# The two archived solution vectors
# -----------------------------------------------------------------------------

def flip_counts(pickled):
    """How many perturbation variables each archived solution vector sets.

    The vectors are keyed by QUBO index, so the qubovert model is rebuilt to
    recover the name -> index mapping; this is the same rebuild verify_paper.py
    uses for Table VII, and it is checked against the pickled QUBO there.
    Returns {label: (flips, path)} for whichever vectors are present.
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

    taus = [name for name in model.variables if str(name).startswith("tau_")]
    counts = {}
    for label, path in (("Digital Annealer", FUJITSU_SOLUTION),
                        ("Gurobi", GUROBI_SOLUTION)):
        if not os.path.exists(path):
            continue
        with open(path, "rb") as handle:
            raw = pickle.load(handle)
        solution = {int(key): int(value) for key, value in raw.items()}
        converted = model.convert_solution(solution)
        counts[label] = {
            "flips": sum(int(bool(converted[name])) for name in taus),
            "path": path,
            "valid": bool(model.is_solution_valid(converted)),
        }
    return counts


# -----------------------------------------------------------------------------
# Z3, on the same constraint system
# -----------------------------------------------------------------------------

def solve_z3(form, minimize, repeats=1):
    """Z3 on the linear form. `minimize` proves the smallest perturbation.

    The reported time is encode + solve, which is what the paper's 0.034 s and
    0.038 s are; Table V's figures are solve-only and are not comparable.
    """
    import z3

    best = None
    result = model = variables = None
    for _ in range(max(1, repeats)):
        started = time.perf_counter()
        variables = {name: z3.Bool(name) for name in form.names}

        def linear(entry):
            return (z3.Sum([z3.If(variables[key[0]], int(entry[key]), 0)
                            for key in entry if len(key) == 1])
                    + int(sum(entry[key] for key in entry if len(key) == 0)))

        solver = z3.Optimize() if minimize else z3.Solver()
        for entry in form.equalities:
            solver.add(linear(entry) == 0)
        for entry in form.inequalities:
            solver.add(linear(entry) <= -1)
        if minimize:
            solver.minimize(z3.Sum([z3.If(variables[name], 1, 0)
                                    for name in form.taus]))
        result = solver.check()
        if result == z3.sat:
            model = solver.model()
        elapsed = time.perf_counter() - started
        best = elapsed if best is None else min(best, elapsed)

    if result != z3.sat:
        return {"status": str(result), "seconds": best, "assignment": None}

    assignment = {name: bool(z3.is_true(model[variables[name]]))
                  for name in form.names}
    return {
        "status": "sat",
        "seconds": best,
        "distance": sum(assignment[name] for name in form.taus),
        "assignment": assignment,
    }


def check_assignment(form, assignment):
    """Re-verify a solver's assignment against the raw constraint dictionaries.

    Deliberately not done through the matrices above: this reads the pickled
    entries directly, so a mistake in building `Aeq`/`Alt` cannot make a bad
    witness look good.
    """
    def value(entry):
        return (sum(entry[key] * int(assignment[key[0]])
                    for key in entry if len(key) == 1)
                + sum(entry[key] for key in entry if len(key) == 0))

    return (all(value(entry) == 0 for entry in form.equalities)
            and all(value(entry) < 0 for entry in form.inequalities))


# -----------------------------------------------------------------------------
# Gurobi, on the same constraint system
# -----------------------------------------------------------------------------

def solve_gurobi(form, minimize, repeats=1):
    """Gurobi on the linear form, built with addConstr on the matrices above.

    The license environment is created once and outside the timed region, the
    way the paper's figure was measured: it is a model build and a solve, not a
    license handshake. Both the wall time and Gurobi's own `Runtime` are
    returned, because they differ by the model build and the paper's 0.0015 s
    includes it.
    """
    import gurobipy as gp
    from gurobipy import GRB

    environment = gp.Env(params={"OutputFlag": 0})
    gp.Model(env=environment).optimize()      # warm-up, outside the timing

    n = len(form.names)
    best_wall = best_runtime = None
    status = distance = objective = None
    for _ in range(max(1, repeats)):
        started = time.perf_counter()
        model = gp.Model(env=environment)
        x = model.addMVar(n, vtype=GRB.BINARY)
        model.addConstr(form.Aeq @ x == form.beq)
        model.addConstr(form.Alt @ x <= form.blt - 1)
        if minimize:
            weights = np.zeros(n)
            weights[form.tau_index] = 1.0
            model.setObjective(weights @ x, GRB.MINIMIZE)
        model.optimize()
        wall = time.perf_counter() - started
        status = model.Status
        if status == GRB.OPTIMAL:
            distance = int(round(x.X[form.tau_index].sum()))
            objective = model.ObjVal if minimize else None
            assignment = {name: bool(round(x.X[i]))
                          for i, name in enumerate(form.names)}
        best_wall = wall if best_wall is None else min(best_wall, wall)
        best_runtime = (model.Runtime if best_runtime is None
                        else min(best_runtime, model.Runtime))

    return {
        "status": status,
        "optimal": status == GRB.OPTIMAL,
        "seconds": best_wall,
        "solver_seconds": best_runtime,
        "distance": distance,
        "objective": objective,
        "assignment": assignment if status == GRB.OPTIMAL else None,
    }


# -----------------------------------------------------------------------------
# Command line
# -----------------------------------------------------------------------------

def load_instance(path=INSTANCE):
    if not os.path.exists(path):
        raise SystemExit(
            f"{path} is not present. Unpack it with:\n"
            f"    tar xzf {HARDWARE_ARCHIVE}")
    with open(path, "rb") as handle:
        return pickle.load(handle)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="linear_baseline.py",
        description="Section IV-B's complete baselines on the two-class "
                    "instance, in its linear constraint form.")
    parser.add_argument("--with-gurobi", action="store_true",
                        help="also solve the linear form with Gurobi "
                             "(needs a license)")
    parser.add_argument("--repeats", type=int, default=3,
                        help="report the best of N runs for each timed solver "
                             "(default 3)")
    options = parser.parse_args(argv)

    pickled = load_instance()
    form = LinearForm(pickled)

    print("=" * 79)
    print(" Section IV-B -- the two-class instance in its LINEAR constraint "
          "form")
    print("=" * 79)
    print(f" instance    : {INSTANCE}")
    print(f" constraints : {len(form.equalities)} equalities + "
          f"{len(form.inequalities)} budget inequality, all strictly linear")
    print(f" variables   : {len(form.names)} named "
          f"({len(form.taus)} perturbation + "
          f"{len(form.names) - len(form.taus)} determined by the equalities); "
          f"the QUBO carries 4 slack")
    print( "               ancillas on top of these, for 113 in total")
    print()
    print(" Timings are reported, never asserted: they are machine dependent. "
          "The paper's")
    print(" figures for this paragraph were measured on an Apple M1 Pro.")
    print()

    # -- enumeration ---------------------------------------------------------
    result = enumerate_patterns(form)
    print(f" exhaustive enumeration over 2^{len(form.taus)} = "
          f"{result['patterns']:,} perturbation patterns")
    print(f"   feasible assignments : {result['feasible']:,}"
          f"   (paper {CLAIMED['feasible']:,})")
    print(f"   d_min                : {result['d_min']}"
          f"   (paper {CLAIMED['d_min']})")
    print(f"   wall time            : {result['seconds']:.4f} s"
          f"   (paper {CLAIMED['enumeration_seconds']} s)")
    print(f"   feasible by distance : {result['histogram']}")
    print()

    # -- flip counts ---------------------------------------------------------
    print(" perturbation size of the archived solution vectors")
    counts = flip_counts(pickled)
    if not counts:
        print("   no solution vector found next to the instance")
    for label, entry in counts.items():
        claimed = CLAIMED["fujitsu_flips" if label.startswith("Digital")
                          else "gurobi_flips"]
        print(f"   {label:<17}: {entry['flips']:>2} of "
              f"{len(form.taus)} perturbation variables set"
              f"   (paper {claimed}); all constraints satisfied: "
              f"{entry['valid']}")
    print(f"   d_min from enumeration above is {result['d_min']}, so both "
          f"flip more pixels than they need to;")
    print( "   the objective H_0 is identically zero, so every feasible "
           "assignment has the same energy")
    print( "   and nothing in the QUBO prefers a smaller perturbation.")
    print()

    # -- Z3 ------------------------------------------------------------------
    print(" Z3 on the same constraint system (encode + solve)")
    try:
        import z3  # noqa: F401
    except ImportError as error:
        print(f"   UNAVAILABLE: z3-solver is not installed ({error})")
        print( "   install it with: pip install -r requirements.txt")
    else:
        witness = solve_z3(form, minimize=False, repeats=options.repeats)
        print(f"   witness            : {witness['status']}, "
              f"{witness.get('distance')} pixels, "
              f"{witness['seconds']:.4f} s"
              f"   (paper {CLAIMED['z3_witness_seconds']} s)")
        if witness["assignment"] is not None:
            print(f"   witness re-checked against the pickled constraint "
                  f"dictionaries: "
                  f"{check_assignment(form, witness['assignment'])}")
        minimum = solve_z3(form, minimize=True, repeats=options.repeats)
        print(f"   proven minimum     : {minimum['status']}, d_min = "
              f"{minimum.get('distance')}, "
              f"{minimum['seconds']:.4f} s"
              f"   (paper {CLAIMED['z3_minimum_seconds']} s)")
    print()

    # -- Gurobi --------------------------------------------------------------
    print(" Gurobi on the same constraint system (build + solve)")
    if not options.with_gurobi:
        print("   SKIPPED: pass --with-gurobi to run it (needs a license)")
    else:
        try:
            import gurobipy  # noqa: F401
        except ImportError as error:
            print(f"   UNAVAILABLE: gurobipy is not installed ({error})")
        else:
            try:
                feasible = solve_gurobi(form, minimize=False,
                                        repeats=options.repeats)
                smallest = solve_gurobi(form, minimize=True,
                                        repeats=options.repeats)
            except Exception as error:          # license, mostly
                print(f"   UNAVAILABLE: {error!r}")
            else:
                for label, entry in (("feasible perturbation", feasible),
                                     ("smallest perturbation", smallest)):
                    print(f"   {label:<22}: status {entry['status']}, "
                          f"{entry['distance']} pixels, "
                          f"{entry['seconds']:.4f} s build+solve "
                          f"({entry['solver_seconds']:.4f} s in the solver)"
                          f"   (paper {CLAIMED['gurobi_seconds']} s)")
                if smallest["assignment"] is not None:
                    print(f"   minimiser re-checked against the pickled "
                          f"constraint dictionaries: "
                          f"{check_assignment(form, smallest['assignment'])}")
    print()
    print(" Table VII gives 61.447 s for Gurobi on the QUBO form of this same "
          "instance.")
    print(" The distance between the two is a property of the penalty "
          "encoding, not of the")
    print(" instance: squaring each residual removes the linear structure both "
          "complete")
    print(" methods exploit. See Section IV-B.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
