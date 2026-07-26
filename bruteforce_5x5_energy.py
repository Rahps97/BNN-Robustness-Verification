#!/usr/bin/env python3
"""bruteforce_5x5_energy.py -- energy equals feasibility, checked exhaustively.

The response letter states that, because the objective H_0 is identically zero
in every reported experiment, reaching the target energy -E_off is an exact
feasibility criterion, and that this was "confirmed independently on the 5x5
instance by brute-forcing all 39,203 perturbations within budget". This script
is that experiment.

39,203 is C(16,0) + C(16,1) + ... + C(16,8): every subset of the 16 perturbable
pixels of the 5x5 instance that flips at most the perturbation bound of 8. For
each one it computes both sides of the claimed equivalence and compares them:

    x^T Q x == -E_off        <=>        every encoded constraint is satisfied

The left-hand side is scored against the shipped QUBO_W.txt. The right-hand
side is evaluated from H.constraints -- the QCBO constraint polynomials the
builder recorded, before any penalty was formed -- so the two sides come from
different objects and a mistake in the penalisation cannot make them agree
trivially.

The two completions this needs
------------------------------

A perturbation names only the 16 tau variables, and the QUBO has 276. The other
260 fall into two groups, handled differently:

  * 253 are determined by the 198 equalities once the taus are fixed. They are
    recovered by the forced propagation in linear_baseline.propagate(), which is
    the same routine the two-class instance uses; see that module for why each
    step is forced rather than heuristic. Every completion is then re-checked
    against all 198 equalities rather than assumed correct.
  * 7 are slack ancillas that qubovert adds when it penalises the budget
    inequality and the misclassification inequality. They appear in no recorded
    constraint, only in the penalties, so the QUBO energy of a perturbation is
    the MINIMUM over their 2^7 settings. That minimum is what the equivalence is
    about, and it is what this script computes: enumerating the 128 settings
    directly, over a decomposition of x^T Q x into its ancilla-free part, its
    cross terms and its ancilla-only part.

What it should print
--------------------

Of the 39,203 perturbations, 8,773 satisfy every encoded constraint and exactly
those 8,773 reach the target energy of -533. The equivalence therefore holds in
both directions and neither is vacuous: 30,430 perturbations satisfy the
constraints of the network but not the budget or the misclassification
inequality, and none of them reaches the target.

Exit status is 0 if the equivalence holds on every perturbation and 1 if it does
not.

Usage
-----

    python bruteforce_5x5_energy.py
"""

import itertools
import sys
import time
from math import comb

import numpy as np

SIZE = 5
TARGET_COUNT = 39203          # sum_{k=0}^{8} C(16, k), the response letter's figure


def constraint_rows(entries, position, width):
    """A list of qubovert constraint polynomials as `A x <= / == / > b` rows."""
    A = np.zeros((len(entries), width))
    b = np.zeros(len(entries))
    for row, entry in enumerate(entries):
        for key, coefficient in entry.items():
            if len(key) == 1:
                A[row, position[key[0]]] = coefficient
            elif len(key) == 0:
                b[row] -= coefficient
            else:
                raise ValueError("the 5x5 constraints are expected to be "
                                 "linear; found a degree-2 term")
    return A, b


def main():
    # get_args.py parses sys.argv at import time, so hide anything we were given.
    sys.argv = [sys.argv[0]]

    import verify_paper as vp
    from linear_baseline import propagate

    info = vp.read_info(SIZE)
    H, ordered = vp.rebuild_qubo(SIZE, info)
    order = list(ordered)
    position = {name: i for i, name in enumerate(order)}
    width = len(order)

    taus = [name for name in order if str(name).startswith("tau_")]
    tau_columns = [position[name] for name in taus]
    ancillas = [name for name in order if str(name).startswith("__a")]
    ancilla_columns = [position[name] for name in ancillas]
    free_columns = [i for i in range(width) if i not in set(ancilla_columns)]
    epsilon = info["epsilon"]

    print("=" * 79)
    print(" 5x5 -- energy equals feasibility, over every perturbation within "
          "budget")
    print("=" * 79)
    print(f" instance   : {vp.qubo_dir(SIZE)}")
    print(f" variables  : {width} "
          f"({len(taus)} perturbation, "
          f"{width - len(taus) - len(ancillas)} determined by the equalities, "
          f"{len(ancillas)} slack ancillas)")
    print(f" constraints: " + " + ".join(
        f"{kind} {len(value)}" for kind, value in sorted(H.constraints.items())))

    expected = sum(comb(len(taus), k) for k in range(epsilon + 1))
    print(f" enumerating every subset of the {len(taus)} perturbable pixels "
          f"with at most {epsilon} flips:")
    print(f"   sum_(k=0..{epsilon}) C({len(taus)}, k) = {expected:,}"
          f"   (the response letter's {TARGET_COUNT:,})")
    if expected != TARGET_COUNT:
        print(f" MISMATCH: expected {TARGET_COUNT:,}")
        return 1

    started = time.perf_counter()
    grid = np.zeros((expected, len(taus)), dtype=np.int64)
    row = 0
    for flips in range(epsilon + 1):
        for combination in itertools.combinations(range(len(taus)), flips):
            for column in combination:
                grid[row, column] = 1
            row += 1
    assert row == expected

    # -- complete each perturbation through the equalities --------------------
    equalities = H.constraints["eq"]
    Aeq, beq = constraint_rows(equalities, position, width)
    X = np.zeros((expected, width), dtype=np.int64)
    X[:, tau_columns] = grid
    known = np.zeros(width, dtype=bool)
    known[tau_columns] = True
    stuck = propagate(X, Aeq, beq, known)
    undetermined = sorted(np.nonzero(~known)[0].tolist())
    if stuck or undetermined != sorted(ancilla_columns):
        print(f" the equalities did not determine every non-ancilla variable: "
              f"{len(stuck)} unused, undetermined "
              f"{[str(order[i]) for i in undetermined]}")
        return 1

    binary = ((X[:, free_columns] >= 0) & (X[:, free_columns] <= 1)).all(1)
    satisfies_equalities = (X @ Aeq.T == beq).all(1)

    # -- the constraint side of the equivalence -------------------------------
    predicate = {"eq": lambda v: v == 0, "ne": lambda v: v != 0,
                 "lt": lambda v: v < 0, "le": lambda v: v <= 0,
                 "gt": lambda v: v > 0, "ge": lambda v: v >= 0}
    feasible = binary & satisfies_equalities
    print()
    print(" constraint side, from H.constraints:")
    for kind, entries in sorted(H.constraints.items()):
        A, b = constraint_rows(entries, position, width)
        if A[:, ancilla_columns].any():
            print(f" a slack ancilla appears in a recorded {kind} constraint; "
                  f"the ancilla minimisation below would not be sound")
            return 1
        satisfied = predicate[kind](X @ A.T - b).all(1)
        print(f"   {kind:<3} {len(entries):>4} constraint(s): "
              f"{int(satisfied.sum()):>6,} of {expected:,} perturbations "
              f"satisfy them all")
        feasible = feasible & satisfied
    print(f"   {'all':<3} {sum(len(v) for v in H.constraints.values()):>4} "
          f"constraint(s): {int(feasible.sum()):>6,} of {expected:,} "
          f"perturbations are fully feasible")

    # -- the energy side, minimised over the slack ancillas -------------------
    Q = vp.load_qubo_matrix(SIZE)
    offset = H.to_qubo()[()]
    target = -float(offset)
    F = X[:, free_columns].astype(np.float64)
    cross = (Q[np.ix_(free_columns, ancilla_columns)]
             + Q[np.ix_(ancilla_columns, free_columns)].T)
    base = np.einsum("ij,ij->i", F, F @ Q[np.ix_(free_columns, free_columns)])
    projected = F @ cross
    ancilla_block = Q[np.ix_(ancilla_columns, ancilla_columns)]
    best = None
    for mask in range(1 << len(ancillas)):
        s = np.array([(mask >> i) & 1 for i in range(len(ancillas))],
                     dtype=np.float64)
        energy = base + projected @ s + float(s @ ancilla_block @ s)
        best = energy if best is None else np.minimum(best, energy)
    elapsed = time.perf_counter() - started

    reaches_target = best == target
    print()
    print(f" energy side, from the shipped QUBO_W.txt "
          f"({Q.shape[0]}x{Q.shape[1]}), minimised over the "
          f"{1 << len(ancillas)} ancilla settings:")
    print(f"   target energy -E_off        : {target:,.0f}")
    print(f"   best energy, min over rows  : {best.min():,.0f}")
    print(f"   perturbations reaching it   : {int(reaches_target.sum()):,}")
    print(f"   none below the target       : {bool((best >= target).all())}")

    agree = bool((reaches_target == feasible).all())
    print()
    print(f" energy == target  <=>  every encoded constraint satisfied : "
          f"{agree}")
    print(f"   feasible and at the target      : "
          f"{int((feasible & reaches_target).sum()):,}")
    print(f"   feasible but not at the target  : "
          f"{int((feasible & ~reaches_target).sum()):,}")
    print(f"   at the target but infeasible    : "
          f"{int((~feasible & reaches_target).sum()):,}")
    print(f"   neither                         : "
          f"{int((~feasible & ~reaches_target).sum()):,}")
    print(f" elapsed: {elapsed:.1f} s")
    if not agree:
        print(" RESULT: FAIL -- the equivalence does not hold on every "
              "perturbation.")
        return 1
    print(f" RESULT: PASS -- the equivalence holds on all {expected:,} "
          f"perturbations within budget.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
