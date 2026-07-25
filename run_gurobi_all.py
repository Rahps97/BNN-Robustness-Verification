"""Run Gurobi on every BNN-derived QUBO and report the numbers needed to check
the manuscript's Gurobi column.

Wraps the same model and early-stopping callback as Gurobi.py, but loops over all
four instances, reads the target energy from each Info.txt, and reports the
resulting energy gap. Writes gurobi_results.json.

Usage (from the repository root, with the Dataset/QUBO/TrainedNN folders present):

    python run_gurobi_all.py                 # all four instances
    python run_gurobi_all.py 5 7             # only 5x5 and 7x7
    NO_IMPR_NODES=100000 python run_gurobi_all.py

A valid Gurobi license is required.

Interpreting the output
-----------------------
Each QUBO is built so that H(x) >= -offset, with equality exactly when every
encoded constraint is satisfied. Info.txt records this bound in its
"Minimum Energy" field. So:

    gap = best_energy - target_energy      (0 means fully feasible)

The manuscript's "Constraints Satisfied" column equals offset - gap, which is why
it is reported here as `implied_constraints_satisfied`.
"""
import json
import os
import sys
import time

import numpy as np
import gurobipy as gp
from gurobipy import GRB

INPUT_DIM = {5: 31, 7: 63, 11: 127, 28: 1023}
NO_IMPR_NODES = int(os.environ.get("NO_IMPR_NODES", 10_000_000))


def early_stop_callback(model, where):
    """Stop the MIP search after too many nodes without an incumbent improvement."""
    if where == GRB.Callback.MIPSOL:
        obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
        nodecnt = model.cbGet(GRB.Callback.MIPSOL_NODCNT)
        if obj + 1e-9 < model._best_obj:
            model._best_obj = obj
            model._last_impr_node = nodecnt
    elif where == GRB.Callback.MIP:
        nodecnt = model.cbGet(GRB.Callback.MIP_NODCNT)
        if nodecnt - model._last_impr_node >= model._max_no_impr_nodes:
            model.terminate()


def read_target_energy(folder):
    """Return the 'Minimum Energy' field recorded in Info.txt, or None."""
    try:
        with open(os.path.join(folder, "Info.txt")) as fh:
            for line in fh:
                if line.startswith("Minimum Energy"):
                    return int(float(line.split(":")[1].strip()))
    except OSError:
        pass
    return None


def solve_qubo_upper_tri(Q, max_no_improvement_nodes):
    """Minimize sum_{i<=j} Q[i,j] x_i x_j over binary x, with early stopping."""
    Q = np.asarray(Q, dtype=float)
    n = Q.shape[0]
    assert Q.shape[0] == Q.shape[1], "Q must be square"

    t0 = time.perf_counter()
    m = gp.Model("QUBO_upper")
    m.Params.OutputFlag = 0
    x = m.addVars(n, vtype=GRB.BINARY, name="x")

    obj = gp.QuadExpr()
    for i in range(n):
        if Q[i, i] != 0.0:
            obj += Q[i, i] * x[i]
    for i in range(n):
        for j in range(i + 1, n):
            if Q[i, j] != 0.0:
                obj += Q[i, j] * x[i] * x[j]
    m.setObjective(obj, GRB.MINIMIZE)

    m._best_obj = float("inf")
    m._last_impr_node = 0
    m._max_no_impr_nodes = max_no_improvement_nodes
    m.optimize(early_stop_callback)
    runtime = time.perf_counter() - t0

    obj_val = m.ObjVal if m.SolCount > 0 else None
    return obj_val, m.Status, runtime


def main():
    sizes = [int(a) for a in sys.argv[1:]] or [5, 7, 11, 28]
    results = []

    for size in sizes:
        d = INPUT_DIM[size]
        folder = f"QUBO/{size}x{size}/{d}x7x10/"
        if not os.path.exists(folder + "QUBO_W.txt"):
            print(f"[skip] {folder} not found")
            continue

        print(f"\n=== {size}x{size} ({d}x7x10) ===", flush=True)
        Q = np.loadtxt(folder + "QUBO_W.txt")
        target = read_target_energy(folder)
        obj_val, status, runtime = solve_qubo_upper_tri(Q, NO_IMPR_NODES)

        row = {
            "instance": f"{size}x{size}",
            "architecture": f"{d}x7x10",
            "variables": int(Q.shape[0]),
            "target_energy": target,
            "best_energy": obj_val,
            "gap": (obj_val - target) if (obj_val is not None and target is not None) else None,
            "fully_feasible": bool(obj_val is not None and target is not None
                                   and abs(obj_val - target) < 1e-6),
            "implied_constraints_satisfied": (abs(target) - (obj_val - target))
                if (obj_val is not None and target is not None) else None,
            "gurobi_status": int(status),
            "runtime_sec": round(runtime, 3),
            "no_improvement_nodes": NO_IMPR_NODES,
        }
        results.append(row)
        print(json.dumps(row, indent=2), flush=True)

        # Checkpoint after every instance. The 28x28 model has 2,235 variables and
        # a dense objective, so it can run for a long time; if it is interrupted the
        # results already obtained for the smaller instances must not be lost.
        with open("gurobi_results.json", "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"[checkpoint] gurobi_results.json now holds {len(results)} instance(s)",
              flush=True)

    print("\nDone. Results in gurobi_results.json")


if __name__ == "__main__":
    main()
