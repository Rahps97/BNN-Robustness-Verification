import time
import numpy as np
import gurobipy as gp
from gurobipy import GRB

Sizes = [5, 7, 11, 28]
DataSize = {5: 31, 7: 63, 11: 127, 28: 1023}
InputSize = Sizes[0]   # change index for 7/11/28
InputDataSize = DataSize[InputSize]
QUBOFolder = f"QUBO/{InputSize}x{InputSize}/{InputDataSize}x7x10/"
Q = np.loadtxt(QUBOFolder + "QUBO_W.txt")  # upper-triangular QUBO matrix

# The no-improvement limit the stagnation callback below enforces. It is NOT the
# same for every instance: the published runs used 10^5 on 5x5 and 7x7 and 10^7 on
# 11x11 and 28x28. This is recoverable from the shipped solver logs, because each
# run ended when the callback fired, so
#
#     (nodes explored) - (node of the last improving incumbent)
#
# lands just above the limit that was in force -- the small overshoot being the
# in-flight nodes that drain after model.terminate():
#
#     5x5      166,017 -    64,212 =    101,805   (10^5 +  1,805)
#     7x7      131,629 -    31,607 =    100,022   (10^5 +     22)
#     11x11 17,297,995 - 7,297,465 = 10,000,530   (10^7 +    530)
#     28x28 10,169,478 -   168,317 = 10,001,161   (10^7 +  1,161)
#
# A single hardcoded 10^7 therefore reproduces only the two larger runs. See the
# "Gurobi Solver" section of README.md.
NoImprovementNodes = {5: 100_000, 7: 100_000, 11: 10_000_000, 28: 10_000_000}
MaxNoImprovementNodes = NoImprovementNodes[InputSize]


def early_stop_callback(model, where):
    """
    Callback that stops the MIP search if there is no improvement
    in the incumbent for more than `model._max_no_impr_nodes` nodes.
    """
    # When a new incumbent is found
    if where == GRB.Callback.MIPSOL:
        obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
        nodecnt = model.cbGet(GRB.Callback.MIPSOL_NODCNT)

        # First incumbent or improved incumbent
        if obj + 1e-9 < model._best_obj:
            model._best_obj = obj
            model._last_impr_node = nodecnt

    # Periodic MIP callback (general progress)
    elif where == GRB.Callback.MIP:
        nodecnt = model.cbGet(GRB.Callback.MIP_NODCNT)

        # If we've gone too many nodes since last improvement, stop
        if nodecnt - model._last_impr_node >= model._max_no_impr_nodes:
            # This triggers GRB.INTERRUPTED status
            model.terminate()



def solve_qubo_upper_tri(
    Q,
    max_no_improvement_nodes=MaxNoImprovementNodes,
    verbose=True,
):
    """
    Solve QUBO: minimize sum_{i<=j} Q[i,j] x_i x_j,  x_i in {0,1}.

    Assumes Q is upper-triangular (including diagonal).

    Stops early if no improvement in the incumbent has been observed
    over `max_no_improvement_nodes` explored MIP nodes.

    Returns
    -------
    x_opt : np.ndarray or None
        Best binary solution found (or None if no feasible solution).
    obj_val : float or None
        Objective value of best solution (or None).
    model : gp.Model
        The underlying Gurobi model.
    runtime_sec : float
        Wall-clock time spent in build + solve.
    """
    Q = np.asarray(Q, dtype=float)
    n = Q.shape[0]
    assert Q.shape[0] == Q.shape[1], "Q must be square"

    t0 = time.perf_counter()

    m = gp.Model("QUBO_upper")

    if not verbose:
        m.Params.OutputFlag = 0

    # Do NOT set TimeLimit here if you are using node-based stopping
    # m.Params.TimeLimit = 0  # <- don't do this

    # Optional: log file
    m.Params.LogFile = "gurobi.log"

    # Binary variables
    x = m.addVars(n, vtype=GRB.BINARY, name="x")

    # Build quadratic objective from upper-triangular Q
    obj = gp.QuadExpr()

    # Diagonal: Q[i,i] * x_i
    for i in range(n):
        qii = Q[i, i]
        if qii != 0.0:
            obj += qii * x[i]

    # Off-diagonal: Q[i,j] * x_i x_j for i < j
    for i in range(n):
        for j in range(i + 1, n):
            qij = Q[i, j]
            if qij != 0.0:
                obj += qij * x[i] * x[j]

    m.setObjective(obj, GRB.MINIMIZE)

    # --- Attach state for callback ---
    m._best_obj = float("inf")
    m._last_impr_node = 0
    m._max_no_impr_nodes = max_no_improvement_nodes

    # Optimize with callback
    m.optimize(early_stop_callback)

    t1 = time.perf_counter()
    runtime_sec = t1 - t0

    # Inspect status / solutions
    print(f"Gurobi status: {m.Status}, SolCount: {m.SolCount}")
    print(f"Runtime: {runtime_sec:.4f} seconds")

    if m.SolCount > 0:
        # Best feasible solution (not necessarily optimal)
        x_opt = np.array([x[i].X for i in range(n)], dtype=int)
        obj_val = m.ObjVal
    else:
        x_opt = None
        obj_val = None

    return x_opt, obj_val, m, runtime_sec

# ---- run it on your QUBO ----
print(f"{InputSize}x{InputSize}: stopping after "
      f"{MaxNoImprovementNodes:,} non-improving nodes")
x_opt, obj_val, model, runtime = solve_qubo_upper_tri(
    Q, max_no_improvement_nodes=MaxNoImprovementNodes)

print("Optimal objective:", obj_val)
print("First 20 bits of x*:", x_opt)
print("Recorded runtime:", runtime, "seconds")
