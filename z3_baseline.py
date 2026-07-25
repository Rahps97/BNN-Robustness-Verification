"""
z3_baseline.py -- exact SMT (Z3) robustness-verification baseline for the
binarized neural networks used in this repository.

This is the reference/exact counterpart to the QUBO pipeline: instead of
encoding the verification problem as a QUBO and handing it to a heuristic
solver, the same problem is encoded directly as an SMT formula and decided
exactly by Z3. It therefore gives ground truth against which the QUBO
solutions (FEM, Gurobi, SA) can be checked.

Run it from the repository root, after the trained network and the dataset
exist (i.e. after DatasetCreation.py and TrainingNN.py):

    python z3_baseline.py validate     # correctness checks vs. brute force
    python z3_baseline.py sweep        # eps = 0..16 sweep with timings
    python z3_baseline.py scaling      # synthetic scaling study (slow)
    python z3_baseline.py all          # validate + sweep

Encoding (exact, no relaxation):
  * inputs are spins x_i in {-1,+1}
  * a Boolean perturbation variable tau_i is created ONLY for the
    perturbable pixels; x_i = -orig_i when tau_i is true, orig_i otherwise
  * cardinality budget  sum_i tau_i <= eps  via Z3's pseudo-Boolean PbLe
  * hidden layer: pre_j = sum_i W1bin[j][i]*x_i ; h_j = +1 if pre_j >= 0
    else -1  (matches Binarize.forward, which maps inp >= 0 to +1)
  * output layer (linear, binarized weights): out_c = sum_j W2bin[c][j]*h_j
  * misclassification: torch.argmax returns the FIRST index attaining the
    maximum, so "argmax(out) != y" is exactly
        OR_{c < y} (out_c >= out_y)  OR  OR_{c > y} (out_c > out_y)

  SAT   -> an adversarial example exists within eps  -> NOT ROBUST
  UNSAT -> no adversarial example within eps         -> ROBUST

The perturbable-pixel set and the verified input are selected exactly as in
QUBOCreator.py, so this script and the QUBO pipeline address the same
instance.
"""
import sys
import time
import itertools
import numpy as np

import z3


# --------------------------------------------------------------------------- #
#  Instance loading -- mirrors the selection logic in QUBOCreator.py
# --------------------------------------------------------------------------- #
def load_5x5_instance():
    """Load the committed 5x5 / 31x7x10 instance as plain NumPy arrays.

    Returns the binarized weights, the verified input in spin form, its
    label, and the list of perturbable pixel indices.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.autograd import Function
    from utils import to_spin

    InputSize = 5
    InputDataSize = 31
    PetrubSize = 16

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
            # Inference only; never called under torch.no_grad().
            return grad_output

    binarize = Binarize.apply

    class BinaryLinear(nn.Linear):
        def __init__(self, i, o, bias=False):
            super().__init__(i, o, bias=bias)

        def forward(self, inp):
            return binarize(F.linear(inp, binarize(self.weight))).to(inp.device)

    class LastLayer(nn.Linear):
        def __init__(self, i, o, bias=False):
            super().__init__(i, o, bias=bias)

        def forward(self, inp):
            return F.linear(inp, binarize(self.weight)).to(inp.device)

    class QUBONet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = BinaryLinear(InputDataSize, 7)
            self.fc4 = LastLayer(7, 10)

        def forward(self, x):
            return torch.argmax(self.fc4(self.fc1(x)))

    net = QUBONet()
    net.load_state_dict(torch.load(
        f"TrainedNN/{InputSize}x{InputSize}/{InputDataSize}x7x10/{InputDataSize}.pth",
        weights_only=True, map_location="cpu"))
    net.eval()

    train_dataloader = torch.load(f"Dataset/{InputSize}x{InputSize}/Train.txt",
                                  weights_only=False)

    # Same perturbable-pixel selection as QUBOCreator.py: the PetrubSize
    # pixels with the smallest mean absolute value over the training set,
    # restricted to the non-padded region.
    flat = train_dataloader.dataset.tensors[0][:, 0:InputSize * InputSize]
    pixels_to_perturb = [int(v) for v in flat.mean(axis=0).abs().topk(
        min(PetrubSize, flat.shape[1]), largest=False).indices.numpy()]

    # Same input selection as QUBOCreator.py: the first correctly classified
    # training sample.
    spin, label = None, None
    with torch.no_grad():
        for image_index in range(len(train_dataloader)):
            item = train_dataloader.dataset[image_index]
            candidate = to_spin(item[0])
            if net(torch.Tensor(np.array([np.array(candidate)]))) == item[1]:
                spin, label = candidate, int(item[1])
                break
    if spin is None:
        raise RuntimeError("no correctly classified training sample found")

    W1 = np.where(net.fc1.weight.detach().numpy() >= 0, 1, -1).astype(np.int64)
    W2 = np.where(net.fc4.weight.detach().numpy() >= 0, 1, -1).astype(np.int64)
    return dict(W1=W1, W2=W2, x0=np.array(spin, dtype=np.int64), y0=label,
                perturb=pixels_to_perturb)


# --------------------------------------------------------------------------- #
#  Reference NumPy forward pass (independent of torch and of Z3)
# --------------------------------------------------------------------------- #
def bnn_forward(W1, W2, x):
    """Binarized forward pass; returns (predicted label, output logits)."""
    pre = W1 @ x
    h = np.where(pre >= 0, 1, -1)
    out = W2 @ h
    return int(np.argmax(out)), out


# --------------------------------------------------------------------------- #
#  Z3 encoding
# --------------------------------------------------------------------------- #
def build_solver(W1, W2, x0, y0, perturb, eps):
    """Build the SMT instance: 'exists a perturbation of at most eps of the
    perturbable pixels that changes the predicted label'."""
    nh, nin = W1.shape
    nout = W2.shape[0]

    s = z3.Solver()
    tau = {i: z3.Bool(f"tau_{i}") for i in perturb}

    # x_i as integer terms in {-1,+1}
    xs = []
    for i in range(nin):
        if i in tau:
            xs.append(z3.If(tau[i], z3.IntVal(int(-x0[i])), z3.IntVal(int(x0[i]))))
        else:
            xs.append(z3.IntVal(int(x0[i])))

    # perturbation budget
    s.add(z3.PbLe([(tau[i], 1) for i in perturb], eps))

    # hidden layer
    h = []
    for j in range(nh):
        pre = z3.Sum([int(W1[j, i]) * xs[i] for i in range(nin)])
        pj = z3.Int(f"pre_{j}")
        s.add(pj == pre)
        hj = z3.Int(f"h_{j}")
        s.add(hj == z3.If(pj >= 0, 1, -1))
        h.append(hj)

    # output layer
    outs = []
    for c in range(nout):
        oc = z3.Int(f"out_{c}")
        s.add(oc == z3.Sum([int(W2[c, j]) * h[j] for j in range(nh)]))
        outs.append(oc)

    # misclassification (argmax tie-break = lowest index)
    clauses = []
    for c in range(nout):
        if c == y0:
            continue
        clauses.append(outs[c] >= outs[y0] if c < y0 else outs[c] > outs[y0])
    s.add(z3.Or(clauses))
    return s, tau


def solve(W1, W2, x0, y0, perturb, eps, timeout_ms=None):
    """Decide robustness at budget eps. Returns (result, seconds, flipped pixels)."""
    s, tau = build_solver(W1, W2, x0, y0, perturb, eps)
    if timeout_ms:
        s.set("timeout", timeout_ms)
    t0 = time.perf_counter()
    r = s.check()
    dt = time.perf_counter() - t0
    flips = None
    if r == z3.sat:
        m = s.model()
        flips = sorted(i for i in perturb
                       if z3.is_true(m.eval(tau[i], model_completion=True)))
    return str(r), dt, flips


# --------------------------------------------------------------------------- #
#  Brute force ground truth
# --------------------------------------------------------------------------- #
def brute_force_min_distance(W1, W2, x0, y0, perturb, max_k):
    """Smallest number of pixel flips that changes the label, by exhaustive
    enumeration. Only tractable for small perturbable sets."""
    for k in range(0, max_k + 1):
        for combo in itertools.combinations(perturb, k):
            x = x0.copy()
            for i in combo:
                x[i] = -x[i]
            lbl, _ = bnn_forward(W1, W2, x)
            if lbl != y0:
                return k, combo
    return None, None


# --------------------------------------------------------------------------- #
#  Commands
# --------------------------------------------------------------------------- #
def cmd_validate(inst):
    """Check the Z3 encoding against an independent NumPy forward pass and
    against exhaustive enumeration of the whole perturbation space."""
    W1, W2, x0, y0, P = inst["W1"], inst["W2"], inst["x0"], inst["y0"], inst["perturb"]
    print("=" * 74)
    print("VALIDATION")
    print("=" * 74)
    print(f"perturbable pixels : {P}")
    print(f"clean label        : {y0}")
    lbl, out = bnn_forward(W1, W2, x0)
    print(f"numpy clean pred   : {lbl}  logits={out.tolist()}")
    assert lbl == y0, "numpy forward disagrees with the stored clean label"

    ok = True
    # (a) eps = 8 (the budget used by QUBOCreator.py) must be SAT with a
    #     genuine counterexample
    r, dt, flips = solve(W1, W2, x0, y0, P, 8)
    print(f"\n(a) eps=8  -> {r}  ({dt:.4f}s)  flips={flips}")
    if r != "sat":
        ok = False
    else:
        x = x0.copy()
        for i in flips:
            x[i] = -x[i]
        nl, nout = bnn_forward(W1, W2, x)
        ham = int((x != x0).sum())
        print(f"    independent numpy re-eval: label {y0} -> {nl}, "
              f"hamming={ham}, logits={nout.tolist()}")
        ok &= (nl != y0) and (ham <= 8) and all(i in P for i in flips)

    # (b) eps = 2 must be UNSAT (robust)
    r2, dt2, _ = solve(W1, W2, x0, y0, P, 2)
    print(f"(b) eps=2  -> {r2}  ({dt2:.4f}s)   [expected unsat]")
    ok &= (r2 == "unsat")

    # (c) eps = 3 must be SAT (the true minimum adversarial distance)
    r3, dt3, f3 = solve(W1, W2, x0, y0, P, 3)
    print(f"(c) eps=3  -> {r3}  ({dt3:.4f}s)  flips={f3}   [expected sat]")
    ok &= (r3 == "sat")
    if r3 == "sat":
        x = x0.copy()
        for i in f3:
            x[i] = -x[i]
        nl, _ = bnn_forward(W1, W2, x)
        print(f"    numpy re-eval: {y0} -> {nl}, hamming={int((x != x0).sum())}")
        ok &= nl != y0

    # (d) exhaustive brute force over the full 2^|P| space
    t0 = time.perf_counter()
    k, combo = brute_force_min_distance(W1, W2, x0, y0, P, len(P))
    bt = time.perf_counter() - t0
    print(f"\n(d) brute force over 2^{len(P)}: min adversarial hamming = {k} "
          f"via {list(combo)}  ({bt:.1f}s)")
    ok &= (k == 3)

    print("\nVALIDATION: " + ("PASS" if ok else "FAIL"))
    return ok


def cmd_sweep(inst, reps=7, eps_max=16):
    """Decide robustness at every budget from 0 to eps_max and time it."""
    W1, W2, x0, y0, P = inst["W1"], inst["W2"], inst["x0"], inst["y0"], inst["perturb"]
    print("=" * 74)
    print(f"EPS SWEEP  (5x5 / 31x7x10, |perturbable| = {len(P)}, {reps} reps each)")
    print("=" * 74)
    print(f"{'eps':>4} {'result':>7} {'verdict':>9} {'mean_s':>10} {'min_s':>10} "
          f"{'max_s':>10}  witness")
    rows = []
    for eps in range(0, eps_max + 1):
        ts, res, flips = [], None, None
        for _ in range(reps):
            r, dt, f = solve(W1, W2, x0, y0, P, eps)
            ts.append(dt)
            res, flips = r, f
        verdict = "NOT ROBUST" if res == "sat" else "ROBUST"
        rows.append((eps, res, float(np.mean(ts)), min(ts), max(ts)))
        print(f"{eps:>4} {res:>7} {verdict:>9} {np.mean(ts):>10.5f} {min(ts):>10.5f} "
              f"{max(ts):>10.5f}  {flips if flips else ''}")
    tot = sum(r[2] for r in rows)
    print(f"\nsum of mean runtimes over the whole eps=0..{eps_max} sweep: {tot:.4f}s")
    return rows


# --------------------------------------------------------------------------- #
#  Synthetic scaling study
# --------------------------------------------------------------------------- #
def make_synthetic(nin, nhid, nout, n_perturb, seed):
    """Random BNN of the given shape, for studying how Z3 runtime scales."""
    rng = np.random.default_rng(seed)
    W1 = rng.choice([-1, 1], size=(nhid, nin)).astype(np.int64)
    W2 = rng.choice([-1, 1], size=(nout, nhid)).astype(np.int64)
    x0 = rng.choice([-1, 1], size=nin).astype(np.int64)
    y0, _ = bnn_forward(W1, W2, x0)
    perturb = sorted(rng.choice(nin, size=n_perturb, replace=False).tolist())
    return dict(W1=W1, W2=W2, x0=x0, y0=y0, perturb=perturb)


def find_robust_eps(inst, eps_cap, timeout_ms):
    """Largest eps (<= eps_cap) at which the instance is still robust (UNSAT),
    found by scanning upward from 0. Returns (eps_robust, eps_first_sat)."""
    W1, W2, x0, y0, P = inst["W1"], inst["W2"], inst["x0"], inst["y0"], inst["perturb"]
    last_unsat = -1
    for eps in range(0, eps_cap + 1):
        r, dt, _ = solve(W1, W2, x0, y0, P, eps, timeout_ms=timeout_ms)
        if r == "unsat":
            last_unsat = eps
        elif r == "sat":
            return last_unsat, eps
        else:
            return last_unsat, None
    return last_unsat, None


def cmd_scaling(reps=3, timeout_ms=600_000):
    print("=" * 74)
    print("SYNTHETIC SCALING STUDY -- how does Z3 UNSAT time grow with")
    print("the number of perturbable bits?")
    print("=" * 74)
    configs = [
        # (nin, nhid, nout, n_perturb)
        # -- axis 1: grow the perturbation space at the paper's fixed width (7)
        (31,   7, 10,  16),
        (63,   7, 10,  32),
        (127,  7, 10,  64),
        (1023, 7, 10, 128),
        (1023, 7, 10, 256),
        # -- axis 2: grow the hidden width
        (1023, 15, 10, 256),
        (1023, 31, 10, 256),
        (1023, 63, 10, 256),
    ]
    print(f"{'nin':>5} {'nhid':>5} {'|P|':>5} {'seed':>5} {'eps':>4} {'result':>8} "
          f"{'mean_s':>10}")
    results = []
    for (nin, nhid, nout, npert) in configs:
        for seed in (0, 1, 2):
            inst = make_synthetic(nin, nhid, nout, npert, seed)
            # locate the robust/non-robust boundary cheaply
            eps_rob, eps_sat = find_robust_eps(inst, min(npert, 12), timeout_ms=60_000)
            for eps, tag in ((eps_rob, "UNSAT-tight"), (eps_sat, "SAT-first")):
                if eps is None or eps < 0:
                    continue
                ts = []
                r = None
                for _ in range(reps):
                    r, dt, _ = solve(inst["W1"], inst["W2"], inst["x0"], inst["y0"],
                                     inst["perturb"], eps, timeout_ms=timeout_ms)
                    ts.append(dt)
                print(f"{nin:>5} {nhid:>5} {npert:>5} {seed:>5} {eps:>4} {r:>8} "
                      f"{np.mean(ts):>10.4f}   ({tag})")
                results.append((nin, nhid, npert, seed, eps, r, float(np.mean(ts))))
    return results


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "scaling":
        # Purely synthetic; does not need the trained network or the dataset.
        cmd_scaling()
        return

    inst = load_5x5_instance()
    if cmd in ("validate", "all"):
        cmd_validate(inst)
    if cmd in ("sweep", "all"):
        print()
        cmd_sweep(inst)


if __name__ == "__main__":
    main()
