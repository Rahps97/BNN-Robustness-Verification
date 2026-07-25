import os
import re
import sys
import math
import time
import dimod
import pickle
import warnings
import subprocess
import numpy as np
from pathlib import Path
import datetime, torch, time
import multiprocessing, threading
from multiprocessing import Manager
from collections import defaultdict
from torch.quasirandom import SobolEngine
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings('ignore')

# Force UTF-8 output on Windows so emojis don't crash logging
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def get_idle_gpus(threshold_mb=500):
    """Return list of GPU indices with memory usage less than 'threshold_mb'.

    Returns an empty list when nvidia-smi is not installed or produces no
    usable output (CPU-only or non-NVIDIA machines), so the caller can fall
    back to the CPU instead of crashing.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
    except OSError:
        return []

    idle = []
    for line in result.stdout.strip().split("\n"):
        try:
            idx, used = map(int, re.split(r",\s*", line))
        except ValueError:
            continue
        if used < threshold_mb:
            idle.append(idx)

    return idle

def beta_range(h, J):
    """Determine the starting and ending beta from h J

    Args:
        h (dict)

        J (dict)

    Assume each variable in J is also in h.

    We use the minimum bias to give a lower bound on the minimum energy gap, such at the
    final sweeps we are highly likely to settle into the current valley.
    """
    # Get nonzero, absolute biases
    abs_h = [abs(hh) for hh in h.values() if hh != 0]
    abs_J = [abs(jj) for jj in J.values() if jj != 0]
    abs_biases = abs_h + abs_J

    if not abs_biases:
        return [0.1, 1.0]

    # Rough approximation of min change in energy when flipping a qubit
    min_delta_energy = min(abs_biases)

    # Combine absolute biases by variable
    abs_bias_dict = defaultdict(int, {k: abs(v) for k, v in h.items()})
    for (k1, k2), v in J.items():
        abs_bias_dict[k1] += abs(v)
        abs_bias_dict[k2] += abs(v)

    # Find max change in energy when flipping a single qubit
    max_delta_energy = max(abs_bias_dict.values())

    # Selecting betas based on probability of flipping a qubit
    # Hot temp: We want to scale hot_beta so that for the most unlikely qubit flip, we get at least
    # 50% chance of flipping.(This means all other qubits will have > 50% chance of flipping
    # initially.) Most unlikely flip is when we go from a very low energy state to a high energy
    # state, thus we calculate hot_beta based on max_delta_energy.
    #   0.50 = exp(-hot_beta * max_delta_energy)
    #
    # Cold temp: Towards the end of the annealing schedule, we want to minimize the chance of
    # flipping. Don't want to be stuck between small energy tweaks. Hence, set cold_beta so that
    # at minimum energy change, the chance of flipping is set to 1%.
    #   0.01 = exp(-cold_beta * min_delta_energy)
    hot_beta = np.log(2) / max_delta_energy
    cold_beta = np.log(10000) / min_delta_energy

    return [hot_beta, cold_beta]


class FEM_Batched:
    def __init__(
        self, J, h_vec, betamode,
        Tmax, Tmin, N_step, lr, trials, c_grad,
        dev="cuda:0", dtype=torch.float32,
        seeds=-1, h_factor=0.001,
        optimizer="rmsprop", params=None,
        default_chunk_size=None,  # 👈 optional override; otherwise auto
    ):
        """
        Batched FEM without bias node.
        Runs K parameter sets in parallel on the same GPU.

        Design goal:
          - All tensors arrive on CPU.
          - ONLY this class moves data to GPU and chooses chunk_size.
        """
        self.dev = torch.device(dev)
        self.dtype = dtype
        self.trials = trials          # T: number of trials
        self.seeds = seeds
        self.h_factor = h_factor
        self.opt_mode = optimizer

        # --------------------------------------------------------------
        # Move essentials to GPU ONCE here
        # --------------------------------------------------------------
        self.J = J.to(self.dev).to(self.dtype)         # [N, N]
        self.h_vec = h_vec.to(self.dev).to(self.dtype) # [N]
        self.N = self.J.shape[0]

        # K = number of parameter candidates in this batch
        K = Tmax.shape[0]
        self.K = K

        Tmax = Tmax.to(self.dev)
        Tmin = Tmin.to(self.dev)
        lr   = lr.to(self.dev)
        c_grad = c_grad.to(self.dev)
        self.lr, self.c_grad = lr, c_grad
        self.params = params.to(self.dev)

        # --- Batched beta schedule [K, N_step] ---
        steps = torch.linspace(0, 1, N_step, device=self.dev).unsqueeze(0)
        if betamode == "inv":
            self.beta = 1.0 / (Tmax[:, None] * (1 - steps) + Tmin[:, None] * steps)
        elif betamode == "exp":
            self.beta = torch.exp(
                torch.log(Tmin)[:, None] * (1 - steps) + torch.log(Tmax)[:, None] * steps
            )
        elif betamode == "geo":
            self.beta = 1.0 / torch.exp(
                torch.log(Tmax)[:, None] * (1 - steps) + torch.log(Tmin)[:, None] * steps
            )
        else:
            raise ValueError(f"Unknown betamode {betamode}")

        # --- Initialize internal state (allocates [K, N, T] tensors) ---
        self.initialize()

        # --------------------------------------------------------------
        # Decide default chunk size AFTER big tensors are on GPU
        # --------------------------------------------------------------
        if default_chunk_size is None:
            self.default_chunk_size = self._auto_chunk_size(
                safety=0.7, max_chunk=self.trials
            )
        else:
            self.default_chunk_size = int(default_chunk_size)

        print(
            f"[FEM] dev={self.dev}, K={self.K}, N={self.N}, T={self.trials} "
            f"→ default_chunk_size={self.default_chunk_size}"
        )

    # ------------------------------------------------------------------
    def initialize(self):
        """
        Hash-based deterministic RNG for FEM initialization.
        Each candidate k gets its own seed, producing unique trajectories.
        No dependence on PyTorch RNG state. Fully GPU-parallel.
        """

        K, N, T = self.K, self.N, self.trials

        # Make sure seeds is a valid tensor
        if isinstance(self.seeds, (int, float)):
            # NOTE: debug print kept from your original code
            print("SEEDS DIDN'T WORK")
            base_seed = int(self.seeds) if self.seeds > 0 else 0
            seeds_k = torch.arange(
                base_seed + 1, base_seed + 1 + K,
                device=self.dev, dtype=torch.float32
            )
        else:
            # if seeds tensor already passed
            seeds_k = torch.as_tensor(self.seeds, device=self.dev, dtype=torch.float32)
            if seeds_k.numel() != K:
                raise ValueError("FEM_Batched: seeds tensor must have K seeds.")

        # Expand seeds → shape [K, 1, 1]
        seeds_k = seeds_k.view(K, 1, 1)

        # Build index grids for i (nodes) and t (trials)
        i_idx = torch.arange(N, device=self.dev, dtype=torch.float32).view(1, N, 1)
        t_idx = torch.arange(T, device=self.dev, dtype=torch.float32).view(1, 1, T)

        # --- Hash-based RNG (fast, deterministic, parallel) ---
        # mix constants (large primes)
        A = torch.tensor(12.9898, device=self.dev)
        B = torch.tensor(78.233, device=self.dev)
        C = torch.tensor(37.719, device=self.dev)

        # Generate hashed uniform noise in [0,1)
        noise = torch.sin(seeds_k * A + i_idx * B + t_idx * C)
        noise = noise - torch.floor(noise)   # frac(x) = x - floor(x)

        # Map to [-1, 1]
        noise = 2.0 * noise - 1.0

        # Scale by h_factor → final logits h
        self.h = self.h_factor * noise.to(self.dtype)

        # Initialize optimizer states
        self.opt_v = torch.zeros_like(self.h)
        self.opt_m = torch.zeros_like(self.h)

        # Optimizer parameters broadcasted correctly
        if self.opt_mode == "rmsprop":
            alpha, wd, mom = self.params.T
            self.alpha, self.wd, self.mom = (
                alpha[:, None, None],
                wd[:, None, None],
                mom[:, None, None],
            )
        elif self.opt_mode == "adam":
            wd, beta1, beta2 = self.params.T
            self.wd, self.beta1, self.beta2 = (
                wd[:, None, None],
                beta1[:, None, None],
                beta2[:, None, None],
            )
        else:
            raise ValueError("Unsupported optimizer")

    # ------------------------------------------------------------------
    def _auto_chunk_size(self, safety=0.5, max_chunk=None, min_chunk=8):
        """
        Decide a trial chunk_size based on CURRENT free GPU memory and
        the already-allocated [K, N, T] tensors.

        Called AFTER initialize(), so base tensors (h, opt_v, opt_m, beta, etc.)
        already live on the GPU. We only estimate memory needed for temporaries
        like p, Jp, grad of shape [K, N, chunk_size].
        """
        try:
            torch.cuda.set_device(self.dev)
            free_bytes, total_bytes = torch.cuda.mem_get_info()
        except Exception:
            # If mem_get_info is not available, just pick something modest
            return min(max_chunk or self.trials, self.trials)

        K, N, T = self.h.shape
        dtype_bytes = torch.finfo(self.dtype).bits // 8

        # Rough estimate:
        #  p, Jp, grad + some optimizer views ~ 8 * [K, N, chunk_size]
        factor = 8
        bytes_per_unit_t = dtype_bytes * max(K, 1) * max(N, 1) * factor

        safe_bytes = int(free_bytes * safety)
        if bytes_per_unit_t <= 0 or safe_bytes <= 0:
            return min(max_chunk or T, T)

        max_chunk_by_mem = safe_bytes // bytes_per_unit_t
        if max_chunk_by_mem < 1:
            max_chunk_by_mem = 1

        if max_chunk is None:
            max_chunk = T

        chunk = int(
            max(
                min_chunk,
                min(max_chunk, T, max_chunk_by_mem),
            )
        )
        return chunk

    # ------------------------------------------------------------------
    def update(self, chunk_size=None):
        """
        Memory-safe update loop.

        - If chunk_size is None → use self.default_chunk_size chosen
          via _auto_chunk_size() based on real GPU memory.
        - Otherwise, use the provided chunk_size (optional override).
        """
        nsteps = self.beta.shape[1]
        K, N, T = self.h.shape

        if chunk_size is None:
            chunk_size = self.default_chunk_size

        for step in range(nsteps):
            beta_t = self.beta[:, step][:, None, None]   # [K,1,1]

            t0 = 0
            while t0 < T:
                t1 = min(t0 + chunk_size, T)
                try:
                    h_chunk = self.h[:, :, t0:t1]

                    p = torch.sigmoid(h_chunk)
                    # matrix multiply on smaller block
                    Jp = torch.matmul(self.J, torch.round(p))
                    grad = -self.c_grad[:, None, None] * (
                        (2 * Jp + self.h_vec[:, None]) + h_chunk / beta_t
                    ) * p * (1 - p)

                    if self.opt_mode == "rmsprop":
                        v = self.opt_v[:, :, t0:t1]
                        m = self.opt_m[:, :, t0:t1]

                        v = self.alpha * v + (1 - self.alpha) * grad.pow(2)
                        h_chunk -= self.lr[:, None, None] * grad / (torch.sqrt(v) + 1e-8)
                        h_chunk *= (1 - self.wd)
                        h_chunk += self.mom * m
                        m = h_chunk.clone()

                        # write back
                        self.opt_v[:, :, t0:t1] = v
                        self.opt_m[:, :, t0:t1] = m
                        self.h[:, :, t0:t1] = h_chunk

                    elif self.opt_mode == "adam":
                        m = self.opt_m[:, :, t0:t1]
                        v = self.opt_v[:, :, t0:t1]

                        m = self.beta1 * m + (1 - self.beta1) * grad
                        v = self.beta2 * v + (1 - self.beta2) * grad.pow(2)
                        m_hat = m / (1 - self.beta1)
                        v_hat = v / (1 - self.beta2)
                        h_chunk -= self.lr[:, None, None] * m_hat / (torch.sqrt(v_hat) + 1e-8)
                        h_chunk *= (1 - self.wd)

                        # write back
                        self.opt_m[:, :, t0:t1] = m
                        self.opt_v[:, :, t0:t1] = v
                        self.h[:, :, t0:t1] = h_chunk

                    t0 = t1

                except torch.cuda.OutOfMemoryError:
                    # 🔻 Adaptive backoff if we guessed too aggressively
                    torch.cuda.empty_cache()
                    if chunk_size <= 1:
                        raise
                    chunk_size = max(1, chunk_size // 2)
                    print(f"[FEM] ⚠️ OOM at step={step}, reducing chunk_size to {chunk_size}")

    # ------------------------------------------------------------------
    def calc_energy(self):
        """Return per-candidate, per-trial energy and config."""
        p = torch.sigmoid(self.h)
        config = torch.round(p)                       # [K, N, trials]
        # E = -hᵀs - ½ sᵀJ s
        Js = torch.matmul(self.J, config)             # [N,N]·[K,N,trials] → broadcasted
        E_pair =  -(config * Js).sum(1)               # [K,trials]
        E_field = -(config * self.h_vec[None, :, None]).sum(1)
        energy = E_pair + E_field
        return energy.cpu().numpy(), config.cpu().numpy()



def rule_no_limit(params, down_limit, up_limit, search_precision):
    # for hyper-parameters without numerical range limit
    return np.linspace(params * down_limit, params * up_limit, search_precision)

def rule_limit(params, down_limit, up_limit, limit_val, search_precision):
    # for hyper-parameters with numerical range limit
    limit = params * up_limit if params * up_limit < limit_val else limit_val
    return np.linspace(params * down_limit, limit, search_precision)

def build_candidates(params_dic, param, search_precision):
    p = params_dic[param]
    if p["range_rule"] == "no_limit":
        vals = rule_no_limit(p["val"], p["down_limit"], p["up_limit"], search_precision)
    else:
        vals = rule_limit(p["val"], p["down_limit"], p["up_limit"], p["limit_val"], search_precision)
    return torch.as_tensor(vals, dtype=torch.float32)

def collect_opt_params(params_dic, optimizer, global_params_backup=None):
    """
    Safely collects optimizer parameters, with fallbacks for missing keys.
    Works even when params_dic only has a subset (multi-GPU split).
    """
    def get_val(name, default=1.0):
        if name in params_dic:
            return float(params_dic[name]["val"])
        elif global_params_backup and name in global_params_backup:
            return float(global_params_backup[name]["val"])
        else:
            return default

    if optimizer.lower() == "rmsprop":
        alpha = get_val("alpha", 0.9)
        wd    = get_val("wd", 1e-4)
        mom   = get_val("mom", 0.0)
        return torch.tensor([alpha, wd, mom], dtype=torch.float32)

    elif optimizer.lower() == "adam":
        wd    = get_val("wd", 1e-4)
        b1    = get_val("beta1", 0.9)
        b2    = get_val("beta2", 0.999)
        return torch.tensor([wd, b1, b2], dtype=torch.float32)

    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")


def param_search_batched(seeds, J, h, betamode,
                         Tmax, Tmin, N_step, lr, batch, c_grad,
                         dev, opt_params, optimizer,
                         trial_chunk_size=None):
    """
    Batched parameter evaluation for the FEM solver.

    Design:
      - All inputs assumed on CPU (numpy or torch).
      - No .to(device) here: FEM_Batched owns GPU transfers + chunking.

    Args:
        trial_chunk_size:
            - If None → FEM_Batched chooses chunk_size automatically.
            - If int  → overrides FEM's default chunk_size.
    """
    # Ensure torch tensors on CPU
    J = torch.as_tensor(J, dtype=torch.float32)
    h = torch.as_tensor(h, dtype=torch.float32)
    Tmax = torch.as_tensor(Tmax, dtype=torch.float32)
    Tmin = torch.as_tensor(Tmin, dtype=torch.float32)
    lr = torch.as_tensor(lr, dtype=torch.float32)
    c_grad = torch.as_tensor(c_grad, dtype=torch.float32)
    opt_params = torch.as_tensor(opt_params, dtype=torch.float32)

    # Physically enforce valid scheduling (Tmax >= Tmin)
    delta_T = torch.abs(Tmax - Tmin) + 1e-8
    Tmax = Tmin + delta_T

    # Instantiate solver on the target device
    solver = FEM_Batched(
        -J, -h, betamode,
        Tmax, Tmin, N_step, lr,
        batch, c_grad, dev=dev, dtype=torch.float32,
        seeds=seeds, optimizer=optimizer, params=opt_params,
        default_chunk_size=trial_chunk_size,  # None → auto
    )

    # Run FEM updates (solver handles chunking internally)
    solver.update(chunk_size=None if trial_chunk_size is None else trial_chunk_size)

    # Compute energies
    energy, config = solver.calc_energy()  # energy: [K, trials]
    energy_t = torch.from_numpy(energy)
    config_t = torch.from_numpy(config)

    if energy_t.numel() == 0:
        return torch.tensor([]), torch.tensor([]), Tmin.cpu(), Tmax.cpu()

    # Find per-candidate minima
    min_e_per_cand, best_trial_idx = energy_t.min(dim=1)
    best_config_vectors = torch.stack([
        config_t[k, :, best_trial_idx[k]] for k in range(config_t.shape[0])
    ])

    return min_e_per_cand, best_config_vectors, Tmin.cpu(), Tmax.cpu()



def fast_batched_coord_search(
    J_matrix, h_vec, params_dic, betamode,
    N_step, batch, dev, optimizer,
    search_precision=100, loop=50, seed_base=1234,
    param_chunk=50,
    trial_chunk_size=None,    # 👈 None → FEM auto-chunk; int → override
    precision_decay=1,        # how much to shrink precision for less important params
    top_keep=7,               # number of top-changing params to prioritize
    global_params_backup=None # 🔸 for multi-GPU fallback
):
    """
    Adaptive, GPU-aware coordinate search for FEM.
    Works with parameter-split multi-GPU launcher.

    DESIGN:
      - All tensors stay on CPU here.
      - Only FEM_Batched touches CUDA and chooses chunk_size by default.
    """

    sobol = SobolEngine(dimension=1, scramble=True, seed=seed_base)

    # ---------------------------------------------------------------------
    # Prepare CPU tensors for J and h (NO device move here)
    # ---------------------------------------------------------------------
    J = torch.as_tensor(J_matrix, dtype=torch.float32)
    h = torch.as_tensor(h_vec, dtype=torch.float32)

    # ---------------------------------------------------------------------
    # Helper: Safe parameter getter (handles missing keys per GPU)
    # ---------------------------------------------------------------------
    def get_param_value(name, default=1.0):
        """Return parameter value from local dict or backup."""
        if name in params_dic:
            return float(params_dic[name]["val"])
        elif global_params_backup and name in global_params_backup:
            return float(global_params_backup[name]["val"])
        else:
            return default

    # ---------------------------------------------------------------------
    # Lightweight logger (dataset-specific folder)
    # ---------------------------------------------------------------------
    ds_str = f"{InputSize}x{InputSize}"
    log_dir = f"FEM_Solutions/logs_{ds_str}"
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, f"log_{ds_str}_{str(dev).replace(':','_')}.log")

    log_file = open(log_path, "a", encoding="utf-8", errors="replace", buffering=4096)
    def log(msg):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        log_file.write(f"[{ts}] {msg}\n")
        log_file.flush()

    chunk_str = trial_chunk_size if trial_chunk_size is not None else "auto"
    log(f"🟢 GPU {dev} — active parameters: {list(params_dic.keys())}")
    log(f"💾 GPU {dev} | using chunks: param_chunk={param_chunk}, "
        f"trial_chunk_size={chunk_str}")

    # ---------------------------------------------------------------------
    # Search state
    # ---------------------------------------------------------------------
    best_min_e = float("inf")
    best_params_dic = {k: v["val"] for k, v in params_dic.items()}
    best_config_ever = None
    param_scores = {k: 1.0 for k in params_dic.keys()}  # importance weights

    start = time.time()

    # ---------------------------------------------------------------------
    # Main optimization loop
    # ---------------------------------------------------------------------
    for it in range(loop):

        log(f"\n=== Round {it+1}/{loop} | GPU {dev} "
            f"→ param_chunk={param_chunk}, trial_chunk_size={chunk_str}")

        # determine which params to emphasize this round
        sorted_params = sorted(param_scores.items(), key=lambda x: x[1], reverse=True)
        top_params = [p for p, _ in sorted_params[:top_keep]]

        round_best_e, round_best_param, round_best_val = float("inf"), None, None

        # ================================================================
        # Parameter sweeps
        # ================================================================
        for param in params_dic.keys():
            # variable precision
            base_precision = search_precision
            local_precision = int(
                max(5, base_precision * (1 if param in top_params else precision_decay))
            )

            cand_vals = build_candidates(params_dic, param, local_precision)  # CPU tensor
            K_total = cand_vals.numel()
            if K_total == 0:
                continue

            sweep_best_e, sweep_best_val, sweep_best_seed, sweep_best_cfg = (
                float("inf"), None, None, None
            )

            # --------------------------------------------------------------
            # Candidate batching (param_chunk → K per FEM batch)
            # --------------------------------------------------------------
            for start_idx in range(0, K_total, param_chunk):
                end_idx = min(start_idx + param_chunk, K_total)
                cand_subset = cand_vals[start_idx:end_idx]
                K = cand_subset.numel()
                if K == 0:
                    continue

                # random Sobol seeds
                seeds = (
                    sobol.draw(K).squeeze(-1) * 4999.0 + 1.0
                ).round().to(torch.int64).cpu().numpy()

                # create per-candidate parameter tensors with fallbacks (CPU)
                Tmax = torch.full((K,), get_param_value("Tmax"), dtype=torch.float32)
                Tmin = torch.full((K,), get_param_value("Tmin"), dtype=torch.float32)
                lr   = torch.full((K,), get_param_value("lr"),   dtype=torch.float32)
                cg   = torch.full((K,), get_param_value("c_grad"), dtype=torch.float32)

                if param == "Tmax":   Tmax = cand_subset
                elif param == "Tmin": Tmin = cand_subset
                elif param == "lr":   lr   = cand_subset
                elif param == "c_grad": cg = cand_subset

                opt_params = collect_opt_params(
                    params_dic, optimizer, global_params_backup
                ).repeat(K, 1)

                if optimizer == "rmsprop":
                    if param == "alpha": opt_params[:, 0] = cand_subset
                    if param == "wd":    opt_params[:, 1] = cand_subset
                    if param == "mom":   opt_params[:, 2] = cand_subset
                elif optimizer == "adam":
                    if param == "wd":    opt_params[:, 0] = cand_subset
                    if param == "beta1": opt_params[:, 1] = cand_subset
                    if param == "beta2": opt_params[:, 2] = cand_subset

                # run solver batch (FEM_Batched will move to GPU + chunk internally)
                min_e_per_cand, best_cfg_vecs, Tmin_valid, Tmax_valid = param_search_batched(
                    seeds=seeds, J=J, h=h, betamode=betamode,
                    Tmax=Tmax, Tmin=Tmin, N_step=N_step, lr=lr,
                    batch=batch, c_grad=cg, dev=dev,
                    opt_params=opt_params, optimizer=optimizer,
                    trial_chunk_size=trial_chunk_size,   # None → auto
                )

                if min_e_per_cand.numel() == 0:
                    torch.cuda.empty_cache()
                    continue

                best_idx = torch.argmin(min_e_per_cand)
                best_val = float(cand_subset[best_idx].detach().cpu())
                best_e   = float(min_e_per_cand[best_idx].detach().cpu())

                if best_e <= sweep_best_e:
                    sweep_best_e, sweep_best_val = best_e, best_val
                    sweep_best_seed = int(seeds[best_idx])
                    sweep_best_cfg  = best_cfg_vecs[best_idx].detach().cpu().numpy()

                torch.cuda.empty_cache()

            # Commit this parameter’s result
            if sweep_best_val is not None:
                params_dic[param]["val"] = sweep_best_val
                if sweep_best_e <= round_best_e:
                    round_best_e, round_best_param, round_best_val = (
                        sweep_best_e,
                        param,
                        sweep_best_val,
                    )
                if sweep_best_e <= best_min_e:
                    best_min_e = sweep_best_e
                    best_params_dic = {k: v["val"] for k, v in params_dic.items()}
                    best_params_dic["seed"] = sweep_best_seed
                    best_config_ever = sweep_best_cfg

        # Update parameter importance
        for p in param_scores.keys():
            if p == round_best_param:
                param_scores[p] = param_scores[p] * 1.1 + 0.2
            else:
                param_scores[p] *= 0.95

        log(
            f"Round {it+1}: best {round_best_param}={round_best_val:.6g} "
            f"| E={round_best_e:.6g} | global={best_min_e:.6g}"
        )

    # ---------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------
    tts = time.time() - start
    log(f"\n========== Final Results (GPU {dev}) ==========")
    log(f"Best Energy: {best_min_e:.6f}")
    for k, v in best_params_dic.items():
        log(f"  {k}: {v}")
    log(f"Runtime: {tts/60:.2f} min\n")
    log_file.close()

    return best_params_dic, best_min_e, best_config_ever


# ============================================================
#  🪵  PER-GPU LOGGER (simple timestamp + device prefix)
# ============================================================
class GPULogger:
    _lock = threading.Lock()

    def __init__(self, device, data_size=None, base_folder="logs"):
        """
        Create a per-GPU log inside a dataset-specific folder.
        Example:
            logs_28x28/FEM_cuda_4_20251031_2301.log
        """
        import os, datetime, sys, threading

        folder = f"{base_folder}"
        os.makedirs(folder, exist_ok=True)

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.device = device
        self.path = os.path.join(folder, f"FEM_{device.replace(':','_')}_{ts}.log")
        self.file = open(self.path, "a", encoding="utf-8", errors="replace", buffering=1)

    def write(self, msg="", end="\n"):
        import datetime, sys
        ts = datetime.datetime.now().strftime("[%H:%M:%S]")
        line = f"{ts} [{self.device}] {msg}{end}"
        with GPULogger._lock:
            sys.stdout.write(line)
            sys.stdout.flush()
            self.file.write(line)
            self.file.flush()

    def close(self):
        self.file.close()


# ============================================================
#  🪵  GLOBAL SUMMARY LOGGER (clean “INFO | GPU_x” format)
# ============================================================
def setup_summary_logger(data_size=None, base_folder="logs"):
    """
    Creates a global summary logger inside dataset-specific folder.
    Example:
        logs_28x28/FEM_summary_20251031_2301.log
    """
    import os, datetime, logging, sys

    log_dir = f"{base_folder}"
    os.makedirs(log_dir, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"FEM_summary_{ts}.log")

    logger = logging.getLogger("GPU_summary")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    logger.info(f"🚀 Global summary logger initialized in {log_dir}/")
    return logger, log_path


def worse_than_global(local_e: float,
                      global_e: float,
                      rel_margin: float,
                      abs_floor: float = 1.0,
                      abs_margin: float = 0.0):
    """
    Robust comparison for minimization problems.
    Returns (should_reseed, rel_gap) where rel_gap = fractional gap (0.03 = 3%).
    - Works for both negative and positive energies.
    - Handles sign crossings and near-zero global values safely.
    """
    if not (math.isfinite(local_e) and math.isfinite(global_e)):
        return False, 0.0

    regret = local_e - global_e        # >0 means local is worse
    if regret <= 0:
        return False, 0.0

    scale = max(abs(global_e), abs_floor)
    rel_gap = regret / scale
    should = (rel_gap > rel_margin) or (abs_margin > 0.0 and regret > abs_margin)
    return should, rel_gap


def _run_search_on_device(
    device,
    J_matrix, h_vec, params_dic, betamode,
    N_step, batch, optimizer,
    search_precision, total_rounds, seed_base,
    manager_state,
    BASELINE_ROUNDS=20,
    SYNC_INTERVAL=10, EARLY_STOP_MARGIN=0.03,
    summary_logger=None,
    gpu_logger=None,
    param_chunk=50,
    trial_chunk_size=300,   # None → FEM auto-chunk
):
    """
    GPU worker with deterministic reseed grid + hybrid early-restart.
    All GPUs run for the same number of total_rounds,
    but those that fall behind the global best (by > EARLY_STOP_MARGIN)
    automatically reseed around the best parameter grid.
    """

    log = gpu_logger.write if gpu_logger else (lambda msg: print(f"[{device}] {msg}"))
    gpu_name = f"GPU_{device.split(':')[-1]}"
    summary_logger.info(f"🧠 {gpu_name} initialized and ready")

    # Per-GPU lifetime best
    local_params   = {k: v["val"] for k, v in params_dic.items()}
    local_best_e   = float("inf")
    local_best_cfg = None

    start_time = time.time()
    last_reseed_round = -999  # cooldown tracker

    for round_idx in range(1, total_rounds + 1):
        log(f"\n=== Round {round_idx}/{total_rounds} ===")

        device_index = device.split(":")[-1]
        gpu_id = int(device_index) if device_index.isdigit() else 0
        gpu_seed_base = seed_base + gpu_id * 100_000 + round_idx * 2_000_000

        # --- Run a local FEM parameter search block (BASELINE_ROUNDS) ---
        best_params_dic, best_min_e, best_config_ever = fast_batched_coord_search(
            J_matrix, h_vec, params_dic, betamode,
            N_step, batch, device, optimizer,
            search_precision=search_precision,
            loop=BASELINE_ROUNDS, seed_base=gpu_seed_base,
            param_chunk=param_chunk,
            trial_chunk_size=trial_chunk_size,  # None → FEM auto-chunk
            global_params_backup=None,
        )

        # --- Local improvement (lifetime best on this GPU) ---
        if best_min_e < local_best_e:
            local_best_e   = best_min_e
            local_params   = best_params_dic.copy()
            local_best_cfg = best_config_ever
            summary_logger.info(f"⭐ Improved E={best_min_e:.2f} ({gpu_name})")
            log(f"⭐ Improved | E={best_min_e:.2f}")

        # ----------------------------------------------------------
        # Global sync every SYNC_INTERVAL
        # ----------------------------------------------------------
        if round_idx % SYNC_INTERVAL == 0:
            with manager_state["lock"]:
                global_best_e = manager_state["best_e"]

                if local_best_e < global_best_e:
                    manager_state["best_e"]      = float(local_best_e)
                    manager_state["best_params"] = local_params.copy()
                    manager_state["best_cfg"]    = local_best_cfg
                    manager_state["best_gpu"]    = gpu_name
                    manager_state["replace_count"] += 1

                    summary_logger.info(
                        f"🌍 Updated global best → E={local_best_e:.2f} from {gpu_name}"
                    )
                else:
                    log(f"↩️ Synced with global best from {manager_state['best_gpu']}")

        # ----------------------------------------------------------
        # Hybrid early-reseed check (NOT stop)
        # ----------------------------------------------------------
        with manager_state["lock"]:
            global_best_e      = manager_state["best_e"]
            global_best_params = manager_state.get("best_params", None)

        do_reseed, rel_gap = worse_than_global(
            local_best_e, global_best_e, EARLY_STOP_MARGIN, abs_floor=1.0
        )

        if do_reseed and (round_idx - last_reseed_round > 5):  # 5-round cooldown
            log(
                f"🟡 Local energy {local_best_e:.2f} "
                f"is {rel_gap*100:.2f}% worse than global → reseeding."
            )
            if global_best_params:
                for k in params_dic.keys():
                    if k not in global_best_params:
                        continue
                    val = global_best_params[k]
                    dl  = params_dic[k].get("down_limit", 0.95)
                    ul  = params_dic[k].get("up_limit", 1.05)
                    params_dic[k]["val"]        = val
                    params_dic[k]["down_limit"] = dl
                    params_dic[k]["up_limit"]   = ul
                log("🔁 Reseeded grid around global best parameters.")
                last_reseed_round = round_idx
                torch.cuda.empty_cache()
                continue  # restart next BASELINE_ROUNDS with new grid

        torch.cuda.empty_cache()

    # --------------------------------------------------------------
    # Final per-GPU summary
    # --------------------------------------------------------------
    summary_logger.info(f"✅ {gpu_name} done. Best E={local_best_e:.2f}")
    log(f"\n========== Final Results ({device}) ==========")
    log(f"Best Energy: {local_best_e:.6f}")
    for k, v in local_params.items():
        log(f"  {k}: {v}")
    log(f"Replace Count Observed: {manager_state['replace_count']}")
    log("✅ GPU finished search\n")

    return local_params, local_best_e, local_best_cfg, gpu_name


# ============================================================
#  🧩  MAIN EXECUTION
# ============================================================
if __name__ == "__main__":

        # Example usage with your QUBO loader
    Sizes = [5, 7, 11, 28]
    DataSize = {5: 31, 7: 63, 11: 127, 28: 1023}

    InputSize = Sizes[0]   # change index for 7/11/28
    InputDataSize = DataSize[InputSize]
    QUBOFolder = f"QUBO/{InputSize}x{InputSize}/{InputDataSize}x7x10/"
    Q_matrix = np.loadtxt(QUBOFolder + "QUBO_W.txt")

    # Get SA-based Tmax/Tmin from dimod mapping
    bqm = dimod.BinaryQuadraticModel.from_qubo(Q_matrix)
    ising = bqm.spin
    beta_hot, beta_cold = beta_range(ising.linear, ising.quadratic)
    Tmax = 1 / beta_hot
    Tmin = 1 / beta_cold
    print(f"Initial Tmax={Tmax}, Tmin={Tmin}")

    # ------------------------------------------------------------
    # Generate the three requested variables
    # ------------------------------------------------------------
    Q_matrix = (Q_matrix + Q_matrix.T) / 2.0
    h_vec = np.diag(Q_matrix).copy()
    J_matrix = Q_matrix - np.diag(h_vec)

    print("Q_matrix shape:", Q_matrix.shape)
    print("h_vec shape:", h_vec.shape)
    print("J_matrix shape:", J_matrix.shape)
    print("BQM offset:", bqm.offset)

    params_dic = {
        'lr'     : {'val': 0.001, 'range_rule': 'no_limit', 'down_limit': 0.5, 'up_limit': 1.5},
        'wd'     : {'val': 0.001, 'range_rule': 'no_limit', 'down_limit': 0.5, 'up_limit': 1.5},
        'alpha'  : {'val': 0.5,    'range_rule': 'limit',    'down_limit': 0.5, 'up_limit': 1.5, 'limit_val': 0.99999999},
        'mom'    : {'val': 0.5,    'range_rule': 'limit',    'down_limit': 0.5, 'up_limit': 1.5, 'limit_val': 0.99999999},
        'c_grad' : {'val': 12,      'range_rule': 'no_limit', 'down_limit': 0.5, 'up_limit': 1.5},
        'Tmin'   : {'val': Tmin,   'range_rule': 'no_limit', 'down_limit': 0.5, 'up_limit': 1.5},
        'Tmax'   : {'val': Tmax,   'range_rule': 'no_limit', 'down_limit': 0.5, 'up_limit': 1.5},
    }

    devices = [f"cuda:{i}" for i in get_idle_gpus()][:2]
    if not devices:
        # No idle NVIDIA GPU (or no nvidia-smi at all): run a single CPU worker.
        print("[FEM] No idle CUDA device found; falling back to CPU.")
        devices = ["cpu"]
    total_rounds = 4000
    SYNC_INTERVAL = 2000
    EARLY_STOP_MARGIN = 0.30
    BASELINE_ROUNDS = 1
    seed_base = 2025
    search_precision = 10
    N_step = 100
    batch = 100

    optimizer = "rmsprop"
    betamode = "inv"

    device_chunk_config = {dev: (min(64, search_precision), None) for dev in devices}
    for dev in devices:
        print(f"[{dev}] param_chunk={device_chunk_config[dev][0]}, trial_chunk_size=auto")


    summary_logger, log_path = setup_summary_logger(data_size=InputSize, base_folder=f"FEM_Solutions/logs_{InputSize}x{InputSize}")
    summary_logger.info(f"🌐 Multi-GPU FEM Search Started")
    summary_logger.info(f"Active GPUs: {devices}")
    summary_logger.info(f"Sync interval={SYNC_INTERVAL}, Early-stop margin={EARLY_STOP_MARGIN}")

    manager = multiprocessing.Manager()
    shared_state = manager.dict({
        "best_e": float("inf"),
        "best_params": {},
        "best_gpu": None,
        "replace_count": 0,
        "lock": manager.RLock()
    })

    # Parallel execution
    results = []
    gpu_loggers = {dev: GPULogger(dev, data_size=InputSize, base_folder= f"FEM_Solutions/logs_{InputSize}x{InputSize}") for dev in devices}
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        futures = [
            pool.submit(
                _run_search_on_device,
                dev, J_matrix, h_vec, params_dic.copy(), betamode,
                N_step, batch, optimizer,
                search_precision, total_rounds, seed_base,
                shared_state,
                BASELINE_ROUNDS=BASELINE_ROUNDS,
                SYNC_INTERVAL=SYNC_INTERVAL,
                EARLY_STOP_MARGIN=EARLY_STOP_MARGIN,
                summary_logger=summary_logger,
                gpu_logger=gpu_loggers[dev],
                param_chunk=device_chunk_config[dev][0],
                trial_chunk_size=device_chunk_config[dev][1],
            )
            for dev in devices
        ]
        results = [f.result() for f in futures]



    # Final summary aggregation
    best_idx = min(range(len(results)), key=lambda i: results[i][1])
    best_params_dic, best_min_e, best_cfg, best_gpu = results[best_idx]

    summary_logger.info("")
    summary_logger.info("========== Global Best Result ==========")
    summary_logger.info(f"🏆 Global Best Energy: {best_min_e:.6f}")
    summary_logger.info(f"🖥️ From GPU: {best_gpu}")
    summary_logger.info("All Parameters from Best GPU (complete set):")
    for k, v in best_params_dic.items():
        summary_logger.info(f"  {k}: {v}")

    formatted_cfg = np.array2string(np.array(best_cfg, dtype=int), threshold=np.inf, max_line_width=120)
    summary_logger.info("Full Best Configuration (array format):")
    summary_logger.info(formatted_cfg)

    summary_logger.info("✨ Search Completed — All GPUs finished successfully ✨")
    summary_logger.info(f"Summary log saved at: {log_path}")
