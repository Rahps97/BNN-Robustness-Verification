import time
import numpy as np

# --- D-Wave / Ocean pieces
import dimod  # shared BQM API
from dwave.samplers import SimulatedAnnealingSampler  # Ocean SA (CPU)

# --- OpenJij SQA
# import openjij as oj

def run_dwave_sa(Q_dict, num_reads=1000, num_sweeps=1000, beta_schedule_type="linear"):
    """Run Ocean's SimulatedAnnealingSampler on QUBO dict."""
    bqm = dimod.BinaryQuadraticModel.from_qubo(Q_dict)
    sampler = SimulatedAnnealingSampler()
    t0 = time.time()
    sampleset = sampler.sample(
        bqm,
        num_reads=num_reads,
        num_sweeps=num_sweeps,
        beta_schedule_type=beta_schedule_type,  # ('linear','geometric','custom')
    )
    t1 = time.time()
    best = sampleset.first
    return {
        "best_energy": float(best.energy),
        "best_sample": best.sample,
        "time_sec": t1 - t0,
        "sampleset": sampleset,
    }

# Load the inputs
Sizes = [5, 7, 11, 28]
InputDataSize = {5: 31, 7: 63, 11: 127, 28: 1023}
PetrubSize = {5: 16, 7: 32, 11: 64, 28: 256}
PetrubSizeBound = {5: 3, 7: 32, 11: 32, 28: 128}

# Variables
InputSize = Sizes[0]
InputDataSize = InputDataSize[InputSize]

QUBOFolder = f"QUBO/{InputSize}x{InputSize}/{InputDataSize}x7x10/"
Q_matrix = np.loadtxt(QUBOFolder + "QUBO_W.txt")


res_qubo = Q_matrix
res = np.array(res_qubo)
Q = res
n = Q.shape[0]

# --- Parameters (feel free to tweak for fairness) ---
NUM_READS = 1024
NUM_SWEEPS = 1000

# --- D-Wave Ocean SA ---
out_sa = run_dwave_sa(Q, num_reads=NUM_READS, num_sweeps=NUM_SWEEPS, beta_schedule_type="geometric")

# --- OpenJij SQA ---
# out_sqa = run_openjij_sqa(Q, num_reads=NUM_READS, num_sweeps=NUM_SWEEPS, beta=5.0, gamma=1.0, trotter=32)

print("\n=== Results on same QUBO ===")
print(f"Ocean SA -> best_energy = {out_sa['best_energy']:.6f}, time = {out_sa['time_sec']:.3f}s")

# If you want the best assignments as NumPy bit-vectors:
x_sa  = np.array([out_sa["best_sample"][i] for i in range(n)], dtype=int)

formatted_cfg = np.array2string(np.array(x_sa, dtype=int), threshold=np.inf, max_line_width=120)
print(formatted_cfg)