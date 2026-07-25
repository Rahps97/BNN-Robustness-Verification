# Robustness Verification of Binary Neural Networks: An Ising and Quantum-Inspired Framework

This repository implements an end-to-end pipeline for the **robustness verification of binary neural networks (BNNs)** by mapping the verification problem to a **Quadratic Unconstrained Binary Optimization (QUBO)** instance. Starting from a trained ten-class BNN and a correctly classified input, the pipeline constructs a QUBO representation of the verification problem using penalty-based formulations and searches for adversarial perturbations within a prescribed perturbation budget.

The framework is designed to support both **conventional optimization algorithms** and **unconventional Ising-style or annealing-based hardware**.

The goal of this codebase is to demonstrate that BNN robustness-verification problems can be expressed as QUBO instances and solved using both classical and emerging Ising or annealing platforms, thereby providing a bridge between AI trustworthiness and unconventional computing.

## Installation Details

The following Python packages are required to reproduce the experiments:

```text
torch
torchvision
numpy
scipy
pandas
matplotlib
scikit-learn
tqdm
pyyaml
qubovert
dwave-ocean-sdk
z3-solver
psutil
nvidia-ml-py
```

The following package is optional:

```text
pytest
```

Users should install package versions and hardware-specific builds that are compatible with their systems. In particular, the appropriate PyTorch build depends on the operating system, processor architecture, GPU vendor, CUDA or ROCm version, and installed device driver.

A new Conda environment can be created using:

```bash
conda create -n bnn-qubo python=3.11 pip
conda activate bnn-qubo
```

The hardware-independent packages can then be installed using:

```bash
python -m pip install \
    numpy \
    scipy \
    pandas \
    matplotlib \
    scikit-learn \
    tqdm \
    pyyaml \
    qubovert \
    dwave-ocean-sdk \
    z3-solver \
    psutil \
    nvidia-ml-py
```

PyTorch and TorchVision should be installed separately using the installation method appropriate for the target hardware and software platform:

```bash
python -m pip install torch torchvision
```

Alternatively, `requirements.txt` pins a set of versions that the 5x5 pipeline has been verified to run against end to end on a CPU-only machine with Python 3.9:

```bash
python -m pip install -r requirements.txt
```

Note that `requirements.txt` pins CPU builds of PyTorch and TorchVision. On a machine with a GPU, install the PyTorch build matching your CUDA or ROCm version instead.

### Gurobi Requirement

A valid Gurobi license is required to run the experiments or scripts that use `gurobipy`. Installing the `gurobipy` package alone is not sufficient. Users are responsible for obtaining and activating an appropriate Gurobi license before running these experiments.

The Gurobi Python interface can be installed using:

```bash
python -m pip install gurobipy
```

## Running the Experiments

Run the scripts in the following order.

### 1. Create the binarized dataset

```bash
python DatasetCreation.py
```

This script creates the binarized dataset used for neural-network training and robustness verification.

### 2. Train the neural network

```bash
python TrainingNN.py
```

This script trains the binary neural network using the generated binarized dataset.

### 3. Generate the robustness-verification QUBO

```bash
python QUBOCreator.py
```

This script converts the robustness-verification problem for the trained neural network into a QUBO instance.

### 4. Solve the generated QUBO

The generated QUBO can be solved using any of the following solver scripts.

#### Free-Energy Machine Solver

```bash
python FEM.py
```

This script solves the generated QUBO using the Free-Energy Machine solver.

#### Gurobi Solver

```bash
python Gurobi.py
```

This script solves the generated QUBO using Gurobi. A valid Gurobi license is required. Installing `gurobipy` alone is not sufficient; users must obtain and activate an appropriate Gurobi license before running this solver.

#### Simulated-Annealing Solver

```bash
python SA.py
```

This script solves the generated QUBO using simulated annealing.

The FEM and simulated-annealing solvers are heuristic solvers. Their outputs should therefore be validated against the QUBO constraints and the original BNN before being interpreted as valid adversarial counterexamples or robustness-verification results.

### 5. Exact SMT baseline (optional)

```bash
python z3_baseline.py validate
python z3_baseline.py sweep
```

This script encodes the same robustness-verification problem directly as an SMT formula and decides it exactly with Z3, without going through the QUBO. It selects the perturbable pixels and the verified input exactly as `QUBOCreator.py` does, so it addresses the same instance and can be used as ground truth against which the heuristic QUBO solvers are checked.

`validate` cross-checks the encoding against an independent NumPy forward pass and against exhaustive enumeration of the whole perturbation space. `sweep` decides robustness at every perturbation budget from 0 to 16 and reports timings. `python z3_baseline.py scaling` runs a synthetic study of how Z3 runtime grows with problem size; it is slow and does not need the trained network.

## Repeated Simulated-Annealing Stability Evaluation

The repeated simulated-annealing experiment can be run using:

```bash
python SA_repeated.py
```

This script performs multiple independent simulated-annealing runs to assess the stability of the obtained solutions across different runs and solver configurations.

## Configuring the Experiments

All parameters are plain module-level constants near the top of each script; there is no configuration file and no command-line interface. The table below lists where each configurable quantity actually lives.

| What you want to change | File | Variable |
| --- | --- | --- |
| Image size (5x5, 7x7, 11x11, 28x28) | `DatasetCreation.py` | `InputSize = 5` |
| | `TrainingNN.py` | `InputSize = Sizes[0]` |
| | `QUBOCreator.py` | `InputSize = Sizes[0]` |
| | `SA.py` | `InputSize = Sizes[0]` |
| | `SA_repeated.py` | `INPUT_SIZE = SIZES[0]` |
| | `Gurobi.py` | `InputSize = Sizes[0]` |
| | `FEM.py` | `InputSize = Sizes[0]` (inside `__main__`) |
| Training batch size | `DatasetCreation.py` | `batch_size_train` |
| Downscaling / padding / contradiction removal | `DatasetCreation.py` | `args.downscale`, `args.pad_flattened_dataset`, `args.remove_contradicting`, `args.use_adaptive` |
| Classes included in the dataset | `DatasetCreation.py`, `QUBOCreator.py` | `args.selected_targets` |
| Training epochs, learning rate, seed | `TrainingNN.py` | `n_epochs`, `learning_rate`, `random_seed` |
| Network architecture (layer widths) | `TrainingNN.py` | `Net.__init__` (`self.fc1`, `self.fc4`) |
| | `QUBOCreator.py` | `QUBONet.__init__` (must match `TrainingNN.py`) |
| Number of perturbable pixels | `QUBOCreator.py` | `PetrubSize` |
| Perturbation budget (epsilon) | `QUBOCreator.py` | `PetrubSizeBound`, used as `args.epsilon` |
| Penalty weights in the QUBO | `QUBOCreator.py` | `args.LAMBDA` |
| QUBO objective / bound constraint | `QUBOCreator.py` | `args.objective`, `args.include_perturbation_bound_constraint` |
| SA reads, sweeps, schedule | `SA.py` | `NUM_READS`, `NUM_SWEEPS`, `beta_schedule_type` argument |
| Repeated-SA settings | `SA_repeated.py` | `NUM_REPETITIONS`, `SEEDS`, `NUM_READS`, `NUM_SWEEPS`, `BETA_SCHEDULE_TYPE` |
| FEM search budget | `FEM.py` | `total_rounds`, `search_precision`, `N_step`, `batch` (inside `__main__`) |
| FEM optimizer / annealing mode | `FEM.py` | `optimizer`, `betamode` |
| Z3 perturbation budget sweep | `z3_baseline.py` | `eps_max` argument of `cmd_sweep` |

The image size is the one parameter that must be changed consistently across scripts: `DatasetCreation.py` uses a literal `InputSize = 5`, while the other scripts index into `Sizes = [5, 7, 11, 28]`, so `Sizes[0]` selects 5x5, `Sizes[1]` selects 7x7, and so on. The derived data dimension (`31`, `63`, `127`, `1023`) and all folder paths follow automatically.

Note that the perturbation budget is baked into the QUBO matrix when `QUBOCreator.py` runs. Changing `PetrubSizeBound` therefore requires regenerating the QUBO; the solver scripts read whatever budget is already encoded in `QUBO_W.txt`.

## Worked Example: 5x5 End to End

The repository ships the 5x5 artifacts, so the QUBO can be regenerated and solved without retraining. From the repository root:

```bash
# Generate the QUBO for the trained 5x5 network
python QUBOCreator.py
```

This writes `QUBO/5x5/31x7x10/QUBO_W.txt` (the 276 x 276 QUBO matrix), `Variables.json` (the variable ordering) and `Info.txt`. `Info.txt` records the instance:

```text
Minimum Energy : -533
Total Variables : 276
Epsilon : 8
```

```bash
# Solve it with simulated annealing
python SA.py
```

```text
=== Results on same QUBO ===
Ocean SA -> best_energy = -530.000000, time = 6.19s
[1 0 0 1 0 0 0 1 1 1 1 0 1 0 0 0 ...]
```

The printed vector is the best assignment found, ordered as in `Variables.json`. Its energy should be compared against the `Minimum Energy` recorded in `Info.txt`; as noted above, SA and FEM are heuristics, so a returned assignment is only a robustness result once it has been validated against the QUBO constraints and the original BNN.

```bash
# Cross-check the answer exactly with the SMT baseline
python z3_baseline.py validate
```

```text
perturbable pixels : [0, 20, 4, 24, 5, 15, 10, 1, 3, 9, 19, 23, 21, 14, 2, 22]
clean label        : 0

(b) eps=2  -> unsat  [expected unsat]
(c) eps=3  -> sat    flips=[4, 20, 22]   [expected sat]
    numpy re-eval: 0 -> 6, hamming=3

(d) brute force over 2^16: min adversarial hamming = 3 via [0, 20, 4]

VALIDATION: PASS
```

For this instance the network is robust up to a perturbation budget of 2 and not robust from 3 onwards: flipping three pixels changes the prediction from class 0 to class 6. Exhaustive enumeration of all 2^16 perturbations confirms 3 is the true minimum.

Timings, energies and the particular witness reported by the heuristic solvers vary between runs and machines; the robust/non-robust boundary reported by `z3_baseline.py` does not.
