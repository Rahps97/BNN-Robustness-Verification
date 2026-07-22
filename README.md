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

The exact software and hardware configurations used for the experiments reported in the paper are provided separately in the reproducibility information accompanying this repository.

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

## Repeated Simulated-Annealing Stability Evaluation

The repeated simulated-annealing experiment can be run using:

```bash
python SA_repeated.py
```

This script performs multiple independent simulated-annealing runs to assess the stability of the obtained solutions across different runs and solver configurations.

## Configuring the Experiments

Parameters controlling the following aspects of the experiments are defined directly inside the corresponding Python scripts:

* Dataset size and preprocessing
* Neural-network architecture
* Input sample and target class
* Perturbation budget
* QUBO construction
* Solver settings
* Number of repeated runs

Users should modify these parameters directly in the relevant source files before running the experiments.
