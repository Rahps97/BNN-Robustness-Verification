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

The repository ships a dataset for each image size, and every `Info.txt` and every QUBO in the repository was generated from those exact files. `DatasetCreation.py` therefore refuses to overwrite an existing `Dataset/{S}x{S}/Train.txt` or `Test.txt` and exits with a message. This matters because `args.shuffle` is `True`: a regenerated dataset is in a different order, and `QUBOCreator.py` selects the verified input as the first correctly classified training sample and the perturbable pixels by mean absolute value over the training set. Both can change, which would invalidate the shipped QUBO instances and the reported results.

**If you only want to reproduce the shipped results, skip this step entirely** and start at step 3 (or step 4, since the QUBOs are shipped too). To regenerate the dataset deliberately, for a new image size or a fresh experiment:

```bash
OVERWRITE=1 python DatasetCreation.py
```

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

#### Argmax tie-breaking in the misclassification constraint (`argmax_tie_aware`)

The misclassification constraint built by `bnn_as_qubo.setup_optim_model` compares
each competing class against the true class through a two's-complement sign bit
`argmax_sign_{L}_{k}`. That bit is 1 exactly when

```text
logit_k > logit_gt      (strict)
```

`torch.argmax`, which defines the network's actual prediction, breaks ties towards
the **lowest** index. So a competing class `k < gt` that merely *ties* the true
class already misclassifies, while the strict comparison above does not fire. The
condition that matches `torch.argmax` is

```text
logit_k >= logit_gt   for k < gt,     logit_k > logit_gt   for k > gt
```

which is exactly what `Z3.py` encodes (see the `Z3.py` bullet list below). The two
formulations therefore differ on tie cases, and the QUBO one is the weaker of the
two: it can miss a genuine adversarial example.

**This does not affect any result in the paper or in this repository.** Every
reported outcome is a *non-robustness* finding backed by a concrete perturbation,
and a strict inequality that fires is a genuine misclassification. No robustness
certificate is reported anywhere, and a robustness certificate is the only kind of
answer the strict encoding could get wrong. What is affected is a claim about what
an *exact* solve of the QUBO would prove: with the strict encoding, an exact solve
that returns "infeasible" is not a sound robustness certificate.

A concrete instance of the gap, on the shipped 11x11 network (`127x7x10`, true
label 8): flipping the two perturbable pixels `{0, 1}` gives logits
`[1, 3, 1, 1, -1, 1, -3, -1, 3, -3]`. The maximum, 3, is attained at both index 1
and index 8, so `torch.argmax` returns 1 and the network misclassifies — but no
competing logit *strictly* exceeds `logit_8`, so the strict QUBO does not admit
this perturbation.

`bnn_as_qubo.py` can encode the tie-aware condition instead. For `k < gt` it
constrains `sum_class - 1` rather than `sum_class`, where
`sum_class = (logit_gt - logit_k) / 2`; the sign bit is then 1 iff
`sum_class <= 0`, i.e. iff `logit_k >= logit_gt`. **No auxiliary variables are
added and no constraint is restructured** — only the constant term of one residual
per competing class below the true label changes. The existing two's-complement
field already has the range for it, because the last hidden width is `2**n - 1`
(here 7), so the field spans `[-8, 7]` while `sum_class - 1` spans `[-8, 6]`. The
builder raises `NotImplementedError` if a future architecture violates that.

**It is opt-in and off by default**, because turning it on changes the QUBO matrix,
and the QUBO shipped in this repository and every reported number were produced by
the strict encoding. Like the FEM changes described under *Solver changes made
after `paper-results-v1`* below, this is a post-tag change that, at its default,
reproduces the tagged behaviour exactly. With the default the generated files are
bit-identical to the committed ones:

```bash
python QUBOCreator.py
md5sum QUBO/5x5/31x7x10/{QUBO_W.txt,Variables.json,Info.txt,Info_Relevant.txt}
```

To enable it, either export the environment variable

```bash
BNN_ARGMAX_TIE_AWARE=1 python QUBOCreator.py
```

or set `args.argmax_tie_aware = True` (`get_args.py`; `None`, the default, means
"consult the environment variable").

Two things worth knowing before enabling it:

* The **shipped 5x5 instance is unchanged either way.** Its true label is 0, so
  there is no class index below the true class and the tie-aware branch never
  applies. `QUBOCreator.py` produces the same four files with the flag on or off.
  The 7x7, 11x11 and 28x28 instances (labels 3, 8 and 2) do change.
* The energy offset changes on the instances that change, so `Info.txt`'s
  `Minimum Energy` — the target energy a feasible solution must reach — is
  different. Solutions and target energies from the strict instances are not
  comparable with tie-aware ones.

The behaviour of the flag was checked against the network's own `torch.argmax` by
building both encodings, constructing the forced assignment of every named QUBO
variable for a given perturbation, and asking `qubovert` whether all encoded
constraints hold:

| instance | perturbations checked | tie-aware disagreements with `argmax` | strict disagreements |
| --- | --- | --- | --- |
| 5x5 (label 0) | all 65,536 | 0 | 0 (label 0 admits no tie case) |
| 7x7 (label 3) | 9,989 | 0 | 62 |
| 11x11 (label 8) | 48,245 | 0 | 5,653 |

Every strict disagreement is a perturbation where the true class ties the maximum
and `torch.argmax` awards the prediction to a lower index — that is, a real
adversarial example the strict QUBO rejects. In particular, with the flag on the
`{0, 1}` perturbation of the 11x11 instance is admitted as a feasible solution at
Hamming distance 2, and with the flag off it is not.

### 4. Solve the generated QUBO

The generated QUBO can be solved using any of the following solver scripts.

#### Free-Energy Machine Solver

```bash
python FEM.py
```

This script solves the generated QUBO using the Free-Energy Machine solver.

##### The reported FEM results, and how to check them

The FEM energies reported in the paper were produced by `FEM.py` **before** the two
solver changes described in *Solver changes made after `paper-results-v1`* below,
namely the per-worker `copy.deepcopy` of `params_dic` and the relative parameter
floors. Those are forward-looking robustness fixes; they correct nothing in the
paper and change no reported number.

That code state is marked by the annotated tag `paper-results-v1`:

```bash
git checkout paper-results-v1
```

Note that git tags are not carried across a pull-request merge, and squash- or
rebase-merging rewrites commit hashes. If the tag is absent, the pre-change
behaviour can still be recovered from the current tree without it:

- set `FEM_PARAM_FLOOR_RATIO=0`, which disables the parameter floors entirely, and
- replace the `copy.deepcopy(params_dic)` in the worker dispatch with
  `params_dic.copy()`.

Neither the reported energies nor the recorded configurations depend on this: the
solution vectors in `FEM_best_configurations.txt` can be checked against the shipped
QUBO matrices at any commit, and doing so is the recommended way to verify the FEM
column (see below).

FEM's coordinate search is **stochastic**. It draws Sobol-scrambled seeds, sweeps
one hyperparameter at a time, and carries the winning value into the next round, so
two runs from the same starting point explore different trajectories. The recorded
values in `FEM_best_configurations.txt` and `FEM_HYPERPARAMETERS.md` are the state
of that search at the moment each best energy was recorded — a record of what was
run, not a recipe that replays to the same number.

They are therefore provided so the reported energies can be **verified rather than
replayed**. Each entry in `FEM_best_configurations.txt` includes the full solution
vector, so the reported energy can be recomputed directly against the shipped QUBO
matrix, and the perturbation it encodes can be checked against the original network
with `verify_counterexamples.py`. That is a stronger check than a rerun: it confirms
the solution exists and has the claimed energy, independently of which search
trajectory found it.

##### Solver changes made after `paper-results-v1`

Two defects in the hyperparameter search were fixed after the reported runs. Neither
invalidates a published result — the reported solution vectors verify against the
QUBO matrices regardless of how the search reached them — and neither is a
correction to the paper.

* **Worker parameter state was shared.** The multi-device launcher passed each
  worker a shallow copy of the parameter dictionary, so parallel workers mutated the
  same inner objects. The hyperparameters recorded next to a best energy were
  therefore not guaranteed to be the set that produced it. Workers now receive a
  deep copy.
* **Candidate generation could ratchet parameters to zero.** Candidates are
  generated multiplicatively, as `value * [down_limit, up_limit]` with
  `down_limit = 0.5`, and the winner is written back each round. Zero is an
  absorbing state under that rule: a value can be halved indefinitely but cannot
  recover once it reaches a denormal. Runs were observed with `Tmax` drifting from
  445.79 to ~1e-44, and the 11x11 and 28x28 records show `wd`, `mom` and `Tmin` at
  5.605193857299268e-45, the smallest positive float32 denormal. Every parameter now
  gets a relative lower bound of `initial_value * 1e-6`, tunable through the
  `FEM_PARAM_FLOOR_RATIO` environment variable (`0` restores the old behaviour).

  Note that this collapse was **not** fatal to the search: the 28x28 run reached the
  global optimum with collapsed temperatures, and imposing a floor on 5x5 did not
  improve on the reported energy. The floor is a robustness improvement, not an
  explanation of any result.

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

### 5. Exact SMT verification with Z3

```bash
python Z3.py
```

`Z3.py` decides the robustness-verification problem exactly. It reads the instance from the same `Info.txt` the QUBO pipeline produces (the clean input, the perturbable pixel set, the label and the perturbation budget epsilon), reads the trained checkpoint, and encodes the problem directly as an SMT formula rather than as a QUBO:

* inputs are spins in `{-1, +1}`, with a Boolean variable created only for the perturbable pixels;
* the perturbation budget is the cardinality constraint `sum(changed) <= epsilon`;
* hidden layers use the sign convention of `Binarize.forward`, mapping `>= 0` to `+1`;
* misclassification is the exact negation of `torch.argmax(logits) == label`. Because `torch.argmax` returns the lowest index among tied maxima, this is `logits[c] >= logits[label]` for `c < label` and `logits[c] > logits[label]` for `c > label`.

Note that this is *not* the condition the QUBO encodes by default: the QUBO uses the strict comparison for every competing class, so on tie cases the SMT baseline and the QUBO decide subtly different problems. See *Argmax tie-breaking in the misclassification constraint* above, including why no reported result is affected and how to make the QUBO match.

`SAT` means an adversarial example exists within the budget, so the instance is **not robust**. `UNSAT` is a proof that none exists, so the instance is **certified robust** for that perturbation set. `UNKNOWN` means the timeout was reached and nothing is concluded.

Because Z3 is exact, its verdict is ground truth against which the heuristic QUBO solvers (FEM, SA) can be checked.

#### Reproducing Table V

Table V reports the verification result at the epsilon recorded in each instance's `Info.txt`. The defaults in `Z3.py` are already set for this: `RUN_MODE = "single"` and no epsilon override. Run the script once per row, changing only the `Sizes` index in the configuration block near the bottom of the file:

| Table V row | `Z3.py` setting | Architecture | Epsilon | Result |
| --- | --- | --- | --- | --- |
| 5x5 | `InputSize = Sizes[0]` | 31x7x10 | 8 | SAT (NR) |
| 7x7 | `InputSize = Sizes[1]` | 63x7x10 | 32 | SAT (NR) |
| 11x11 | `InputSize = Sizes[2]` | 127x7x10 | 32 | SAT (NR) |
| 28x28 | `InputSize = Sizes[3]` | 1023x7x10 | 128 | SAT (NR) |

All four are SAT, i.e. **NR (not robust)**: within the stated budget an adversarial example exists in every case. Each run writes `Z3_Results/{S}x{S}/{D}x7x10/Z3_Single_Result.json` with the full witness.

Two things to note about the reported timings:

* **The reported runtime is solve time only.** `runtime_seconds` is measured around `solver.check()` alone; the clock starts after the SMT formula has been constructed, so formula construction, `Info.txt` parsing and checkpoint loading are all excluded. Wall-clock time for the whole script is larger, and the gap grows with instance size.
* Timings vary between machines and between runs. The SAT/UNSAT verdicts do not.

#### Minimum adversarial distance (separate experiment)

Setting `RUN_MODE = "scan"` instead solves epsilon = 0, 1, 2, ... and stops at the first `SAT`. Because the feasible perturbation set grows monotonically with epsilon, that first SAT radius is the exact minimum adversarial distance, and the largest UNSAT radius below it is a certified robust radius. This is a different and much more expensive experiment than the Table V query; it is not what Table V reports.

### 6. Independently validate the results

```bash
python verify_counterexamples.py
```

`Z3.py` is a verifier, so its output should not be taken on trust. `verify_counterexamples.py` is a validation harness, not a second solver. It shares no code with `Z3.py` and does not import it: it re-parses `Info.txt` with its own parser, re-loads the checkpoint and re-binarizes the weights itself, and runs its own NumPy forward pass. If `Z3.py`'s encoding, parsing, binarization or argmax tie-breaking were wrong, the two would disagree.

It runs two checks, selectable as `python verify_counterexamples.py check` and `python verify_counterexamples.py bruteforce`:

* **`check`** reverse-verifies the witness in the result JSON written by `Z3.py`: that it flips exactly the reported coordinates, that every flipped coordinate is in the permitted perturbable set, that the Hamming distance is within the budget solved for, that the clean input reproduces the label recorded in `Info.txt`, and above all that the perturbed input really is classified differently. An `UNSAT` answer is flagged as a proof of absence that no witness can confirm.
* **`bruteforce`** enumerates perturbations by increasing Hamming distance and reports the first distance at which the label changes. This is the exact minimum adversarial distance by construction, independent of Z3 and of any encoding. Distance levels larger than `MAX_COMBINATIONS_PER_LEVEL` are skipped and the script states how far the exhaustive proof reaches.

Set `InputSize` the same way as in `Z3.py`. On the four shipped instances every Table V witness passes reverse-verification, and brute force confirms minimum adversarial distances of 3, 1, 2 and 2 for 5x5, 7x7, 11x11 and 28x28 respectively.

## Repeated Simulated-Annealing Stability Evaluation

The repeated simulated-annealing experiment can be run using:

```bash
python SA_repeated.py
```

This script performs multiple independent simulated-annealing runs to assess the stability of the obtained solutions across different runs and solver configurations.

## Configuring the Experiments

All parameters are plain module-level constants; there is no configuration file. Most scripts declare them near the top, `Z3.py` in a `USER CONFIGURATION` block near the bottom. The only command-line arguments anywhere are the optional `check` / `bruteforce` subcommands of `verify_counterexamples.py`. The table below lists where each configurable quantity actually lives.

| What you want to change | File | Variable |
| --- | --- | --- |
| Image size (5x5, 7x7, 11x11, 28x28) | `DatasetCreation.py` | `InputSize = 5` |
| | `TrainingNN.py` | `InputSize = Sizes[0]` |
| | `QUBOCreator.py` | `InputSize = Sizes[0]` |
| | `SA.py` | `InputSize = Sizes[0]` |
| | `SA_repeated.py` | `INPUT_SIZE = SIZES[0]` |
| | `Gurobi.py` | `InputSize = Sizes[0]` |
| | `FEM.py` | `InputSize = Sizes[0]` (inside `__main__`) |
| | `Z3.py` | `InputSize = Sizes[0]` |
| | `verify_counterexamples.py` | `InputSize = Sizes[0]` |
| Overwriting a shipped dataset | `DatasetCreation.py` | `OVERWRITE=1` environment variable |
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
| Argmax tie-breaking in the QUBO | `get_args.py` | `args.argmax_tie_aware`, or the `BNN_ARGMAX_TIE_AWARE` environment variable (default off) |
| SA reads, sweeps, schedule | `SA.py` | `NUM_READS`, `NUM_SWEEPS`, `beta_schedule_type` argument |
| Repeated-SA settings | `SA_repeated.py` | `NUM_REPETITIONS`, `SEEDS`, `NUM_READS`, `NUM_SWEEPS`, `BETA_SCHEDULE_TYPE` |
| FEM search budget | `FEM.py` | `total_rounds`, `search_precision`, `N_step`, `batch` (inside `__main__`) |
| FEM optimizer / annealing mode | `FEM.py` | `optimizer`, `betamode` |
| FEM hyperparameter lower bound | `FEM.py` | `FEM_PARAM_FLOOR_RATIO` environment variable (default `1e-6`, `0` disables), or a per-parameter `min_val` in `params_dic` |
| Z3 query type (Table V vs. distance scan) | `Z3.py` | `RUN_MODE` (`"single"` / `"scan"`) |
| Z3 solver timeout | `Z3.py` | `TIMEOUT_SECONDS` |
| Z3 epsilon, overriding `Info.txt` | `Z3.py` | `SINGLE_EPSILON_OVERRIDE` |
| Z3 distance-scan range | `Z3.py` | `SCAN_START_EPSILON`, `SCAN_MAX_EPSILON` |
| Brute-force enumeration budget | `verify_counterexamples.py` | `MAX_COMBINATIONS_PER_LEVEL` |

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
# Decide the same instance exactly with Z3 (the Table V query, epsilon = 8)
python Z3.py
```

```text
=== Exact Z3 BNN Verification ===
Architecture           : 31 x 7 x 10
Target / clean pred.   : 0 / 0
Perturbable coordinates: 16
Epsilon                : 8
Status                 : SAT
Conclusion             : COUNTEREXAMPLE FOUND
Changed indices        : [0, 1, 2, 3, 4, 14, 19, 23]
Hamming distance       : 8
Adversarial prediction : 2
Forward label changed  : True
```

`SAT` is the 5x5 row of Table V: within a budget of 8 pixel flips the network is **not robust**, and Z3 returns a concrete witness taking class 0 to class 2.

```bash
# Independently validate that answer
python verify_counterexamples.py
```

```text
  [PASS] independent forward pass reproduces the Info.txt label: independent 0 vs Info.txt 0
  [NOTE] epsilon 8 taken from Info.txt (no override)
  [PASS] changed coordinates match the reported ones: 8 flipped
  [PASS] Hamming distance is within the epsilon budget: 8 <= 8
  [PASS] only permitted pixels were flipped: none outside the perturbable set
  [PASS] independent forward pass changes the label: 0 -> 2
  [PASS] independent adversarial logits match the reported ones: [3, -1, 5, 1, -5, -1, -3, 1, 1, 1]

RESULT: PASS

 distance | combinations | cumulative time (s) | result
----------+--------------+---------------------+--------
        0 |            1 |                0.00 | none
        1 |           16 |                0.00 | none
        2 |          120 |                0.00 | none
        3 |          560 |                0.00 | FOUND 0 -> 6

Minimum adversarial distance is exactly 3: every perturbation of 2 or fewer
perturbable pixels was enumerated and none changes the label.
Witness: flip pixels [0, 20, 4] -> class 6
```

The first block re-derives everything with its own code and confirms the witness is a genuine adversarial example. The second goes further: for this instance the network is robust up to a perturbation budget of 2 and not robust from 3 onwards, since flipping the three pixels 0, 20 and 4 changes the prediction from class 0 to class 6. Exhaustive enumeration proves 3 is the true minimum.

Timings, energies and the particular witness reported by the heuristic solvers vary between runs and machines; the robust/non-robust boundary reported by `Z3.py` does not.
