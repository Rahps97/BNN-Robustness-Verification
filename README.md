# Robustness Verification of Binary Neural Networks: An Ising and Quantum-Inspired Framework

This repository verifies the robustness of binary neural networks (BNNs) by mapping the verification problem to a Quadratic Unconstrained Binary Optimization (QUBO) instance and solving it with conventional optimizers and with Ising-style or annealing-based hardware. It is the code and data behind the paper cited at the end of this file.

## Verifying the reported results

One command re-checks the paper's numbers and prints a pass/fail report:

```bash
git clone https://github.com/Rahps97/BNN-Robustness-Verification.git && cd BNN-Robustness-Verification
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
python verify_paper.py
```

That is the whole procedure. It runs **105 checks** over **all four instances** (5x5, 7x7, 11x11, 28x28), takes **about 20 to 25 seconds** once the archives are unpacked, and needs **no GPU, no solver license, no network access and no annealing hardware**. Every check prints its own pass/fail line, and the script exits non-zero if any reported number fails to reproduce — or if too little ran for the report to mean anything. See *Verdict and exit status* below.

**Tables III to VII are covered, with one exception noted below.** That includes every row of Table VII: the two-class instance, the Gurobi and both hardware solution vectors, and the recorded Gurobi solver logs, all ship alongside the QUBOs, so a reviewer can re-evaluate them without any hardware or license.

**The one exception is three of Table VI's five data columns.** Table VI reports repeated simulated annealing over 30 seeds, and its *% All Constraints Satisfying Solution*, *Average Unsatisfied Constraints* and *Median Time Taken* columns come from `SA_repeated.py`, a separate and much longer experiment that `verify_paper.py` does not invoke. What the default run does check for Table VI is the structure it shares with Tables III and IV: the variable count, the true constraint count and the energy offset. To reproduce the other three columns, run `python SA_repeated.py` directly.

That script takes no command-line arguments: the run is configured by the constants at the top of the file. As shipped it selects the smallest instance (`INPUT_SIZE = SIZES[0]`, the 5x5) and runs `NUM_REPETITIONS = 30` seeds at `NUM_READS = 1024` and `NUM_SWEEPS = 1000`, which on the 5x5 costs about 6 s per repetition — roughly three minutes in total, against the 20 to 25 seconds of `verify_paper.py`. Edit `INPUT_SIZE` to reach the larger instances, which are correspondingly slower. Results are written to `SA_Repeated_Results/`, which is git-ignored, and nothing is written until all repetitions have finished, so an interrupted run leaves no output.

**The Gurobi cell of Table VII used to be a second exception, and is not one any more.** No solution vector had been archived for that solver on the two-class instance, so the row was reported as `UNAVAILABLE`. The first author has since supplied the vector, and it now ships in `data/hardware_results.tar.gz` as `hardware/Gurobi/`, in the same format as the Fujitsu solution. `verify_paper.py` re-evaluates it exactly as it does the Fujitsu row: it reaches the target energy of **-874,674** with **all 65 encoded constraints satisfied**. His own `Verify_Gurobi.ipynb` ships beside it as the provenance. That run used a plain `m.optimize()` with no early-stopping callback, so Gurobi solved this QUBO to proven optimality, unlike the four Table IV runs the callback stopped short of a proof — the instance is far smaller. Its **61.447 s is not asserted**, because wall time is machine dependent: the author's notebook records 16.658 s for the same instance and a third machine takes 25 to 30 s, so a reviewer who re-runs it and sees 17 to 30 s is seeing the expected spread, the same way the SA row's time is handled.

**Table VII's SA row is checked by re-running the solver, not by re-evaluating a stored vector.** No SA vector was archived either, but on this 113-variable instance simulated annealing is cheap, so `verify_paper.py` simply runs it as part of the default run: `dwave-samplers`' `SimulatedAnnealingSampler` at `num_reads = 1000`, up to three seeds, stopping at the first that reaches the target. It reaches the target energy of **-874,674** with **all 65 encoded constraints satisfied**, in about **2.3 s** per seed. Being a stochastic solver, reaching the target is `PASS` and falling short would be `INCONCLUSIVE`, never `FAIL`. For reference, the authors' own stored notebook output records 3.90 s and Table VII reports 3.559 s — the same result on three different machines.

Everything needed ships compressed in `data/` (5.8 MB in total), and `verify_paper.py` unpacks what it needs on the first run and says so. That first run is therefore slower than the figure above, because it also writes out about 137 MB of dense text; how much slower depends on the disk and on what else the machine is doing, and measurements from 35 s to just under two minutes have been seen. Every run afterwards reuses the extracted files. The script reports its own elapsed time on the last line. The 5x5 QUBO and checkpoint are tracked in the repository directly, so `python verify_paper.py --quick` runs on a bare clone without unpacking the QUBO archive, which is 0.54 MB compressed and expands to 137 MB; it still unpacks the two small archives (0.2 MB together) that the Gurobi-log and Table VII checks read.

### What it checks

| Paper table | Checked how |
| --- | --- |
| Tables III, IV, VI (structure only) | Each QUBO is rebuilt from scratch with the authors' own `bnn_as_qubo.setup_optim_model`, and the variable count, the **true** constraint count and the energy offset `E_off` are compared with the tables. The rebuilt matrix is then compared entry by entry with the shipped `QUBO_W.txt`. For Table VI this covers only the columns shared with Tables III and IV; its three repeated-annealing columns are not checked here and require `SA_repeated.py`. |
| Table IV, FEM column | The FEM solution vectors in `FEM_best_configurations.txt` are evaluated against the shipped QUBO matrices as `x^T Q x`, and each is decoded into a perturbation and replayed through an independent NumPy forward pass of the BNN. |
| Table V | The Z3 SMT baseline is re-run at each instance's recorded epsilon, and every witness is reverse-checked with that same independent forward pass. |
| Table V, `d_min` | Recomputed twice per instance: by exhaustive enumeration and by a Z3 minimum-distance scan. |
| Table IV, Gurobi column | The incumbent is read out of each recorded solver log and compared with the table, along with the fact that no run proved optimality and that each was ended by the early-stopping callback rather than by a time limit — the no-improvement span in the log is checked against the limit `run_gurobi_all.py` ships for that instance. The reported energy score is then reached by a second, independent route, as the FEM column is: `dE` is taken against the `Minimum Energy` recorded in that instance's own `Info.txt`, so the score check fails if the shipped instance and the reported number disagree even when the log itself reads exactly as published. See *Gurobi Solver* below. |
| Table VII | The two-class QUBO is rebuilt from the authors' pickled constraint dictionaries with the recipe in their own `Verify.ipynb`, and both hardware samples are re-evaluated against it: the Fujitsu solution attains the target energy exactly and satisfies all 65 encoded constraints in 0.366 s, and the best of D-Wave's 4,000 archived shots reaches −874,318, 356 above the target, satisfying 36 of 65. Those 4,000 are the archived batch of a 5,000-shot run, which was made in two batches of 4,000 and 1,000 shots; only the first was archived, and free D-Wave access has since ended, so the other 1,000 cannot be recovered or re-run. The reported quantum-annealer entry is therefore reproduced here from the archived batch, and the remaining 1,000 shots are not independently verifiable. The SA row has no archived vector, so simulated annealing is re-run on the same instance and reaches −874,674 with 65 of 65 in about 2.3 s. The Gurobi vector, supplied by the first author after submission, is re-evaluated the same way as the Fujitsu one and also reaches −874,674 with 65 of 65; its 61.447 s is machine dependent and is not asserted. Every row of the table is therefore checked here. |

Note that `E_off` is an **energy**, not a constraint count. Earlier versions of Tables III, IV and VI reported it in the *Total Constraints* column; `verify_paper.py` recomputes and reports the two separately so the correction can be checked directly. The same distinction applies to Table VII, whose *Constraints Satisfied* column prints 1,273 (`len(H.to_qubo())`, i.e. the instance's 1,272 QUBO terms plus the constant-offset entry) and 356 (an energy gap) rather than constraint counts; the instance has 65 encoded constraints in total, and the script prints the corrected figures next to the reported ones.

### Sample output

```text
===============================================================================
 verify_paper.py -- reproducing the reported numerical results
===============================================================================
 instances       : 5x5, 7x7, 11x11, 28x28
 QUBO encoding   : strict argmax (args.argmax_tie_aware = False), the encoding
                   used for every shipped QUBO and every reported number
 opt-in checks   : none (no license, no GPU, no network, no hardware needed)

 [setup] 7x7, 11x11, 28x28 not extracted; unpacking data/qubo_and_networks.tar.gz
 [setup] into the repository root (QUBO/ and TrainedNN/; ~138 MB expanded, git-ignored)
 [setup] done.
 [setup] the recorded Gurobi solver logs not extracted; unpacking data/gurobi_logs.tar.gz
 [setup] the Table VII two-class hardware instance not extracted; unpacking data/hardware_results.tar.gz

-------------------------------------------------------------------------------
 Tables III, IV, VI -- QUBO structure (variables / constraints / energy offset)
-------------------------------------------------------------------------------
 5x5 (31x7x10)   rebuilt in 2.2 s
   [     PASS     ] Total Variables                                paper          276   recomputed          276
   [     PASS     ] Total Constraints (true count)                 paper          200   recomputed          200   (eq 198 + gt 1 + lt 1)
   [     PASS     ] Energy offset E_off (an energy, not a count)   paper          533   recomputed          533
   [     PASS     ] rebuilt QUBO == shipped QUBO_W.txt             276x276, max |difference| = 0
   ...

-------------------------------------------------------------------------------
 Table V -- exact SMT baseline (Z3) at the epsilon from Info.txt
-------------------------------------------------------------------------------
 28x28 (1023x7x10)
   [     PASS     ] Z3 verdict at epsilon 128                      paper           NR   recomputed           NR   (SAT, solve 1.360 s)
   [     PASS     ] witness reverse-checked on the original BNN    110 pixels flipped <= budget 128, all perturbable, prediction 2 -> 6

-------------------------------------------------------------------------------
 Table VII -- the two-class instance: hardware rows and SA
-------------------------------------------------------------------------------
 Digital Annealer (Fujitsu)
   [     PASS     ] Best energy                                    paper     -874,674   recomputed     -874,674   (the recorded run reports -874,674)
   [     PASS     ] every encoded constraint satisfied             qubovert is_solution_valid True, penalty value 0, so the target energy is attained exactly
   [     PASS     ] Constraints satisfied (corrected Table VII)    paper           65   recomputed           65   of 65
   [     PASS     ] Total time (s)                                 paper        0.366   recomputed        0.366

 Simulated Annealing (re-run here; no sample was archived)
   running up to 3 seed(s) at 1,000 reads, stopping at the first that reaches the target ...
   [     PASS     ] SA best energy (Table VII)                     paper     -874,674   recomputed     -874,674
                    seeds  : 1 run, 1/1 reached the paper's value; energies min -874,674 / median -874,674 / max -874,674
                    gaps   : vs paper 0 -- reproduced exactly; vs target 0 (0.000%) -> at most 0 encoded constraint(s) violated
                    time   : 2.3 s
   [     PASS     ] Constraints satisfied (corrected Table VII)    paper           65   recomputed           65   of 65

 Gurobi (vector supplied by the first author after submission)
   [     PASS     ] Best energy                                    paper     -874,674   recomputed     -874,674   the target energy, so this run found a global minimum
   [     PASS     ] every encoded constraint satisfied             qubovert is_solution_valid True, penalty value 0, so the target energy is attained exactly
   [     PASS     ] Constraints satisfied (corrected Table VII)    paper           65   recomputed           65   of 65
                    note   : Table VII reports 61.447 s for this row; wall time is machine dependent and is not asserted

===============================================================================
 SUMMARY
===============================================================================
 Tables III/IV/VI  QUBO structure                   33 passed, 3 not run
 Table IV          FEM energies + reverse check     12 passed
 Table V           Z3 SMT baseline                  8 passed
 Table V           minimum adversarial distance     8 passed
 Table IV          Gurobi column (logs)             24 passed
 Table IV          Gurobi column                    4 not run
 Table IV          SA column                        4 not run
 Table IV          FEM solver replay                4 not run
 Table VII         hardware results                 20 passed

 105 passed, 0 failed, 15 not run (--with-fem, --with-gurobi, --with-sa)
 elapsed: 25.8 s

 15 check(s) were NOT run and are therefore NOT verified; see the reasons
 above. Do not read them as confirmed.

 RESULT: PASS -- every check that was run reproduces the paper. (exit 0)
===============================================================================
```

The caveats print **above** the verdict deliberately, so that the last line of the report is never a bare `PASS` sitting on top of the reasons it should be read with.

No row is `unavailable` any more. Table VII's Gurobi cell was the last one, and the vector the first author supplied closed it — see above. The three structural rows shown as *not run* are the optional cross-check that the training set re-selects the same instance; that one needs `data/datasets.tar.gz` unpacked as well, and with it the run is 108 passed, 0 failed, still well under a minute.

The 15 remaining *not run* rows are the opt-in solver reruns listed under *Options* below. **A `not run` row is not a verified row**, which is why the script says so twice — once per row with the flag that would run it, and once in the closing `RESULT` block — and why the totals are reported separately rather than folded together.

### Outcomes

A check is never silently omitted. Every row carries one of:

| Outcome | Meaning |
| --- | --- |
| `PASS` | recomputed, and it matches the paper |
| `FAIL` | recomputed, and it does **not** match the paper |
| `SKIPPED` | not requested; the row states the flag that would run it |
| `UNAVAILABLE` | cannot be run here: no license, no data, no hardware |
| `TIMEOUT` | started but exceeded `--timeout` |
| `INCONCLUSIVE` | ran, but the solver is stochastic and fell short; neither confirms nor refutes the reported number |
| `NOT VERIFIABLE` | no offline substitute exists at all; defined by the harness but not currently emitted — Table VII's one uncheckable cell reports `UNAVAILABLE` instead |

"Never silently omitted" is enforced rather than asserted. Each check group declares up front how many rows it must produce for the selected instances, flags and available resources, and the script compares that with what it actually recorded. A check that cannot run has to leave a `SKIPPED`, `UNAVAILABLE`, `TIMEOUT` or `INCONCLUSIVE` row behind; if any check disappears instead, the run ends in `RESULT: HARNESS ERROR` rather than reporting a smaller pass count as though nothing had happened.

### Verdict and exit status

| Verdict | Exit | When |
| --- | --- | --- |
| `RESULT: PASS` | 0 | at least one check ran, and none failed |
| `RESULT: FAIL` | 1 | a reported number did not reproduce |
| *(no report)* | 2 | bad command line; `argparse` prints usage. No verdict uses this status, because `argparse` already owns it |
| `RESULT: INCONCLUSIVE` | 3 | nothing failed, but too little ran to conclude anything: either no check ran at all, or a whole default check group produced neither a `PASS` nor a `FAIL` because its data, archive or dependency is absent |
| `RESULT: HARNESS ERROR` | 4 | the run did not record the number of rows its own configuration calls for. This is a bug in `verify_paper.py`, not a result about the paper |

**Exit 3 is the one to watch for in CI.** A clone whose `data/` archives are missing, or a pipeline step whose extraction quietly failed, has nothing to verify — so it must not report a green `PASS`. It reports `INCONCLUSIVE` and exits 3. Treat any non-zero status as a failed job; treat 1 and 3 as different problems.

### Options

| Flag | Effect | Cost and requirements |
| --- | --- | --- |
| *(none)*, or `--all` | all four instances, everything that needs nothing external | 105 checks in about 20-25 s once unpacked, CPU only |
| `--quick` | 5x5 only, plus the Table VII rows, which are instance-independent | 42 checks in ~4 s; runs on a bare clone, unpacking only the two small archives (0.2 MB) rather than the 0.54 MB QUBO archive that expands to 137 MB |
| `--instance 5,7` | a chosen subset | — |
| `--with-gurobi` | re-solve the Table IV Gurobi column from scratch, instead of reading the recorded logs | needs a Gurobi license; hours. The common size-limited license caps at 2,000 variables, so 28x28 (2,235) is reported `UNAVAILABLE`, not `FAIL` |
| `--with-sa` | Table IV SA column, at `SA.py`'s settings | minutes for 5x5, hours for 28x28; `--sa-seeds N` sets the seed count (default 3) |
| `--with-fem` | replays FEM at its recorded hyperparameters | seconds to minutes; stochastic, so a shortfall is `INCONCLUSIVE` |
| `--everything` | all three of the above | — |
| `--timeout S` | per-check wall-clock bound | exceeding it is `TIMEOUT`, never `FAIL`. `--timeout 0` means **no limit at all**, including for the opt-in checks, which otherwise get 900 s each. A negative value is rejected |
| `--json PATH` | machine-readable report as well | `-` writes to stdout, after the human-readable report rather than instead of it, so pipe to a file and parse that rather than straight into `jq`. `result` carries the verdict word and `exit_code` the status from the table above; `expected_check_counts` and `recorded_check_counts` carry the per-group self-check |
| `--verbose` | full output of every sub-check | — |
| `--no-extract` | never unpack `data/` automatically | — |

For the heuristic and early-terminated solvers, two different gaps are reported, because they answer different questions: the gap against **the paper's value** is the reproduction question, and the gap against **the target energy** is the feasibility question. They are not the same — Table IV's SA column is 8,019 for 11x11 against a target of 8,020. Since the objective `H_0` is identically zero, a gap of `g` above the target upper-bounds the number of violated encoded constraints.

That same fact gives a lower bound, and it is checked rather than assumed: with `H_0` identically zero the QUBO is a sum of squared constraint residuals, so the target energy `-E_off` is a *proven* lower bound and nothing can score beneath it. A solver that comes back below the target has not found a better solution; the instance is wrong — a corrupted `QUBO_W.txt`, a mis-parsed `Info.txt`, or the wrong matrix. That is reported as `FAIL`, not as a very good `PASS`.

### The shipped data

| Archive | Size | Extracts to | Needed for |
| --- | --- | --- | --- |
| `data/qubo_and_networks.tar.gz` | 0.54 MB | `QUBO/`, `TrainedNN/` | Tables III to VI: the four QUBO instances and their checkpoints |
| `data/gurobi_logs.tar.gz` | 0.09 MB | `gurobi_logs/` | Table IV's Gurobi column, from the recorded solver logs |
| `data/hardware_results.tar.gz` | 0.11 MB | `hardware/` | Table VII: the two-class QUBO, the Gurobi, Fujitsu and D-Wave solution vectors, and the authors' `Verify.ipynb`, `Verify_Gurobi.ipynb` and `SA_Verify.ipynb` |
| `data/datasets.tar.gz` | 5.07 MB | `Dataset/` | only regenerating QUBOs from scratch with `QUBOCreator.py` |

`verify_paper.py` unpacks the first three itself. To do it by hand:

```bash
tar xzf data/qubo_and_networks.tar.gz
tar xzf data/gurobi_logs.tar.gz
tar xzf data/hardware_results.tar.gz
tar xzf data/datasets.tar.gz            # only needed to rerun QUBOCreator.py
```

They extract into the repository root at the paths every script already expects (`QUBO/7x7/...`, not `data/QUBO/7x7/...`). `QUBO/` and `Dataset/` expand to about 138 MB and 313 MB of dense text, which is why they ship compressed and why the expanded paths are git-ignored. The 5x5 files are tracked in the repository and are byte-identical to their copies in the archives, so extracting over them changes nothing and `git status` stays clean.

The three notebooks inside `hardware_results.tar.gz` are the authors' own artifacts, kept for provenance — they show what was actually run on the hardware, how the SA row was produced, and where the Gurobi row's solution vector came from. They are not the verification path: `verify_paper.py` does all of it, needs no Jupyter, and does not read them. Three edits were made across them, all recorded here. In `SA_Verify.ipynb` the SA import was `from neal import SimulatedAnnealingSampler`; `neal` was folded into `dwave-samplers`, which is what `requirements.txt` pins, so the line now reads `from dwave.samplers import SimulatedAnnealingSampler`. In `Verify_Gurobi.ipynb` the QUBO is opened by a bare filename, which only works from inside `hardware/QUBO/`, so the path now reads `QUBO/113-1273-28-15-zero-3-3-561020-H.pickle` like the other two notebooks; and its stored Gurobi banner had the license ID replaced with `<redacted>`, exactly as in the shipped solver logs. Every stored cell output is otherwise the authors' original, including the `state` vector that `hardware/Gurobi/` was built from and the `16.658` s runtime. Note that `Verify.ipynb` covers the Fujitsu sample only, despite extracting next to `hardware/Dwave/`: it reads the Fujitsu solution and timing from `hardware/Result/` and `hardware/Time/` and never opens the D-Wave DataFrame. The D-Wave sample is checked by `verify_paper.py` instead.

---

**A reviewer checking the paper can stop here.** Everything above is the verification path: one command, no configuration, nothing to edit. Everything below is the development path — how to regenerate the dataset, retrain the network, rebuild the QUBOs and rerun each solver from scratch. None of it is needed to check a reported number.

## What the pipeline does

Starting from a trained ten-class BNN and a correctly classified input, the pipeline constructs a QUBO representation of the verification problem using penalty-based formulations and searches for adversarial perturbations within a prescribed perturbation budget.

The framework is designed to support both **conventional optimization algorithms** and **unconventional Ising-style or annealing-based hardware**.

The goal of this codebase is to demonstrate that BNN robustness-verification problems can be expressed as QUBO instances and solved using both classical and emerging Ising or annealing platforms, thereby providing a bridge between AI trustworthiness and unconventional computing.

## Installation Details

`requirements.txt` pins the exact set of versions the 5x5 pipeline was run against end to end on a CPU-only machine with Python 3.9. It is what the three-line procedure at the top of this file installs, and it is enough for `verify_paper.py`:

```bash
python -m pip install -r requirements.txt
```

Note that `requirements.txt` pins CPU builds of PyTorch and TorchVision. On a machine with a GPU, install the PyTorch build matching your CUDA or ROCm version instead. `pandas` is in the list even though no script imports it by name: the archived D-Wave sample used for Table VII is a pickled DataFrame, so unpickling it needs pandas installed. `gurobipy` is deliberately not in the list, because installing it does not provide the license it also needs.

To install by hand instead, the following Python packages cover the experiments:

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

`dwave-ocean-sdk` is the umbrella package; `requirements.txt` pins the two components the code actually imports, `dimod` and `dwave-samplers`, instead.

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

#### Where binarization happens, and the sign convention

Two details of the forward pass are worth stating explicitly, because the QUBO
encoding has to match them exactly and neither is visible from the network
summary. Both `TrainingNN.py` and `QUBOCreator.py` define them identically.

**`sgn(0) = +1`.** `Binarize.forward` maps `inp >= 0` to `+1` and `inp < 0` to
`-1`. It is written out rather than delegated to `Tensor.sign()` because
`sign()` returns `0` at `0`, which would leave a neuron in neither state. The
encoded constraints use the same convention.

For the four networks reported here the tie is unreachable at the activations,
and the reason matters before assuming a reimplementation is equivalent.
`TrainingNN.py` and `QUBOCreator.py` both map each input through `to_spin`
before the forward pass, so `fc1` sees values in `{-1, +1}` rather than the
`{0, 1}` stored in the dataset files, and the binarized weights are in
`{-1, +1}` too. Every fan-in in these architectures is odd, 31, 63, 127 and
1023 into the hidden layer and 7 into the output layer, so each pre-activation
is a sum of an odd number of terms of `±1` and is therefore odd and never `0`. Flipping input
bits under perturbation does not change that. Checked directly on the 5x5
training set: `0` of `9,660` hidden and `0` of `13,800` output pre-activations
are zero.

Two consequences. The convention is not load-bearing for the reported results,
so a reimplementation that breaks the tie the other way still reproduces them.
It does become load-bearing for any even fan-in, and it already applies to the
weights themselves, where a stored weight of exactly `0.0` binarizes to `+1`.

**`BinaryLinear` binarizes twice, `LastLayer` once.** `BinaryLinear.forward`
binarizes the weight matrix and then binarizes the layer output as well, so the
hidden activations are in `{-1, +1}`. `LastLayer.forward` binarizes only the
weight matrix and returns real-valued logits, which training feeds to a softmax
and `QUBOCreator.py` feeds to `argmax`. Reimplementing the forward pass with a
single binarization step, or with the output layer binarized too, produces a
different network from the one the reported QUBOs encode.

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

`test_tie_aware_qubo.py` keeps that encoding honest, and needs nothing beyond
`requirements.txt`:

```bash
python test_tie_aware_qubo.py
```

It builds the 11x11 instance both ways and asserts the invariant the correction
turns on: the energy offset moves by exactly the true label index, because
exactly the classes `k < gt` get the `-1` and each shifts the constant term by
one. For 11x11 the label is 8, so the offset goes 8,020 → 8,028. It also asserts
that nothing else moved — same variable count, same variable order, same
constraint counts — and checks the 5x5 control, whose label is 0 and whose two
encodings must therefore produce a bit-identical matrix. The third test pins the
default: the strict encoding is what you get unless you ask otherwise.

Note that the offset delta is the **label index**, not the number of `gt`-kind
entries in `H.constraints`; that count is 1 on every instance, being the single
`add_constraint_gt_zero` over the sign bits.

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

##### The stopping rule, and why it differs per instance

No time limit, node limit or MIP gap is set. The runs are ended by a stagnation
callback: it records the node index of the most recent *improving* incumbent, and
calls `model.terminate()` once the search has gone `max_no_improvement_nodes`
nodes past it. The quantity that matters is therefore the **no-improvement span**,
not the total node count — a run can explore far more nodes than the limit, as
long as it keeps finding better incumbents.

That limit is **not the same for every instance**, and the reported runs used two
different values:

| Instance | No-improvement limit |
| --- | --- |
| 5x5 | 100,000 (10^5) |
| 7x7 | 100,000 (10^5) |
| 11x11 | 10,000,000 (10^7) |
| 28x28 | 10,000,000 (10^7) |

This is recoverable from the shipped logs in `data/gurobi_logs.tar.gz`. Each one
ends in `Solve interrupted` with no `TimeLimit` set, i.e. the callback fired, so
subtracting the node index on the last `H` incumbent line from the `Explored ...
nodes` total gives the span that triggered it. Each span lands just *above* a
round threshold, the excess being the in-flight nodes that drain between
`model.terminate()` and the solver actually stopping:

| Instance | Explored | Last improving incumbent | Span | Threshold + overshoot |
| --- | --- | --- | --- | --- |
| 5x5 | 166,017 | 64,212 | 101,805 | 10^5 + 1,805 |
| 7x7 | 131,629 | 31,607 | 100,022 | 10^5 + 22 |
| 11x11 | 17,297,995 | 7,297,465 | 10,000,530 | 10^7 + 530 |
| 28x28 | 10,169,478 | 168,317 | 10,001,161 | 10^7 + 1,161 |

Note the 11x11 row in particular: 17.3 million nodes explored is well above 10^7,
so a reading based on the node total alone would wrongly conclude that a 10^7 rule
could not have fired.

A single hardcoded limit therefore cannot reproduce all four runs. `Gurobi.py`
selects it from `NoImprovementNodes[InputSize]` and `run_gurobi_all.py` from
`DEFAULT_NO_IMPR_NODES[size]`, both defaulting to the table above.
`verify_paper.py` re-derives each span from the shipped log and checks it against
`run_gurobi_all.py`'s table, so the two cannot drift apart unnoticed.

**One line of each log was redacted.** Every log opened with

```
Set parameter LicenseID to value <the ID>
```

which is an identifier tied to a named academic Gurobi account. It is not a
credential and it serves no reproducibility purpose, but it would have become
permanent once a DOI is minted, so in all four logs that line now reads
`Set parameter LicenseID to value <redacted>`. Nothing else was touched: the
redaction changed one line and three bytes per file, and every other byte — the
model statistics, the node counts, the incumbent and bound lines, the timings and
the MIP gaps — is unchanged, as are the archive's member order and per-member tar
metadata. The academic-licence banner on the next line was left alone; it names no
account. `verify_paper.py` was re-run against a fresh extraction of the repacked
archive and reports the same result as before the redaction, including all 24
log-derived Gurobi checks. So if you are reading a log and wondering whether it was
edited: yes, on exactly one line, and only to remove the licence ID.

To force one value across every instance instead — for a quick smoke test, say —
set `NO_IMPR_NODES`:

```bash
NO_IMPR_NODES=100000 python run_gurobi_all.py       # all four at 10^5
```

Reproducing the reported runs still takes 47 minutes to 18.7 hours per instance on
32 cores, and none of them proves optimality: all four are reported incumbents
with MIP gaps of 31.8% to 212.6% remaining.

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

To run all of this at once for every instance, without editing `InputSize` anywhere, use `python verify_paper.py` — see *Verifying the reported results* at the top of this file.

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
| Gurobi no-improvement limit | `Gurobi.py` | `NoImprovementNodes[InputSize]` (10^5 for 5x5/7x7, 10^7 for 11x11/28x28) |
| | `run_gurobi_all.py` | `DEFAULT_NO_IMPR_NODES[size]`, or `NO_IMPR_NODES` environment variable to force one value everywhere |
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
Info file              : QUBO/5x5/31x7x10/Info.txt
Checkpoint             : TrainedNN/5x5/31x7x10/31.pth
Architecture           : 31 x 7 x 10
Specification          : torch_argmax
Target / clean pred.   : 0 / 0
Perturbable coordinates: 16
Epsilon                : 8
Status                 : SAT
Runtime (s)            : 0.047204
Conclusion             : COUNTEREXAMPLE FOUND
Changed indices        : [0, 1, 2, 3, 4, 14, 19, 23]
Hamming distance       : 8
Adversarial prediction : 2
Forward label changed  : True
Adversarial logits     : [3, -1, 5, 1, -5, -1, -3, 1, 1, 1]
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

## Citing this work

If you use this code, please cite the paper and the archived software release.

**Paper.** *Robustness Verification of Binary Neural Networks: An Ising and Quantum-Inspired Framework*, Rahul Singh, Seyran Saeedi and Zheng Zhang. Preprint: [arXiv:2602.13536](https://arxiv.org/abs/2602.13536).

**Software.** `CITATION.cff` in the repository root carries the machine-readable citation metadata; GitHub renders it as the *Cite this repository* panel in the sidebar, with ready-made APA and BibTeX exports.

The archival DOI has not been minted yet. `.zenodo.json` and `CITATION.cff` hold the metadata Zenodo reads when a GitHub release is archived; cutting a release on this repository mints the DOI automatically. The DOI then belongs in the paper's Data Availability statement and in its reference list.

## Licence and reuse

| What | Licence | File |
| --- | --- | --- |
| Code | MIT | [`LICENSE`](LICENSE) |
| Data | CC BY 4.0 | [`LICENSE-DATA`](LICENSE-DATA) |
| Two Apache-2.0 derived functions | Apache-2.0 | [`NOTICE`](NOTICE) |

**These terms are provisional until a release is tagged.** MIT and CC BY 4.0 are the intended licences and are what the repository is offered under today, but they are still pending written confirmation by all three authors and by their institution: the authors are UCSB-affiliated and the work is funded by NSF 2311295 and DOE DE-SC0021323, so the copyright holder line in `LICENSE` is being confirmed as well. Nothing above is settled enough to rely on for a permanent record — treat the grant as final only once this repository carries a tagged release, because the licence files inside an archived snapshot cannot be changed afterwards.

"Data" means `Dataset/`, `QUBO/`, `TrainedNN/`, `FEM_best_configurations.txt`, `FEM_HYPERPARAMETERS.md` and the four archives in `data/` — the binarized MNIST subsets, the QUBO instances, the trained checkpoints, the recorded Gurobi logs and the Gurobi, Fujitsu and D-Wave samples. Everything executable is MIT, including the three notebooks that ship inside `data/hardware_results.tar.gz`.

Two licences rather than one because MIT is written for software and reads badly over a directory of `.txt` matrices, and because the conventional split for a code-plus-data release is a permissive software licence next to a Creative Commons data licence. Creative Commons itself recommends against putting CC licences on code, and Springer Nature's own licence chooser offers exactly this pairing. Neither licence is more restrictive than the other in practice: both permit commercial use, modification and redistribution, and both ask only for attribution.

**Attribution is satisfied by citing the paper and the archived release** — see [Citing this work](#citing-this-work) and `CITATION.cff`. You do not need to do anything else.

### MNIST

Everything under `Dataset/` is derived from MNIST by class selection, adaptive average-pool downsampling and binarization; the checkpoints and QUBO matrices are derived in turn from those subsets. The original MNIST image files are not redistributed here.

MNIST was never released under an explicit licence. Its original distribution point carried no licence, no copyright notice and no terms of use, and since January 2025 it no longer serves the files at all. The licences asserted for MNIST by third-party mirrors — CC BY-SA 3.0, MIT, CC0-1.0, "unknown" — conflict with each other and none traces back to the dataset's authors. The CC BY 4.0 grant here is over this repository's derived artefacts only; it makes no claim about MNIST itself, and nothing in it should be read as asserting that MNIST had terms these files inherit. `LICENSE-DATA` says this at length, and points at NIST Special Database 19 and at QMNIST for anyone who needs a clean chain of title to the underlying images.

### Dependencies

Every runtime dependency is permissively licensed and none constrains this repository: Apache-2.0 (`qubovert`, `dimod`, `dwave-samplers` and the rest of Ocean), BSD-3-Clause (`torch`, `torchvision`, `numpy`, `scipy`, `pandas`), MIT (`z3-solver`), MPL-2.0 (`tqdm`, used unmodified, and MPL-2.0 is file-level copyleft that does not reach a larger work) and matplotlib's PSF-style licence. `gurobipy` is a proprietary client, is not redistributed here, and is not needed to reproduce any reported number.

### The Free Energy Machine implementation

`FEM.py` is an **independent reimplementation** of the Free Energy Machine of Shen et al., *Nature Computational Science* **5**, 322–332 (2025), [doi:10.1038/s43588-025-00782-0](https://doi.org/10.1038/s43588-025-00782-0). It was written from the method as published, is batched over hyperparameter candidates in a way the reference implementation is not, and shares no source with the authors' released code at [`Fanerst/FEM`](https://github.com/Fanerst/FEM) — a line-level comparison against that repository finds no run of three or more matching non-comment lines. This is worth stating because `Fanerst/FEM` carries no licence file at all and is therefore all-rights-reserved by default; the Zenodo snapshot of it ([10.5281/zenodo.14874189](https://doi.org/10.5281/zenodo.14874189)) is CC BY 4.0. Neither applies here, since nothing was copied, but please cite Shen et al. for the method.

The one exception is `beta_range()` in `FEM.py`, which *is* third-party — see [`NOTICE`](NOTICE).
