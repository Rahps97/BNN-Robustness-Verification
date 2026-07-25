"""
verify_counterexamples.py -- an INDEPENDENT validation harness for the results
produced by Z3.py.

This script is deliberately NOT a second solver, and it deliberately shares no
code with Z3.py. It imports nothing from Z3.py: it re-parses Info.txt with its
own parser, re-loads the checkpoint and re-binarizes the weights itself, and
runs its own NumPy forward pass. If Z3.py's encoding, its Info.txt parsing, its
weight binarization or its argmax tie-breaking were wrong, this script would
still be right, and the two would disagree. That is the entire point: a
verifier that checks its own answers with its own code proves nothing.

The only thing taken from Z3.py is the answer under test, read from the result
JSON that Z3.py writes.

It provides two independent checks.

1. `check` -- reverse-verification of a returned counterexample.

   A counterexample is only meaningful if it is a genuine adversarial example.
   For every witness reported by Z3.py this checks, from scratch, that:

     * the witness has the right length and is Boolean;
     * the coordinates it flips are exactly the ones reported;
     * every flipped coordinate is in the permitted perturbable set;
     * the Hamming distance to the clean input is within the epsilon budget;
     * the independent forward pass on the clean input reproduces the label
       recorded in Info.txt;
     * the independent forward pass on the witness returns a DIFFERENT label,
       i.e. the perturbation really does change the prediction;
     * the predicted label and logits agree with the ones Z3.py reported.

   Note that an UNSAT answer is a proof of absence and cannot be checked by
   re-evaluating a witness; use `bruteforce` to corroborate it on the small
   instances.

2. `bruteforce` -- exhaustive validation of the minimum adversarial distance.

   Enumerates every perturbation of 0, 1, 2, ... of the perturbable pixels in
   increasing order and reports the first Hamming distance at which the label
   changes. This is ground truth by construction, independent of Z3 and of any
   encoding. Full enumeration of 2^|P| is only tractable for the smallest
   instance, so enumeration proceeds by increasing distance and stops once a
   level would exceed MAX_COMBINATIONS_PER_LEVEL; everything reported up to
   that point is still exhaustive, and the script says exactly how far the
   proof reaches.

Usage (run from the repository root, after Z3.py has been run):

    python verify_counterexamples.py            # both checks
    python verify_counterexamples.py check
    python verify_counterexamples.py bruteforce

Select the instance with the InputSize constant below, exactly as in Z3.py.
"""

import ast
import itertools
import json
import math
import os
import re
import sys
import time

import numpy as np
import torch


# -----------------------------------------------------------------------------
# USER CONFIGURATION
# -----------------------------------------------------------------------------

Sizes = [5, 7, 11, 28]
DataSize = {5: 31, 7: 63, 11: 127, 28: 1023}

InputSize = Sizes[0]   # 5x5 example; use index 1/2/3 for 7x7 / 11x11 / 28x28
InputDataSize = DataSize[InputSize]

INFO_PATH = f"QUBO/{InputSize}x{InputSize}/{InputDataSize}x7x10/Info.txt"
CHECKPOINT_PATH = f"TrainedNN/{InputSize}x{InputSize}/{InputDataSize}x7x10/{InputDataSize}.pth"

# The result JSON written by Z3.py in single mode.
RESULT_JSON_PATH = (
    f"Z3_Results/{InputSize}x{InputSize}/{InputDataSize}x7x10/Z3_Single_Result.json"
)

# Exhaustive enumeration stops before any distance level that would need more
# than this many combinations. Raise it to push the proof further, at the cost
# of runtime.
MAX_COMBINATIONS_PER_LEVEL = 5_000_000


# -----------------------------------------------------------------------------
# Independent Info.txt parsing
# -----------------------------------------------------------------------------


def parse_info(info_path):
    """Parse Info.txt without using Z3.py's parser.

    Returns a dict with the clean input as a 0/1 list, the perturbable pixel
    indices, the label and the epsilon budget.
    """
    with open(info_path, "r", encoding="utf-8") as handle:
        text = handle.read()

    def scalar(key):
        match = re.search(rf"^{re.escape(key)}\s*:\s*(\S+)\s*$", text, flags=re.MULTILINE)
        if match is None:
            raise ValueError(f"{info_path}: could not find '{key}'.")
        return match.group(1)

    # "Input to perturb : tensor([...])" spans many lines, so slice out the
    # bracketed region by scanning for the matching close bracket.
    anchor = text.index("Input to perturb")
    open_at = text.index("[", anchor)
    depth = 0
    close_at = -1
    for position in range(open_at, len(text)):
        if text[position] == "[":
            depth += 1
        elif text[position] == "]":
            depth -= 1
            if depth == 0:
                close_at = position + 1
                break
    if close_at < 0:
        raise ValueError(f"{info_path}: 'Input to perturb' has no closing bracket.")

    values = ast.literal_eval(text[open_at:close_at])
    clean = []
    for position, value in enumerate(values):
        if value not in (0, 1, 0.0, 1.0):
            raise ValueError(
                f"{info_path}: input entry {position} is {value}, expected 0 or 1."
            )
        clean.append(int(value))

    pixels_match = re.search(r"^Pixels perturbed\s*:\s*(\[.*?\])\s*$", text, flags=re.MULTILINE)
    if pixels_match is None:
        raise ValueError(f"{info_path}: could not find 'Pixels perturbed'.")
    perturbable = [int(v) for v in ast.literal_eval(pixels_match.group(1))]

    return {
        "clean": clean,
        "perturbable": perturbable,
        "label": int(float(scalar("Label to perturb"))),
        "epsilon": int(scalar("Epsilon")),
    }


# -----------------------------------------------------------------------------
# Independent checkpoint loading and forward pass
# -----------------------------------------------------------------------------


def load_weights(checkpoint_path):
    """Load the checkpoint and binarize the weights independently of Z3.py.

    Binarize.forward in QUBOCreator.py maps w >= 0 to +1 and w < 0 to -1, so
    that is exactly what is applied here.
    """
    try:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:  # Older PyTorch versions have no weights_only argument.
        state = torch.load(checkpoint_path, map_location="cpu")

    matrices = []
    for key, value in state.items():
        if key.endswith(".weight") and hasattr(value, "ndim") and value.ndim == 2:
            array = value.detach().cpu().numpy()
            matrices.append(np.where(array >= 0, 1, -1).astype(np.int64))
    if not matrices:
        raise ValueError(f"{checkpoint_path}: no 2-D linear weights found.")
    return matrices


def forward(matrices, boolean_input):
    """Forward pass of the binarized network. Returns (label, logits).

    Boolean 0/1 inputs are mapped to spins -1/+1; every layer but the last is
    binarized with the >= 0 -> +1 convention; the last layer is linear.
    torch.argmax returns the lowest index among tied maxima, and so does
    np.argmax, so the label is computed the same way the network computes it.
    """
    activation = np.where(np.asarray(boolean_input, dtype=np.int64) == 1, 1, -1)
    for position, matrix in enumerate(matrices):
        activation = matrix @ activation
        if position < len(matrices) - 1:
            activation = np.where(activation >= 0, 1, -1).astype(np.int64)
    return int(np.argmax(activation)), activation


# -----------------------------------------------------------------------------
# Check 1: reverse-verify a reported counterexample
# -----------------------------------------------------------------------------


def check_result_json(result_json_path):
    """Independently re-check one result JSON written by Z3.py."""
    print("=" * 78)
    print("REVERSE-VERIFICATION OF THE REPORTED COUNTEREXAMPLE")
    print("=" * 78)

    if not os.path.exists(result_json_path):
        print(f"No result file at '{result_json_path}'.")
        print("Run Z3.py first (RUN_MODE = \"single\") to produce it.")
        return False

    with open(result_json_path, "r", encoding="utf-8") as handle:
        reported = json.load(handle)

    info = parse_info(reported["info_path"])
    matrices = load_weights(reported["checkpoint_path"])

    print(f"result file        : {result_json_path}")
    print(f"info file          : {reported['info_path']}")
    print(f"checkpoint         : {reported['checkpoint_path']}")
    print(f"reported status    : {reported['status']}")
    print(f"epsilon            : {info['epsilon']}")
    print(f"perturbable pixels : {len(info['perturbable'])}")

    checks = []

    def record(name, passed, detail=""):
        checks.append((name, bool(passed)))
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}{(': ' + detail) if detail else ''}")

    print("\nclean input")
    clean_label, clean_logits = forward(matrices, info["clean"])
    record(
        "independent forward pass reproduces the Info.txt label",
        clean_label == info["label"],
        f"independent {clean_label} vs Info.txt {info['label']}",
    )
    record(
        "independent clean logits match the ones Z3.py reported",
        clean_logits.tolist() == reported["clean_logits"],
        f"{clean_logits.tolist()}",
    )
    # The budget the witness must respect is the one that was actually solved
    # for. It equals Info.txt's epsilon on the default path; it differs only if
    # Z3.py was run with SINGLE_EPSILON_OVERRIDE set, which is legitimate but
    # means the run is not the Table V query.
    budget = reported["epsilon"]
    if budget != info["epsilon"]:
        print(
            f"  [NOTE] epsilon override in use: solved at {budget}, "
            f"Info.txt records {info['epsilon']}. This is not the Table V query."
        )
    else:
        print(f"  [NOTE] epsilon {budget} taken from Info.txt (no override)")

    if reported["status"] != "SAT":
        print(
            "\nStatus is not SAT, so there is no witness to reverse-verify.\n"
            "An UNSAT answer asserts that no counterexample exists; that is a\n"
            "proof of absence and cannot be confirmed by re-evaluating a single\n"
            "point. Use 'bruteforce' to corroborate it by exhaustion."
        )
        passed = all(ok for _, ok in checks)
        print("\nRESULT: " + ("PASS" if passed else "FAIL"))
        return passed

    witness = reported["adversarial_input_boolean"]
    print("\nreported witness")

    record(
        "witness has the same length as the clean input",
        witness is not None and len(witness) == len(info["clean"]),
    )
    if witness is None or len(witness) != len(info["clean"]):
        print("\nRESULT: FAIL")
        return False

    record("witness is Boolean", all(value in (0, 1) for value in witness))

    changed = [i for i in range(len(witness)) if witness[i] != info["clean"][i]]
    record(
        "changed coordinates match the reported ones",
        changed == list(reported["changed_indices"]),
        f"{len(changed)} flipped",
    )
    record(
        "Hamming distance matches the reported one",
        len(changed) == reported["hamming_distance"],
        f"independent {len(changed)} vs reported {reported['hamming_distance']}",
    )
    record(
        "Hamming distance is within the epsilon budget",
        len(changed) <= budget,
        f"{len(changed)} <= {budget}",
    )
    outside = sorted(set(changed) - set(info["perturbable"]))
    record(
        "only permitted pixels were flipped",
        not outside,
        "none outside the perturbable set" if not outside else f"outside: {outside}",
    )

    adversarial_label, adversarial_logits = forward(matrices, witness)
    record(
        "independent forward pass changes the label",
        adversarial_label != info["label"],
        f"{info['label']} -> {adversarial_label}",
    )
    record(
        "independent adversarial label matches the reported one",
        adversarial_label == reported["adversarial_prediction"],
        f"independent {adversarial_label} vs reported {reported['adversarial_prediction']}",
    )
    record(
        "independent adversarial logits match the reported ones",
        adversarial_logits.tolist() == reported["adversarial_logits"],
        f"{adversarial_logits.tolist()}",
    )
    record(
        "Z3.py reported this as a counterexample",
        bool(reported["counterexample_found"]),
    )
    record(
        "Z3.py did not claim robustness",
        not reported["certified_robust"],
    )

    passed = all(ok for _, ok in checks)
    print(
        f"\nGenuine adversarial example: {len(changed)} of "
        f"{len(info['perturbable'])} perturbable pixels flipped "
        f"(budget {budget}), prediction {info['label']} -> {adversarial_label}."
    )
    print("\nRESULT: " + ("PASS" if passed else "FAIL"))
    return passed


# -----------------------------------------------------------------------------
# Check 2: exhaustive minimum adversarial distance
# -----------------------------------------------------------------------------


def brute_force_minimum_distance(info_path, checkpoint_path,
                                 max_combinations=MAX_COMBINATIONS_PER_LEVEL):
    """Enumerate perturbations by increasing Hamming distance.

    Returns (minimum_distance, witness_indices, proven_exhaustive_through).
    minimum_distance is None if no adversarial example was found within the
    levels that could be enumerated exhaustively.
    """
    print("=" * 78)
    print("EXHAUSTIVE MINIMUM ADVERSARIAL DISTANCE")
    print("=" * 78)

    info = parse_info(info_path)
    matrices = load_weights(checkpoint_path)
    perturbable = list(info["perturbable"])
    clean = list(info["clean"])

    clean_label, _ = forward(matrices, clean)
    if clean_label != info["label"]:
        raise ValueError(
            f"Independent forward pass predicts {clean_label} but Info.txt records "
            f"{info['label']}; the checkpoint and Info.txt do not correspond."
        )

    print(f"info file          : {info_path}")
    print(f"checkpoint         : {checkpoint_path}")
    print(f"clean label        : {clean_label}")
    print(f"perturbable pixels : {len(perturbable)}")
    print(f"per-level cap      : {max_combinations:,} combinations")
    print("\n distance | combinations | cumulative time (s) | result")
    print("----------+--------------+---------------------+--------")

    started = time.perf_counter()
    proven_through = -1

    for distance in range(0, len(perturbable) + 1):
        level_size = math.comb(len(perturbable), distance)
        if level_size > max_combinations:
            print(
                f" {distance:8d} | {level_size:12,} | "
                f"{time.perf_counter() - started:19.2f} | SKIPPED (over cap)"
            )
            break

        for combination in itertools.combinations(perturbable, distance):
            candidate = clean[:]
            for index in combination:
                candidate[index] = 1 - candidate[index]
            label, _ = forward(matrices, candidate)
            if label != clean_label:
                elapsed = time.perf_counter() - started
                print(
                    f" {distance:8d} | {level_size:12,} | {elapsed:19.2f} | "
                    f"FOUND {clean_label} -> {label}"
                )
                print(
                    f"\nMinimum adversarial distance is exactly {distance}: every "
                    f"perturbation of {distance - 1} or fewer\nperturbable pixels was "
                    f"enumerated and none changes the label."
                )
                print(f"Witness: flip pixels {list(combination)} -> class {label}")
                return distance, list(combination), distance

        proven_through = distance
        print(
            f" {distance:8d} | {level_size:12,} | "
            f"{time.perf_counter() - started:19.2f} | none"
        )

    if proven_through >= 0:
        print(
            f"\nNo adversarial example exists within Hamming distance "
            f"{proven_through};\nlarger distances were not enumerated "
            f"(raise MAX_COMBINATIONS_PER_LEVEL to go further)."
        )
    return None, None, proven_through


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------


def main():
    command = sys.argv[1] if len(sys.argv) > 1 else "all"
    if command not in ("check", "bruteforce", "all"):
        raise SystemExit("Usage: python verify_counterexamples.py [check|bruteforce|all]")

    ok = True
    if command in ("check", "all"):
        ok &= check_result_json(RESULT_JSON_PATH)
        print()
    if command in ("bruteforce", "all"):
        brute_force_minimum_distance(INFO_PATH, CHECKPOINT_PATH)

    if command in ("check", "all") and not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
