from __future__ import annotations

import ast
import csv
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from z3 import Bool, BoolVal, If, Or, Solver, Sum, is_true, sat, unknown, unsat


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class VerificationInfo:
    info_path: str
    minimum_energy: float | None
    total_variables: int | None
    min_qubo_element: float | None
    max_qubo_element: float | None
    total_pixels_to_perturb: int
    pixels_to_perturb: list[int]
    input_boolean: list[int]
    target_label: int
    epsilon: int
    perturbation_bound_included: bool


@dataclass(frozen=True)
class BinaryLinearLayer:
    name: str
    weight: np.ndarray  # shape: [out_features, in_features], entries in {-1, +1}
    bias: np.ndarray  # shape: [out_features], integer entries


@dataclass
class VerificationResult:
    info_path: str
    checkpoint_path: str
    specification: str
    status: str
    reason_unknown: str | None
    runtime_seconds: float
    timeout_seconds: float
    epsilon: int
    bound_included: bool
    input_dimension: int
    perturbable_count: int
    perturbable_indices: list[int]
    target_label: int
    clean_prediction: int
    clean_logits: list[int]
    certified_robust: bool
    counterexample_found: bool
    witness_forward_changes_label: bool | None
    adversarial_prediction: int | None
    adversarial_logits: list[int] | None
    adversarial_input_boolean: list[int] | None
    changed_indices: list[int] | None
    hamming_distance: int | None
    architecture: list[int]


@dataclass
class RadiusScanSummary:
    info_path: str
    checkpoint_path: str
    specification: str
    scan_status: str
    start_epsilon: int
    requested_max_epsilon: int
    effective_max_epsilon: int
    tested_radii: list[int]
    total_runtime_seconds: float
    first_sat_epsilon: int | None
    minimum_adversarial_distance: int | None
    certified_robust_radius: int | None
    first_unknown_epsilon: int | None
    exact_minimum_proven: bool
    target_label: int
    clean_prediction: int
    perturbable_count: int
    architecture: list[int]


# -----------------------------------------------------------------------------
# Info.txt parsing
# -----------------------------------------------------------------------------


def _extract_scalar(text: str, key: str, cast: type, default: Any = None) -> Any:
    match = re.search(rf"^{re.escape(key)}\s*:\s*(.*?)\s*$", text, flags=re.MULTILINE)
    if match is None:
        return default
    value = match.group(1).strip()
    return cast(value)


def _extract_tensor_list(text: str, key: str) -> list[float]:
    """Extract a possibly multiline ``tensor([...])`` value from Info.txt."""
    marker = f"{key} :"
    start = text.find(marker)
    if start < 0:
        # Permit arbitrary whitespace before/after the colon.
        match = re.search(rf"{re.escape(key)}\s*:\s*tensor\s*\(", text)
        if match is None:
            raise ValueError(f"Could not find '{key}' in Info file.")
        tensor_start = match.end()
    else:
        tensor_match = re.search(r"tensor\s*\(", text[start:])
        if tensor_match is None:
            raise ValueError(f"Found '{key}', but its value is not written as tensor(...).")
        tensor_start = start + tensor_match.end()

    list_start = text.find("[", tensor_start)
    if list_start < 0:
        raise ValueError(f"Could not find the opening '[' for '{key}'.")

    depth = 0
    list_end = None
    for index in range(list_start, len(text)):
        char = text[index]
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                list_end = index + 1
                break

    if list_end is None:
        raise ValueError(f"Could not find the closing ']' for '{key}'.")

    value = ast.literal_eval(text[list_start:list_end])
    if not isinstance(value, list):
        raise ValueError(f"'{key}' did not contain a one-dimensional list.")
    return [float(item) for item in value]


def read_info_file(info_path: str | Path) -> VerificationInfo:
    """Read the Info.txt format produced by the supplied QUBO creator."""
    info_path = Path(info_path)
    text = info_path.read_text(encoding="utf-8")

    pixels_text_match = re.search(
        r"^Pixels perturbed\s*:\s*(\[.*?\])\s*$", text, flags=re.MULTILINE
    )
    if pixels_text_match is None:
        raise ValueError("Could not parse 'Pixels perturbed' from Info file.")
    pixels = [int(v) for v in ast.literal_eval(pixels_text_match.group(1))]

    input_values = _extract_tensor_list(text, "Input to perturb")
    input_boolean: list[int] = []
    for index, value in enumerate(input_values):
        rounded = int(round(value))
        if rounded not in (0, 1) or not np.isclose(value, rounded):
            raise ValueError(
                f"Input entry {index} is {value}; expected Boolean values written as 0 or 1."
            )
        input_boolean.append(rounded)

    label_value = _extract_scalar(text, "Label to perturb", float)
    if label_value is None or not float(label_value).is_integer():
        raise ValueError(f"Label must be integer-valued, received {label_value!r}.")

    bound_text = _extract_scalar(
        text, "Perturbation Bound Constriant included", str, default=None
    )
    if bound_text is None:
        # Also accept a corrected spelling in future Info files.
        bound_text = _extract_scalar(
            text, "Perturbation Bound Constraint included", str, default=None
        )
    if bound_text is None:
        raise ValueError("Could not parse whether the perturbation bound is included.")
    bound_included = bound_text.strip().lower() in {"true", "1", "yes"}

    total_pixels = _extract_scalar(text, "Total Pixels to perturb", int)
    epsilon = _extract_scalar(text, "Epsilon", int)
    if total_pixels is None or epsilon is None:
        raise ValueError("Info file must contain total perturbable pixels and epsilon.")

    if total_pixels != len(pixels):
        raise ValueError(
            f"Info file says {total_pixels} perturbable pixels but lists {len(pixels)}."
        )
    if len(set(pixels)) != len(pixels):
        raise ValueError("'Pixels perturbed' contains duplicate indices.")
    if any(index < 0 or index >= len(input_boolean) for index in pixels):
        raise ValueError("A perturbable pixel index is outside the input tensor.")
    if epsilon < 0:
        raise ValueError("Epsilon must be nonnegative.")

    return VerificationInfo(
        info_path=str(info_path),
        minimum_energy=_extract_scalar(text, "Minimum Energy", float),
        total_variables=_extract_scalar(text, "Total Variables", int),
        min_qubo_element=_extract_scalar(text, "Minimum elemenet of the QUBO", float),
        max_qubo_element=_extract_scalar(text, "Maximum elemenet of the QUBO", float),
        total_pixels_to_perturb=total_pixels,
        pixels_to_perturb=pixels,
        input_boolean=input_boolean,
        target_label=int(label_value),
        epsilon=epsilon,
        perturbation_bound_included=bound_included,
    )


# -----------------------------------------------------------------------------
# Checkpoint loading and exact BNN forward evaluation
# -----------------------------------------------------------------------------


def _unwrap_state_dict(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(checkpoint, Mapping):
        for candidate_key in ("state_dict", "model_state_dict"):
            candidate = checkpoint.get(candidate_key)
            if isinstance(candidate, Mapping):
                checkpoint = candidate
                break

    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint is not a PyTorch state_dict or a wrapped state_dict.")

    normalized: dict[str, torch.Tensor] = {}
    for key, value in checkpoint.items():
        if not isinstance(value, torch.Tensor):
            continue
        normalized_key = str(key)
        while normalized_key.startswith("module."):
            normalized_key = normalized_key[len("module.") :]
        normalized[normalized_key] = value.detach().cpu()

    if not normalized:
        raise ValueError("No tensors were found in the checkpoint.")
    return normalized


def load_binary_linear_layers(checkpoint_path: str | Path) -> list[BinaryLinearLayer]:
    """Infer the ordered fully connected architecture from a PyTorch state_dict.

    The state_dict insertion order is retained. This matches the supplied model,
    whose relevant keys are expected to be ``fc1.weight`` and ``fc4.weight``.
    Convolutional weights are intentionally rejected because the supplied creator
    defines a fully connected network.
    """
    checkpoint_path = Path(checkpoint_path)
    try:
        raw = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:  # Compatibility with older PyTorch versions.
        raw = torch.load(checkpoint_path, map_location="cpu")

    state = _unwrap_state_dict(raw)
    weight_keys = [
        key
        for key, value in state.items()
        if key.endswith(".weight") and isinstance(value, torch.Tensor) and value.ndim == 2
    ]
    if not weight_keys:
        raise ValueError("No two-dimensional linear-layer weights were found.")

    layers: list[BinaryLinearLayer] = []
    previous_out_features: int | None = None

    for weight_key in weight_keys:
        layer_name = weight_key[: -len(".weight")]
        raw_weight = state[weight_key].numpy()
        if raw_weight.ndim != 2:
            raise ValueError(f"Layer {layer_name!r} is not a linear layer.")

        # This exactly matches Binarize.forward in QUBOCreater.py.
        binary_weight = np.where(raw_weight >= 0, 1, -1).astype(np.int64)
        out_features, in_features = binary_weight.shape

        if previous_out_features is not None and in_features != previous_out_features:
            raise ValueError(
                f"Layer {layer_name!r} expects {in_features} inputs, but the previous "
                f"layer produces {previous_out_features}. Check checkpoint key order."
            )

        bias_key = f"{layer_name}.bias"
        if bias_key in state:
            raw_bias = state[bias_key].numpy().astype(float)
            rounded_bias = np.rint(raw_bias)
            if not np.allclose(raw_bias, rounded_bias):
                raise ValueError(
                    f"Layer {layer_name!r} has non-integer bias values. The supplied "
                    "network uses bias=False; extend the encoding before using such a model."
                )
            bias = rounded_bias.astype(np.int64)
        else:
            bias = np.zeros(out_features, dtype=np.int64)

        layers.append(BinaryLinearLayer(layer_name, binary_weight, bias))
        previous_out_features = out_features

    return layers


def architecture_from_layers(layers: Sequence[BinaryLinearLayer]) -> list[int]:
    if not layers:
        return []
    return [int(layers[0].weight.shape[1])] + [
        int(layer.weight.shape[0]) for layer in layers
    ]


def forward_binary_network(
    input_boolean: Sequence[int], layers: Sequence[BinaryLinearLayer]
) -> tuple[int, np.ndarray]:
    """Independent NumPy forward pass matching the supplied PyTorch BNN."""
    activation = np.where(np.asarray(input_boolean, dtype=np.int64) == 1, 1, -1)

    if activation.ndim != 1:
        raise ValueError("Input must be one-dimensional.")
    if activation.size != layers[0].weight.shape[1]:
        raise ValueError(
            f"Input has length {activation.size}, but the checkpoint expects "
            f"{layers[0].weight.shape[1]}."
        )

    logits: np.ndarray | None = None
    for layer_index, layer in enumerate(layers):
        preactivation = layer.weight @ activation + layer.bias
        if layer_index < len(layers) - 1:
            activation = np.where(preactivation >= 0, 1, -1).astype(np.int64)
        else:
            logits = preactivation.astype(np.int64)

    assert logits is not None
    prediction = int(np.argmax(logits))  # NumPy and torch.argmax choose the first maximum.
    return prediction, logits


# -----------------------------------------------------------------------------
# Z3 encoding
# -----------------------------------------------------------------------------


def _sum_z3(terms: Sequence[Any]) -> Any:
    if not terms:
        return 0
    return Sum(list(terms))


def _encode_network(
    input_spin_expressions: Sequence[Any], layers: Sequence[BinaryLinearLayer]
) -> list[Any]:
    previous = list(input_spin_expressions)

    for layer_index, layer in enumerate(layers):
        layer_outputs: list[Any] = []
        for output_index in range(layer.weight.shape[0]):
            weighted_terms = [
                int(layer.weight[output_index, input_index]) * previous[input_index]
                for input_index in range(layer.weight.shape[1])
            ]
            preactivation = _sum_z3(weighted_terms) + int(layer.bias[output_index])

            if layer_index < len(layers) - 1:
                # Exact sign convention from Binarize.forward: zero maps to +1.
                layer_outputs.append(If(preactivation >= 0, 1, -1))
            else:
                layer_outputs.append(preactivation)
        previous = layer_outputs

    return previous


# The one sound misclassification specification: it is exactly the negation of
# "torch.argmax(logits) == target" for the network the paper trains and
# evaluates. Two weaker variants used to be selectable here and both were
# unsound, so they were removed; see the note above _misclassification_constraint.
SPECIFICATION = "torch_argmax"


def _misclassification_constraint(logits: Sequence[Any], target: int) -> Any:
    """Encode "the network no longer predicts ``target``", exactly.

    torch.argmax returns the LOWEST index among tied maxima, so the prediction
    differs from ``target`` exactly when some competitor c satisfies

        logits[c] >= logits[target]   if c < target   (c wins the tie)
        logits[c] >  logits[target]   if c > target   (target wins the tie)

    Two weaker variants were previously selectable and are deliberately gone:

      * "strict_competitor" required logits[c] > logits[target] for every c,
        dropping the c < target tie case. It under-approximates
        misclassification, so it returns UNSAT -- and previously reported
        certified_robust=True -- on instances that do have a counterexample.
        On the shipped 11x11 instance at epsilon 2 it certified robustness
        while the correct encoding finds a genuine 2-pixel flip taking 8 -> 1.
      * "qubo_margin" required only logits[c] >= logits[target] for every c,
        adding the c > target tie case that torch.argmax resolves in favour of
        target. It over-approximates, producing SAT witnesses that do not
        actually change the label. On the shipped 5x5 instance at epsilon 3 it
        returns a "counterexample" that still classifies as 0.
    """
    competitors = [index for index in range(len(logits)) if index != target]
    if not competitors:
        raise ValueError("The network must have at least two output classes.")

    conditions = [
        logits[index] >= logits[target] if index < target else logits[index] > logits[target]
        for index in competitors
    ]
    return Or(conditions)


def verify_instance(
    info_path: str | Path,
    checkpoint_path: str | Path,
    *,
    timeout_seconds: float = 300.0,
    epsilon_override: int | None = None,
    require_clean_correct: bool = True,
    result_json_path: str | Path | None = None,
) -> VerificationResult:
    """Verify one BNN robustness instance exactly with Z3.

    Status interpretation:
      * SAT: a counterexample exists within the perturbation budget.
      * UNSAT: the instance is formally robust for the encoded perturbation set.
      * UNKNOWN: no conclusion, usually because the timeout was reached.
    """
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive.")

    info = read_info_file(info_path)
    layers = load_binary_linear_layers(checkpoint_path)
    architecture = architecture_from_layers(layers)

    if len(info.input_boolean) != architecture[0]:
        raise ValueError(
            f"Info input length is {len(info.input_boolean)}, but checkpoint architecture "
            f"starts with {architecture[0]}."
        )
    if not 0 <= info.target_label < architecture[-1]:
        raise ValueError(
            f"Target label {info.target_label} is outside the output range "
            f"[0, {architecture[-1] - 1}]."
        )

    clean_prediction, clean_logits_array = forward_binary_network(info.input_boolean, layers)
    if require_clean_correct and clean_prediction != info.target_label:
        raise ValueError(
            f"Clean checkpoint prediction is {clean_prediction}, but Info.txt label is "
            f"{info.target_label}. The checkpoint and Info.txt likely do not correspond."
        )

    epsilon = info.epsilon if epsilon_override is None else int(epsilon_override)
    if epsilon < 0:
        raise ValueError("epsilon_override must be nonnegative.")

    perturbable_set = set(info.pixels_to_perturb)
    input_variables: dict[int, Any] = {}
    input_spin_expressions: list[Any] = []
    changed_expressions: list[Any] = []

    for index, original_value in enumerate(info.input_boolean):
        if index in perturbable_set:
            variable = Bool(f"x_{index}")
            input_variables[index] = variable
            input_spin_expressions.append(If(variable, 1, -1))
            changed_expressions.append(
                If(variable != BoolVal(bool(original_value)), 1, 0)
            )
        else:
            input_spin_expressions.append(1 if original_value == 1 else -1)

    solver = Solver()
    solver.set(timeout=max(1, int(round(timeout_seconds * 1000))))

    # The perturbation budget is what makes the query meaningful. Without it
    # every instance is trivially satisfiable, so refuse to run rather than
    # report a result that looks like a counterexample at "epsilon N" but was
    # actually obtained with an unbounded perturbation.
    if not (info.perturbation_bound_included or epsilon_override is not None):
        raise ValueError(
            f"{info.info_path} reports 'Perturbation Bound Constriant included : "
            "False', so it carries no perturbation budget, and no "
            "epsilon_override was supplied. Verifying without a budget would "
            "make every instance trivially satisfiable. Regenerate the QUBO "
            "with args.include_perturbation_bound_constraint = True, or pass an "
            "explicit epsilon_override."
        )
    solver.add(_sum_z3(changed_expressions) <= epsilon)

    logits = _encode_network(input_spin_expressions, layers)
    solver.add(_misclassification_constraint(logits, info.target_label))

    # NOTE: this measures solve time only. The clock starts after the formula
    # has been built, so encoding/construction time is deliberately excluded.
    start_time = time.perf_counter()
    check_result = solver.check()
    runtime_seconds = time.perf_counter() - start_time

    adversarial_input: list[int] | None = None
    adversarial_prediction: int | None = None
    adversarial_logits: list[int] | None = None
    changed_indices: list[int] | None = None
    hamming_distance: int | None = None
    witness_forward_changes_label: bool | None = None
    reason_unknown: str | None = None

    if check_result == sat:
        model = solver.model()
        adversarial_input = list(info.input_boolean)
        for index, variable in input_variables.items():
            adversarial_input[index] = 1 if is_true(
                model.eval(variable, model_completion=True)
            ) else 0

        # zip(..., strict=True) requires Python 3.10; check the lengths instead
        # so that the script also runs on Python 3.9.
        if len(adversarial_input) != len(info.input_boolean):
            raise RuntimeError(
                "Internal error: the witness has length "
                f"{len(adversarial_input)} but the clean input has length "
                f"{len(info.input_boolean)}."
            )
        changed_indices = [
            index
            for index, (before, after) in enumerate(
                zip(info.input_boolean, adversarial_input)
            )
            if before != after
        ]
        hamming_distance = len(changed_indices)
        adversarial_prediction, adversarial_logits_array = forward_binary_network(
            adversarial_input, layers
        )
        adversarial_logits = [int(value) for value in adversarial_logits_array.tolist()]
        witness_forward_changes_label = adversarial_prediction != info.target_label
        status = "SAT"
    elif check_result == unsat:
        status = "UNSAT"
    elif check_result == unknown:
        status = "UNKNOWN"
        reason_unknown = solver.reason_unknown()
    else:  # Defensive fallback.
        status = str(check_result).upper()

    # Never report a conclusion on the solver status alone. A SAT answer counts
    # as a counterexample only once the independent NumPy forward pass confirms
    # that the witness really changes the label, and it must also respect the
    # budget and touch only perturbable coordinates.
    witness_within_budget = hamming_distance is not None and hamming_distance <= epsilon
    witness_only_perturbable = changed_indices is not None and all(
        index in perturbable_set for index in changed_indices
    )
    counterexample_found = bool(
        status == "SAT"
        and witness_forward_changes_label
        and witness_within_budget
        and witness_only_perturbable
    )
    if status == "SAT" and not counterexample_found:
        raise RuntimeError(
            "Z3 returned SAT but the witness failed independent re-evaluation "
            f"(label changed: {witness_forward_changes_label}, hamming "
            f"{hamming_distance} <= epsilon {epsilon}: {witness_within_budget}, "
            f"only perturbable coordinates flipped: {witness_only_perturbable}). "
            "This indicates an encoding bug; the result is not trustworthy."
        )

    result = VerificationResult(
        info_path=str(Path(info_path)),
        checkpoint_path=str(Path(checkpoint_path)),
        specification=SPECIFICATION,
        status=status,
        reason_unknown=reason_unknown,
        runtime_seconds=runtime_seconds,
        timeout_seconds=float(timeout_seconds),
        epsilon=epsilon,
        bound_included=bool(info.perturbation_bound_included or epsilon_override is not None),
        input_dimension=len(info.input_boolean),
        perturbable_count=len(info.pixels_to_perturb),
        perturbable_indices=list(info.pixels_to_perturb),
        target_label=info.target_label,
        clean_prediction=clean_prediction,
        clean_logits=[int(value) for value in clean_logits_array.tolist()],
        certified_robust=status == "UNSAT",
        counterexample_found=counterexample_found,
        witness_forward_changes_label=witness_forward_changes_label,
        adversarial_prediction=adversarial_prediction,
        adversarial_logits=adversarial_logits,
        adversarial_input_boolean=adversarial_input,
        changed_indices=changed_indices,
        hamming_distance=hamming_distance,
        architecture=architecture,
    )

    if result_json_path is not None:
        result_json_path = Path(result_json_path)
        result_json_path.parent.mkdir(parents=True, exist_ok=True)
        result_json_path.write_text(
            json.dumps(asdict(result), indent=2), encoding="utf-8"
        )

    return result


# -----------------------------------------------------------------------------
# Batch execution and table output
# -----------------------------------------------------------------------------


def _csv_safe_result(result: VerificationResult) -> dict[str, Any]:
    row = asdict(result)
    for key, value in list(row.items()):
        if isinstance(value, list):
            row[key] = json.dumps(value)
    return row


def verify_many(
    info_paths: Iterable[str | Path],
    checkpoint_path: str | Path,
    *,
    output_csv_path: str | Path,
    timeout_seconds: float = 300.0,
    epsilon_override: int | None = None,
) -> list[VerificationResult]:
    """Verify multiple Info files and write one reviewer-table-ready CSV."""
    results: list[VerificationResult] = []
    for info_path in info_paths:
        result = verify_instance(
            info_path,
            checkpoint_path,
            timeout_seconds=timeout_seconds,
            epsilon_override=epsilon_override,
        )
        results.append(result)
        print(
            f"{Path(info_path).name}: {result.status}, "
            f"time={result.runtime_seconds:.6f}s, "
            f"hamming={result.hamming_distance}"
        )

    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    if results:
        rows = [_csv_safe_result(result) for result in results]
        with output_csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    return results



def scan_minimum_adversarial_distance(
    info_path: str | Path,
    checkpoint_path: str | Path,
    *,
    start_epsilon: int = 0,
    max_epsilon: int | None = None,
    timeout_seconds: float = 300.0,
    output_csv_path: str | Path | None = None,
    summary_json_path: str | Path | None = None,
    per_radius_json_dir: str | Path | None = None,
    stop_on_unknown: bool = True,
) -> tuple[RadiusScanSummary, list[VerificationResult]]:
    """Find the exact minimum Hamming distance to a label-changing input.

    The function checks epsilon = start_epsilon, start_epsilon + 1, ... and
    stops at the first SAT result. Because the feasible perturbation set grows
    monotonically with epsilon, the first SAT radius is the exact minimum
    adversarial distance when scanning begins at zero and all smaller radii are
    UNSAT.

    If Z3 returns UNKNOWN before the first SAT radius, the exact minimum cannot
    be established. By default the scan stops immediately in that case.
    """
    info = read_info_file(info_path)

    if start_epsilon < 0:
        raise ValueError("start_epsilon must be nonnegative.")

    requested_max = info.epsilon if max_epsilon is None else int(max_epsilon)
    if requested_max < start_epsilon:
        raise ValueError(
            f"max_epsilon ({requested_max}) must be >= start_epsilon "
            f"({start_epsilon})."
        )

    # Flipping more than all perturbable coordinates has no additional effect.
    effective_max = min(requested_max, len(info.pixels_to_perturb))

    json_dir: Path | None = None
    if per_radius_json_dir is not None:
        json_dir = Path(per_radius_json_dir)
        json_dir.mkdir(parents=True, exist_ok=True)

    results: list[VerificationResult] = []
    first_sat_epsilon: int | None = None
    first_unknown_epsilon: int | None = None

    print("\n=== Z3 Minimum-Adversarial-Distance Scan ===")
    print(f"Info file              : {Path(info_path)}")
    print(f"Checkpoint             : {Path(checkpoint_path)}")
    print(f"Specification          : {SPECIFICATION}")
    print(f"Radius range           : {start_epsilon} ... {effective_max}")
    print(f"Timeout per radius (s) : {timeout_seconds}")
    print("\n epsilon | status  | runtime (s) | witness distance | prediction")
    print("---------+---------+-------------+------------------+-----------")

    for epsilon in range(start_epsilon, effective_max + 1):
        radius_json_path = None
        if json_dir is not None:
            radius_json_path = json_dir / f"epsilon_{epsilon:04d}.json"

        result = verify_instance(
            info_path,
            checkpoint_path,
            timeout_seconds=timeout_seconds,
            epsilon_override=epsilon,
            result_json_path=radius_json_path,
        )
        results.append(result)

        witness_text = "-" if result.hamming_distance is None else str(result.hamming_distance)
        prediction_text = (
            "-" if result.adversarial_prediction is None
            else str(result.adversarial_prediction)
        )
        print(
            f" {epsilon:7d} | {result.status:7s} | "
            f"{result.runtime_seconds:11.6f} | "
            f"{witness_text:16s} | {prediction_text}"
        )

        if result.status == "UNKNOWN":
            first_unknown_epsilon = epsilon
            if stop_on_unknown:
                break

        if result.status == "SAT":
            first_sat_epsilon = epsilon
            break

    total_runtime = float(sum(result.runtime_seconds for result in results))

    # The first SAT radius is an exact minimum only if every smaller radius from
    # zero was checked and proved UNSAT.
    all_before_sat_unsat = False
    if first_sat_epsilon is not None:
        prior_results = [r for r in results if r.epsilon < first_sat_epsilon]
        all_before_sat_unsat = (
            start_epsilon == 0
            and len(prior_results) == first_sat_epsilon
            and all(r.status == "UNSAT" for r in prior_results)
        )

    exact_minimum_proven = first_sat_epsilon is not None and all_before_sat_unsat
    minimum_distance = first_sat_epsilon if exact_minimum_proven else None
    certified_radius = (
        first_sat_epsilon - 1 if exact_minimum_proven and first_sat_epsilon > 0 else None
    )

    if exact_minimum_proven:
        scan_status = "EXACT_MINIMUM_FOUND"
    elif first_unknown_epsilon is not None:
        scan_status = "INCONCLUSIVE_UNKNOWN"
    elif first_sat_epsilon is not None:
        scan_status = "SAT_FOUND_MINIMUM_NOT_PROVEN"
    elif results and all(r.status == "UNSAT" for r in results):
        scan_status = "CERTIFIED_ROBUST_THROUGH_MAX_EPSILON"
        # A full scan beginning at zero proves robustness through effective_max.
        if start_epsilon == 0:
            certified_radius = effective_max
    else:
        scan_status = "INCONCLUSIVE"

    first_result = results[0] if results else verify_instance(
        info_path,
        checkpoint_path,
        timeout_seconds=timeout_seconds,
        epsilon_override=start_epsilon,
    )

    summary = RadiusScanSummary(
        info_path=str(Path(info_path)),
        checkpoint_path=str(Path(checkpoint_path)),
        specification=SPECIFICATION,
        scan_status=scan_status,
        start_epsilon=start_epsilon,
        requested_max_epsilon=requested_max,
        effective_max_epsilon=effective_max,
        tested_radii=[result.epsilon for result in results],
        total_runtime_seconds=total_runtime,
        first_sat_epsilon=first_sat_epsilon,
        minimum_adversarial_distance=minimum_distance,
        certified_robust_radius=certified_radius,
        first_unknown_epsilon=first_unknown_epsilon,
        exact_minimum_proven=exact_minimum_proven,
        target_label=first_result.target_label,
        clean_prediction=first_result.clean_prediction,
        perturbable_count=first_result.perturbable_count,
        architecture=first_result.architecture,
    )

    if output_csv_path is not None and results:
        output_csv_path = Path(output_csv_path)
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        rows = [_csv_safe_result(result) for result in results]
        with output_csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    if summary_json_path is not None:
        summary_json_path = Path(summary_json_path)
        summary_json_path.parent.mkdir(parents=True, exist_ok=True)
        summary_json_path.write_text(
            json.dumps(asdict(summary), indent=2), encoding="utf-8"
        )

    print("\n=== Scan Summary ===")
    print(f"Status                 : {summary.scan_status}")
    print(f"Total solver time (s)  : {summary.total_runtime_seconds:.6f}")
    if summary.exact_minimum_proven:
        print(
            "Minimum adversarial d. : "
            f"{summary.minimum_adversarial_distance}"
        )
        if summary.certified_robust_radius is None:
            print("Certified robust radius: none (counterexample exists at epsilon 0)")
        else:
            print(
                "Certified robust radius: "
                f"{summary.certified_robust_radius}"
            )
    elif summary.scan_status == "CERTIFIED_ROBUST_THROUGH_MAX_EPSILON":
        print(
            "Conclusion             : no counterexample exists through epsilon "
            f"{summary.effective_max_epsilon}"
        )
    elif summary.first_unknown_epsilon is not None:
        print(
            "Conclusion             : exact minimum unresolved because Z3 returned "
            f"UNKNOWN at epsilon {summary.first_unknown_epsilon}"
        )
    elif summary.first_sat_epsilon is not None:
        print(f"First tested SAT radius: {summary.first_sat_epsilon}")
        print("Conclusion             : SAT found, but exact minimum was not proven")

    return summary, results


def print_result(result: VerificationResult) -> None:
    print("\n=== Exact Z3 BNN Verification ===")
    print(f"Info file              : {result.info_path}")
    print(f"Checkpoint             : {result.checkpoint_path}")
    print(f"Architecture           : {' x '.join(map(str, result.architecture))}")
    print(f"Specification          : {result.specification}")
    print(f"Target / clean pred.   : {result.target_label} / {result.clean_prediction}")
    print(f"Perturbable coordinates: {result.perturbable_count}")
    print(f"Epsilon                : {result.epsilon}")
    print(f"Status                 : {result.status}")
    print(f"Runtime (s)            : {result.runtime_seconds:.6f}")

    if result.status == "UNSAT":
        print("Conclusion             : CERTIFIED ROBUST for the encoded perturbation set")
    elif result.status == "SAT":
        print("Conclusion             : COUNTEREXAMPLE FOUND")
        print(f"Changed indices        : {result.changed_indices}")
        print(f"Hamming distance       : {result.hamming_distance}")
        print(f"Adversarial prediction : {result.adversarial_prediction}")
        print(f"Forward label changed  : {result.witness_forward_changes_label}")
        print(f"Adversarial logits     : {result.adversarial_logits}")
    else:
        print(f"Conclusion             : INCONCLUSIVE ({result.reason_unknown})")


# -----------------------------------------------------------------------------
# USER CONFIGURATION
# -----------------------------------------------------------------------------

# Run this script from the project root and edit these constants as needed.
#
# To reproduce Table V of the paper, keep RUN_MODE = "single" (the default) and
# run this script once per row, changing only the Sizes index below:
#
#   InputSize = Sizes[0]   ->  5x5,   31x7x10,  epsilon 8    -> SAT (not robust)
#   InputSize = Sizes[1]   ->  7x7,   63x7x10,  epsilon 32   -> SAT (not robust)
#   InputSize = Sizes[2]   ->  11x11, 127x7x10, epsilon 32   -> SAT (not robust)
#   InputSize = Sizes[3]   ->  28x28, 1023x7x10, epsilon 128 -> SAT (not robust)
#
# The epsilon of each row is the one recorded in that instance's Info.txt, so it
# is picked up automatically; leave SINGLE_EPSILON_OVERRIDE at None.
Sizes = [5, 7, 11, 28]
DataSize = {5: 31, 7: 63, 11: 127, 28: 1023}

InputSize = Sizes[0]   # 5x5 example; use index 1/2/3 for 7x7 / 11x11 / 28x28
InputDataSize = DataSize[InputSize]

INFO_PATH = f"QUBO/{InputSize}x{InputSize}/{InputDataSize}x7x10/Info.txt"
CHECKPOINT_PATH = f"TrainedNN/{InputSize}x{InputSize}/{InputDataSize}x7x10/{InputDataSize}.pth"
TIMEOUT_SECONDS = 600.0

# Choose:
#   "single" -> verify the epsilon recorded in Info.txt. This is the Table V
#               query and is the default.
#   "scan"   -> test epsilon = 0, 1, 2, ... and stop at the first SAT radius,
#               which yields the exact minimum adversarial distance. This is a
#               separate experiment and is much slower for the larger sizes.
RUN_MODE = "single"

# --------------------------- Scan-mode settings ------------------------------
SCAN_START_EPSILON = 0

# None means use the epsilon stored in Info.txt. The value is automatically
# capped at the number of perturbable coordinates.
SCAN_MAX_EPSILON = None

SCAN_CSV_PATH = f"Z3_Results/{InputSize}x{InputSize}/{InputDataSize}x7x10/Z3_Radius_Scan.csv"
SCAN_SUMMARY_JSON_PATH = f"Z3_Results/{InputSize}x{InputSize}/{InputDataSize}x7x10/Z3_Radius_Scan_Summary.json"
SCAN_PER_RADIUS_JSON_DIR = f"Z3_Results/{InputSize}x{InputSize}/{InputDataSize}x7x10/Radius_JSON"

# Stop immediately if any smaller-radius query is UNKNOWN. Continuing could
# find a SAT witness, but it would not prove that the witness radius is minimal.
STOP_ON_UNKNOWN = True

# -------------------------- Single-mode settings -----------------------------
# None means use the epsilon stored in Info.txt.
SINGLE_EPSILON_OVERRIDE = None
SINGLE_RESULT_JSON_PATH = f"Z3_Results/{InputSize}x{InputSize}/{InputDataSize}x7x10/Z3_Single_Result.json"


if __name__ == "__main__":
    if RUN_MODE == "scan":
        scan_minimum_adversarial_distance(
            INFO_PATH,
            CHECKPOINT_PATH,
            start_epsilon=SCAN_START_EPSILON,
            max_epsilon=SCAN_MAX_EPSILON,
            timeout_seconds=TIMEOUT_SECONDS,
            output_csv_path=SCAN_CSV_PATH,
            summary_json_path=SCAN_SUMMARY_JSON_PATH,
            per_radius_json_dir=SCAN_PER_RADIUS_JSON_DIR,
            stop_on_unknown=STOP_ON_UNKNOWN,
        )
    elif RUN_MODE == "single":
        verification_result = verify_instance(
            INFO_PATH,
            CHECKPOINT_PATH,
            timeout_seconds=TIMEOUT_SECONDS,
            epsilon_override=SINGLE_EPSILON_OVERRIDE,
            result_json_path=SINGLE_RESULT_JSON_PATH,
        )
        print_result(verification_result)
    else:
        raise ValueError("RUN_MODE must be either 'scan' or 'single'.")
