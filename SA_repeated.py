"""Repeated simulated-annealing experiment for BNN-verification QUBOs.

This script extends the original one-shot SA solver by running independent,
seeded SA batches and reporting reviewer-ready stability statistics.

Outputs
-------
A timestamped result directory containing:
    repeated_runs.csv
    summary.json
    best_samples.jsonl

Required packages
-----------------
    pip install numpy torch dimod dwave-samplers

Notes
-----
* Energy statistics always run.
* Forward validation is optional. When enabled, the script reads Info.txt,
  Variables.json, and the trained PyTorch checkpoint, decodes perturbation
  variables, and checks each candidate with an independent BNN forward pass.
* If automatic perturbation-variable detection does not match your
  Variables.json naming convention, set MANUAL_PIXEL_TO_QUBO_INDEX below.
"""

from __future__ import annotations

import ast
import csv
import json
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import dimod
from dwave.samplers import SimulatedAnnealingSampler


# =============================================================================
# USER CONFIGURATION
# =============================================================================

SIZES = [5, 7, 11, 28]
INPUT_DATA_SIZE = {5: 31, 7: 63, 11: 127, 28: 1023}

# Select the QUBO/network size to test.
INPUT_SIZE = SIZES[0]
INPUT_DIMENSION = INPUT_DATA_SIZE[INPUT_SIZE]

QUBO_FOLDER = Path(
    f"QUBO/{INPUT_SIZE}x{INPUT_SIZE}/{INPUT_DIMENSION}x7x10"
)
QUBO_PATH = QUBO_FOLDER / "QUBO_W.txt"
INFO_PATH = QUBO_FOLDER / "Info.txt"
VARIABLES_PATH = QUBO_FOLDER / "Variables.json"
CHECKPOINT_PATH = Path(
    f"TrainedNN/{INPUT_SIZE}x{INPUT_SIZE}/{INPUT_DIMENSION}x7x10/"
    f"{INPUT_DIMENSION}.pth"
)

# Repeated-run protocol.
NUM_REPETITIONS = 30
SEEDS = list(range(NUM_REPETITIONS))
NUM_READS = 1024
NUM_SWEEPS = 1000
BETA_SCHEDULE_TYPE = "geometric"  # "linear", "geometric", or custom

# Numerical tolerance used to compare sampled energy with Info.txt target energy.
ENERGY_ATOL = 1e-6

# Forward validation is strongly recommended for the reviewer-facing table.
ENABLE_FORWARD_VALIDATION = True

# When False, an unrecognized Variables.json naming convention produces an
# energy-only experiment instead of stopping the run. Set True after confirming
# the perturbation-variable mapping if forward validation is mandatory.
STRICT_FORWARD_VALIDATION = False

# QUBO perturbation variables are normally binary flip indicators:
#   tau_i = 0 -> keep original pixel
#   tau_i = 1 -> flip original pixel
# Set to "new_value" only if your QUBO variable directly stores the new pixel.
PERTURBATION_ENCODING = "flip"  # "flip" or "new_value"

# Prefixes used to locate perturbation variables in Variables.json.
PERTURBATION_VARIABLE_PREFIXES = ("tau", "perturb")

# How integer identifiers in perturbation-variable names should be interpreted:
#   "auto"     : try absolute pixel indices, then positions 0...P-1
#   "pixel"    : identifier is the absolute input pixel index
#   "ordinal"  : identifier is the position in Info.txt's pixel list
TAU_INDEX_MODE = "auto"

# Optional explicit mapping {absolute_pixel_index: qubo_variable_index}.
# Leave as None to use automatic detection.
# Example:
# MANUAL_PIXEL_TO_QUBO_INDEX = {0: 12, 20: 13, 4: 14, ...}
MANUAL_PIXEL_TO_QUBO_INDEX: dict[int, int] | None = None

# Save each experiment in a unique timestamped directory.
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = Path("SA_Repeated_Results") / (
    f"{INPUT_SIZE}x{INPUT_SIZE}_{INPUT_DIMENSION}x7x10_{RUN_TIMESTAMP}"
)


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass(frozen=True)
class InstanceInfo:
    minimum_energy: float | None
    total_variables: int | None
    pixels_to_perturb: list[int]
    input_boolean: list[int]
    target_label: int
    epsilon: int
    perturbation_bound_included: bool


@dataclass(frozen=True)
class BinaryLinearLayer:
    name: str
    weight: np.ndarray
    bias: np.ndarray


@dataclass
class RunResult:
    repetition: int
    seed: int
    num_reads: int
    num_sweeps: int
    beta_schedule_type: str
    runtime_seconds: float
    best_energy: float
    target_energy: float | None
    best_energy_gap: float | None
    mean_energy: float
    std_energy: float
    median_energy: float
    minimum_energy_occurrences: int
    target_energy_reads: int | None
    target_energy_fraction: float | None
    target_energy_batch_success: bool | None
    lower_than_target_reads: int | None
    forward_validation_available: bool
    valid_counterexample_found: bool | None
    valid_counterexample_reads: int | None
    valid_counterexample_fraction: float | None
    best_valid_energy: float | None
    best_valid_hamming_distance: int | None
    best_valid_prediction: int | None
    best_sample: list[int]


# =============================================================================
# INFO.TXT PARSING
# =============================================================================


def _extract_scalar(
    text: str,
    key: str,
    cast: type,
    default: Any = None,
) -> Any:
    match = re.search(
        rf"^{re.escape(key)}\s*:\s*(.*?)\s*$",
        text,
        flags=re.MULTILINE,
    )
    if match is None:
        return default
    return cast(match.group(1).strip())


def _extract_tensor_list(text: str, key: str) -> list[float]:
    """Extract a possibly multiline tensor([...]) from Info.txt."""
    match = re.search(rf"{re.escape(key)}\s*:\s*tensor\s*\(", text)
    if match is None:
        raise ValueError(f"Could not find '{key} : tensor(...)' in Info.txt.")

    list_start = text.find("[", match.end())
    if list_start < 0:
        raise ValueError(f"Could not find opening '[' for '{key}'.")

    depth = 0
    list_end: int | None = None
    for index in range(list_start, len(text)):
        if text[index] == "[":
            depth += 1
        elif text[index] == "]":
            depth -= 1
            if depth == 0:
                list_end = index + 1
                break

    if list_end is None:
        raise ValueError(f"Could not find closing ']' for '{key}'.")

    value = ast.literal_eval(text[list_start:list_end])
    return [float(item) for item in value]


def read_info_file(path: str | Path) -> InstanceInfo:
    path = Path(path)
    text = path.read_text(encoding="utf-8")

    pixels_match = re.search(
        r"^Pixels perturbed\s*:\s*(\[.*?\])\s*$",
        text,
        flags=re.MULTILINE,
    )
    if pixels_match is None:
        raise ValueError("Could not parse 'Pixels perturbed' from Info.txt.")
    pixels = [int(value) for value in ast.literal_eval(pixels_match.group(1))]

    raw_input = _extract_tensor_list(text, "Input to perturb")
    input_boolean: list[int] = []
    for index, value in enumerate(raw_input):
        rounded = int(round(value))
        if rounded not in (0, 1) or not np.isclose(value, rounded):
            raise ValueError(
                f"Info.txt input entry {index} is {value}; expected 0 or 1."
            )
        input_boolean.append(rounded)

    label = _extract_scalar(text, "Label to perturb", float)
    if label is None or not float(label).is_integer():
        raise ValueError(f"Invalid label in Info.txt: {label!r}.")

    epsilon = _extract_scalar(text, "Epsilon", int)
    if epsilon is None or epsilon < 0:
        raise ValueError(f"Invalid epsilon in Info.txt: {epsilon!r}.")

    bound_text = _extract_scalar(
        text,
        "Perturbation Bound Constriant included",
        str,
        default=None,
    )
    if bound_text is None:
        bound_text = _extract_scalar(
            text,
            "Perturbation Bound Constraint included",
            str,
            default="True",
        )

    return InstanceInfo(
        minimum_energy=_extract_scalar(text, "Minimum Energy", float),
        total_variables=_extract_scalar(text, "Total Variables", int),
        pixels_to_perturb=pixels,
        input_boolean=input_boolean,
        target_label=int(label),
        epsilon=int(epsilon),
        perturbation_bound_included=(
            str(bound_text).strip().lower() in {"true", "1", "yes"}
        ),
    )


# =============================================================================
# QUBO AND VARIABLE-MAPPING HELPERS
# =============================================================================


def dense_qubo_to_dict(matrix: np.ndarray, atol: float = 0.0) -> dict[tuple[int, int], float]:
    """Convert the saved dense QUBO matrix to Ocean's QUBO dictionary.

    Every stored nonzero entry is retained. This preserves the energy represented
    by the file even when coefficients are not restricted to one triangle.
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"QUBO matrix must be square; received {matrix.shape}.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("QUBO matrix contains NaN or infinite values.")

    rows, columns = np.nonzero(np.abs(matrix) > atol)
    return {
        (int(row), int(column)): float(matrix[row, column])
        for row, column in zip(rows, columns, strict=True)
    }


def _int_like(value: Any) -> int | None:
    try:
        integer = int(value)
    except (TypeError, ValueError):
        return None
    return integer if str(value).strip() == str(integer) else None


def load_index_to_variable(path: str | Path, expected_count: int) -> list[Any]:
    """Load several common Variables.json layouts into index -> label order."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    if isinstance(data, list):
        labels = data
    elif isinstance(data, dict):
        numeric_keys = [_int_like(key) for key in data.keys()]
        numeric_values = [_int_like(value) for value in data.values()]

        if all(value is not None for value in numeric_keys):
            labels = [None] * (max(numeric_keys) + 1)
            for key, label in data.items():
                labels[int(key)] = label
        elif all(value is not None for value in numeric_values):
            labels = [None] * (max(numeric_values) + 1)
            for label, index in data.items():
                labels[int(index)] = label
        else:
            raise ValueError(
                "Variables.json dictionary must map indices to labels or labels to indices."
            )
    else:
        raise ValueError("Variables.json must contain a list or dictionary.")

    if len(labels) != expected_count:
        raise ValueError(
            f"Variables.json contains {len(labels)} variables, but QUBO has "
            f"{expected_count}."
        )
    if any(label is None for label in labels):
        raise ValueError("Variables.json index mapping contains missing entries.")
    return labels


def _label_text(label: Any) -> str:
    if isinstance(label, str):
        return label
    try:
        return json.dumps(label, separators=(",", ":"))
    except TypeError:
        return repr(label)


def _label_integer_tokens(label: Any) -> list[int]:
    return [int(token) for token in re.findall(r"\d+", _label_text(label))]


def _looks_like_perturbation_variable(label: Any) -> bool:
    text = _label_text(label).lower()
    return any(prefix.lower() in text for prefix in PERTURBATION_VARIABLE_PREFIXES)


def _detect_mapping_for_identifiers(
    labels: Sequence[Any],
    desired_identifiers: Sequence[int],
) -> dict[int, int] | None:
    """Map each desired identifier to exactly one tau-like QUBO index."""
    mapping: dict[int, int] = {}
    for identifier in desired_identifiers:
        matches = [
            index
            for index, label in enumerate(labels)
            if _looks_like_perturbation_variable(label)
            and identifier in _label_integer_tokens(label)
        ]
        if len(matches) != 1:
            return None
        mapping[int(identifier)] = int(matches[0])
    return mapping


def detect_pixel_to_qubo_index(
    labels: Sequence[Any],
    pixels_to_perturb: Sequence[int],
) -> tuple[dict[int, int], str]:
    """Automatically detect perturbation-variable indices.

    Returns
    -------
    mapping:
        Absolute input pixel -> QUBO variable index.
    mode:
        "manual", "pixel", or "ordinal".
    """
    if MANUAL_PIXEL_TO_QUBO_INDEX is not None:
        mapping = {int(pixel): int(index) for pixel, index in MANUAL_PIXEL_TO_QUBO_INDEX.items()}
        missing = sorted(set(pixels_to_perturb) - set(mapping))
        if missing:
            raise ValueError(f"Manual perturbation mapping is missing pixels: {missing}.")
        return mapping, "manual"

    allowed_modes = {"auto", "pixel", "ordinal"}
    if TAU_INDEX_MODE not in allowed_modes:
        raise ValueError(f"TAU_INDEX_MODE must be one of {sorted(allowed_modes)}.")

    if TAU_INDEX_MODE in {"auto", "pixel"}:
        absolute = _detect_mapping_for_identifiers(labels, pixels_to_perturb)
        if absolute is not None:
            return {
                int(pixel): int(absolute[int(pixel)])
                for pixel in pixels_to_perturb
            }, "pixel"
        if TAU_INDEX_MODE == "pixel":
            raise ValueError("Could not identify all tau variables using absolute pixel indices.")

    if TAU_INDEX_MODE in {"auto", "ordinal"}:
        ordinal_ids = list(range(len(pixels_to_perturb)))
        ordinal = _detect_mapping_for_identifiers(labels, ordinal_ids)
        if ordinal is not None:
            return {
                int(pixel): int(ordinal[position])
                for position, pixel in enumerate(pixels_to_perturb)
            }, "ordinal"

    candidate_labels = [
        f"{index}: {_label_text(label)}"
        for index, label in enumerate(labels)
        if _looks_like_perturbation_variable(label)
    ]
    preview = "\n".join(candidate_labels[:40]) or "<none detected>"
    raise ValueError(
        "Could not automatically map perturbation variables. Set "
        "MANUAL_PIXEL_TO_QUBO_INDEX explicitly. Tau-like labels detected:\n"
        f"{preview}"
    )


# =============================================================================
# INDEPENDENT BNN FORWARD VALIDATION
# =============================================================================


def _unwrap_state_dict(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(checkpoint, Mapping):
        for wrapper_key in ("state_dict", "model_state_dict"):
            wrapped = checkpoint.get(wrapper_key)
            if isinstance(wrapped, Mapping):
                checkpoint = wrapped
                break

    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint is not a state_dict or wrapped state_dict.")

    normalized: dict[str, torch.Tensor] = {}
    for key, value in checkpoint.items():
        if not isinstance(value, torch.Tensor):
            continue
        normalized_key = str(key)
        while normalized_key.startswith("module."):
            normalized_key = normalized_key[len("module.") :]
        normalized[normalized_key] = value.detach().cpu()
    return normalized


def load_binary_linear_layers(path: str | Path) -> list[BinaryLinearLayer]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")

    state = _unwrap_state_dict(checkpoint)
    weight_keys = [
        key
        for key, value in state.items()
        if key.endswith(".weight") and value.ndim == 2
    ]
    if not weight_keys:
        raise ValueError("No two-dimensional linear-layer weights found in checkpoint.")

    layers: list[BinaryLinearLayer] = []
    previous_output: int | None = None
    for weight_key in weight_keys:
        name = weight_key[: -len(".weight")]
        raw_weight = state[weight_key].numpy()
        weight = np.where(raw_weight >= 0, 1, -1).astype(np.int64)
        output_count, input_count = weight.shape

        if previous_output is not None and input_count != previous_output:
            raise ValueError(
                f"Checkpoint layer order mismatch at {name}: expected {previous_output} "
                f"inputs, found {input_count}."
            )

        bias_key = f"{name}.bias"
        if bias_key in state:
            raw_bias = state[bias_key].numpy().astype(float)
            rounded = np.rint(raw_bias)
            if not np.allclose(raw_bias, rounded):
                raise ValueError("Forward validator currently requires integer biases.")
            bias = rounded.astype(np.int64)
        else:
            bias = np.zeros(output_count, dtype=np.int64)

        layers.append(BinaryLinearLayer(name=name, weight=weight, bias=bias))
        previous_output = output_count

    return layers


def forward_binary_network(
    input_boolean: Sequence[int],
    layers: Sequence[BinaryLinearLayer],
) -> tuple[int, np.ndarray]:
    activation = np.where(np.asarray(input_boolean, dtype=np.int64) == 1, 1, -1)

    if activation.ndim != 1:
        raise ValueError("Input must be one-dimensional.")
    if activation.size != layers[0].weight.shape[1]:
        raise ValueError(
            f"Input length {activation.size} does not match checkpoint input "
            f"dimension {layers[0].weight.shape[1]}."
        )

    logits: np.ndarray | None = None
    for layer_index, layer in enumerate(layers):
        preactivation = layer.weight @ activation + layer.bias
        if layer_index < len(layers) - 1:
            activation = np.where(preactivation >= 0, 1, -1).astype(np.int64)
        else:
            logits = preactivation.astype(np.int64)

    assert logits is not None
    return int(np.argmax(logits)), logits


def decode_candidate_input(
    sample_bits: np.ndarray,
    info: InstanceInfo,
    pixel_to_qubo_index: Mapping[int, int],
) -> list[int]:
    candidate = list(info.input_boolean)

    if PERTURBATION_ENCODING not in {"flip", "new_value"}:
        raise ValueError("PERTURBATION_ENCODING must be 'flip' or 'new_value'.")

    for pixel in info.pixels_to_perturb:
        qubo_index = pixel_to_qubo_index[int(pixel)]
        bit = int(sample_bits[qubo_index])
        if bit not in (0, 1):
            raise ValueError(f"QUBO sample contains nonbinary value {bit}.")

        if PERTURBATION_ENCODING == "flip":
            candidate[pixel] = int(candidate[pixel] ^ bit)
        else:
            candidate[pixel] = bit

    return candidate


def validate_sample(
    sample_bits: np.ndarray,
    info: InstanceInfo,
    pixel_to_qubo_index: Mapping[int, int],
    layers: Sequence[BinaryLinearLayer],
) -> tuple[bool, int, int]:
    candidate = decode_candidate_input(sample_bits, info, pixel_to_qubo_index)
    hamming_distance = int(
        sum(before != after for before, after in zip(info.input_boolean, candidate, strict=True))
    )
    prediction, _ = forward_binary_network(candidate, layers)

    within_budget = (
        not info.perturbation_bound_included
        or hamming_distance <= info.epsilon
    )
    label_changed = prediction != info.target_label
    return bool(within_budget and label_changed), hamming_distance, prediction


# =============================================================================
# SA EXECUTION AND STATISTICS
# =============================================================================


def run_dwave_sa(
    bqm: dimod.BinaryQuadraticModel,
    *,
    num_reads: int,
    num_sweeps: int,
    beta_schedule_type: str,
    seed: int,
) -> tuple[dimod.SampleSet, float]:
    """Run one independently seeded Ocean SA batch."""
    sampler = SimulatedAnnealingSampler()
    start = time.perf_counter()
    sampleset = sampler.sample(
        bqm,
        num_reads=num_reads,
        num_sweeps=num_sweeps,
        beta_schedule_type=beta_schedule_type,
        seed=seed,
    )
    runtime = time.perf_counter() - start
    return sampleset, runtime


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    cutoff = 0.5 * np.sum(sorted_weights)
    return float(sorted_values[np.searchsorted(cumulative, cutoff, side="left")])


def sample_row_to_ordered_bits(
    sample_row: np.ndarray,
    sample_variables: Sequence[Any],
    variable_count: int,
) -> np.ndarray:
    positions = {variable: position for position, variable in enumerate(sample_variables)}
    missing = [index for index in range(variable_count) if index not in positions]
    if missing:
        raise ValueError(f"SampleSet is missing QUBO variables: {missing[:10]}.")
    return np.asarray(
        [sample_row[positions[index]] for index in range(variable_count)],
        dtype=np.int8,
    )


def analyze_sampleset(
    *,
    sampleset: dimod.SampleSet,
    runtime_seconds: float,
    repetition: int,
    seed: int,
    variable_count: int,
    target_energy: float | None,
    info: InstanceInfo | None,
    pixel_to_qubo_index: Mapping[int, int] | None,
    layers: Sequence[BinaryLinearLayer] | None,
) -> RunResult:
    energies = np.asarray(sampleset.record.energy, dtype=float)
    occurrences = np.asarray(sampleset.record.num_occurrences, dtype=np.int64)

    if int(np.sum(occurrences)) != NUM_READS:
        raise RuntimeError(
            f"SampleSet contains {int(np.sum(occurrences))} reads; expected {NUM_READS}."
        )

    mean_energy = float(np.average(energies, weights=occurrences))
    variance = float(np.average((energies - mean_energy) ** 2, weights=occurrences))
    std_energy = float(np.sqrt(max(variance, 0.0)))
    median_energy = _weighted_median(energies, occurrences)

    best_index = int(np.argmin(energies))
    best_energy = float(energies[best_index])
    best_bits = sample_row_to_ordered_bits(
        sampleset.record.sample[best_index],
        list(sampleset.variables),
        variable_count,
    )
    best_occurrences = int(
        np.sum(occurrences[np.isclose(energies, best_energy, atol=ENERGY_ATOL, rtol=0.0)])
    )

    if target_energy is None:
        best_gap = None
        target_reads = None
        target_fraction = None
        target_success = None
        lower_than_target_reads = None
    else:
        target_mask = np.isclose(
            energies,
            target_energy,
            atol=ENERGY_ATOL,
            rtol=0.0,
        )
        target_reads = int(np.sum(occurrences[target_mask]))
        target_fraction = float(target_reads / NUM_READS)
        target_success = target_reads > 0
        best_gap = float(best_energy - target_energy)
        lower_than_target_reads = int(
            np.sum(occurrences[energies < target_energy - ENERGY_ATOL])
        )

    validation_available = (
        info is not None
        and pixel_to_qubo_index is not None
        and layers is not None
    )

    valid_reads: int | None = None
    valid_fraction: float | None = None
    valid_found: bool | None = None
    best_valid_energy: float | None = None
    best_valid_hamming: int | None = None
    best_valid_prediction: int | None = None

    if validation_available:
        valid_reads = 0
        valid_candidates: list[tuple[float, int, int]] = []
        for row_index in np.argsort(energies):
            bits = sample_row_to_ordered_bits(
                sampleset.record.sample[row_index],
                list(sampleset.variables),
                variable_count,
            )
            valid, hamming, prediction = validate_sample(
                bits,
                info,
                pixel_to_qubo_index,
                layers,
            )
            if valid:
                valid_reads += int(occurrences[row_index])
                valid_candidates.append(
                    (float(energies[row_index]), int(hamming), int(prediction))
                )

        valid_found = valid_reads > 0
        valid_fraction = float(valid_reads / NUM_READS)
        if valid_candidates:
            best_valid_energy, best_valid_hamming, best_valid_prediction = min(
                valid_candidates,
                key=lambda value: value[0],
            )

    return RunResult(
        repetition=repetition,
        seed=seed,
        num_reads=NUM_READS,
        num_sweeps=NUM_SWEEPS,
        beta_schedule_type=BETA_SCHEDULE_TYPE,
        runtime_seconds=float(runtime_seconds),
        best_energy=best_energy,
        target_energy=target_energy,
        best_energy_gap=best_gap,
        mean_energy=mean_energy,
        std_energy=std_energy,
        median_energy=median_energy,
        minimum_energy_occurrences=best_occurrences,
        target_energy_reads=target_reads,
        target_energy_fraction=target_fraction,
        target_energy_batch_success=target_success,
        lower_than_target_reads=lower_than_target_reads,
        forward_validation_available=validation_available,
        valid_counterexample_found=valid_found,
        valid_counterexample_reads=valid_reads,
        valid_counterexample_fraction=valid_fraction,
        best_valid_energy=best_valid_energy,
        best_valid_hamming_distance=best_valid_hamming,
        best_valid_prediction=best_valid_prediction,
        best_sample=[int(value) for value in best_bits.tolist()],
    )


# =============================================================================
# OUTPUT
# =============================================================================


def _csv_row(result: RunResult) -> dict[str, Any]:
    row = asdict(result)
    row.pop("best_sample")
    return row


def write_run_outputs(results: Sequence[RunResult], summary: Mapping[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = OUTPUT_DIR / "repeated_runs.csv"
    rows = [_csv_row(result) for result in results]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    samples_path = OUTPUT_DIR / "best_samples.jsonl"
    with samples_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(
                json.dumps(
                    {
                        "repetition": result.repetition,
                        "seed": result.seed,
                        "best_energy": result.best_energy,
                        "best_sample": result.best_sample,
                    }
                )
                + "\n"
            )

    summary_path = OUTPUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(dict(summary), indent=2), encoding="utf-8")


def _mean_or_none(values: Sequence[float | None]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    return float(np.mean(numeric)) if numeric else None


def _std_or_none(values: Sequence[float | None]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    return float(np.std(numeric, ddof=1)) if len(numeric) > 1 else (0.0 if numeric else None)


def build_summary(
    results: Sequence[RunResult],
    *,
    info: InstanceInfo,
    variable_count: int,
    mapping_mode: str | None,
    clean_prediction: int | None,
) -> dict[str, Any]:
    best_energies = np.asarray([result.best_energy for result in results], dtype=float)
    runtimes = np.asarray([result.runtime_seconds for result in results], dtype=float)

    target_success_values = [
        result.target_energy_batch_success
        for result in results
        if result.target_energy_batch_success is not None
    ]
    valid_success_values = [
        result.valid_counterexample_found
        for result in results
        if result.valid_counterexample_found is not None
    ]

    return {
        "input_size": INPUT_SIZE,
        "architecture": f"{INPUT_DIMENSION}x7x10",
        "qubo_path": str(QUBO_PATH),
        "info_path": str(INFO_PATH),
        "variables_path": str(VARIABLES_PATH),
        "checkpoint_path": str(CHECKPOINT_PATH),
        "qubo_variables": variable_count,
        "repetitions": len(results),
        "seeds": [result.seed for result in results],
        "num_reads_per_repetition": NUM_READS,
        "num_sweeps": NUM_SWEEPS,
        "beta_schedule_type": BETA_SCHEDULE_TYPE,
        "total_reads": len(results) * NUM_READS,
        "target_energy": info.minimum_energy,
        "epsilon": info.epsilon,
        "perturbable_pixels": len(info.pixels_to_perturb),
        "target_label": info.target_label,
        "clean_prediction": clean_prediction,
        "forward_validation_enabled": ENABLE_FORWARD_VALIDATION,
        "strict_forward_validation": STRICT_FORWARD_VALIDATION,
        "forward_validation_available": all(
            result.forward_validation_available for result in results
        ),
        "perturbation_mapping_mode": mapping_mode,
        "best_energy_mean": float(np.mean(best_energies)),
        "best_energy_std": float(np.std(best_energies, ddof=1)) if len(results) > 1 else 0.0,
        "best_energy_median": float(np.median(best_energies)),
        "best_energy_min": float(np.min(best_energies)),
        "best_energy_max": float(np.max(best_energies)),
        "best_energy_gap_mean": _mean_or_none(
            [result.best_energy_gap for result in results]
        ),
        "best_energy_gap_std": _std_or_none(
            [result.best_energy_gap for result in results]
        ),
        "target_energy_successful_batches": int(sum(bool(value) for value in target_success_values))
        if target_success_values
        else None,
        "target_energy_batch_success_rate": float(np.mean(target_success_values))
        if target_success_values
        else None,
        "target_energy_read_fraction_mean": _mean_or_none(
            [result.target_energy_fraction for result in results]
        ),
        "target_energy_read_fraction_std": _std_or_none(
            [result.target_energy_fraction for result in results]
        ),
        "valid_counterexample_successful_batches": int(sum(bool(value) for value in valid_success_values))
        if valid_success_values
        else None,
        "valid_counterexample_batch_success_rate": float(np.mean(valid_success_values))
        if valid_success_values
        else None,
        "valid_counterexample_read_fraction_mean": _mean_or_none(
            [result.valid_counterexample_fraction for result in results]
        ),
        "valid_counterexample_read_fraction_std": _std_or_none(
            [result.valid_counterexample_fraction for result in results]
        ),
        "runtime_mean_seconds": float(np.mean(runtimes)),
        "runtime_std_seconds": float(np.std(runtimes, ddof=1)) if len(results) > 1 else 0.0,
        "runtime_median_seconds": float(np.median(runtimes)),
        "runtime_q1_seconds": float(np.percentile(runtimes, 25)),
        "runtime_q3_seconds": float(np.percentile(runtimes, 75)),
        "runtime_total_seconds": float(np.sum(runtimes)),
        "timestamp": RUN_TIMESTAMP,
    }


def print_summary(summary: Mapping[str, Any]) -> None:
    print("\n=== Repeated SA Summary ===")
    print(f"QUBO                  : {summary['qubo_path']}")
    print(f"QUBO variables        : {summary['qubo_variables']}")
    print(f"Repetitions           : {summary['repetitions']}")
    print(f"Reads / sweeps        : {NUM_READS} / {NUM_SWEEPS}")
    print(f"Target energy         : {summary['target_energy']}")
    print(
        "Best energy           : "
        f"{summary['best_energy_mean']:.6f} +/- {summary['best_energy_std']:.6f}"
    )
    print(
        "Best energy range     : "
        f"[{summary['best_energy_min']:.6f}, {summary['best_energy_max']:.6f}]"
    )

    if summary["target_energy_batch_success_rate"] is not None:
        print(
            "Target-energy success : "
            f"{summary['target_energy_successful_batches']}/{summary['repetitions']} "
            f"({100.0 * summary['target_energy_batch_success_rate']:.2f}%)"
        )
        print(
            "Target read fraction  : "
            f"{summary['target_energy_read_fraction_mean']:.6f} +/- "
            f"{summary['target_energy_read_fraction_std']:.6f}"
        )

    if summary["valid_counterexample_batch_success_rate"] is not None:
        print(
            "Valid CE success       : "
            f"{summary['valid_counterexample_successful_batches']}/"
            f"{summary['repetitions']} "
            f"({100.0 * summary['valid_counterexample_batch_success_rate']:.2f}%)"
        )
        print(
            "Valid CE read fraction : "
            f"{summary['valid_counterexample_read_fraction_mean']:.6f} +/- "
            f"{summary['valid_counterexample_read_fraction_std']:.6f}"
        )

    print(
        "Runtime median (s)    : "
        f"{summary['runtime_median_seconds']:.6f} "
        f"[IQR {summary['runtime_q1_seconds']:.6f}, "
        f"{summary['runtime_q3_seconds']:.6f}]"
    )
    print(f"Output directory      : {OUTPUT_DIR}")


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================


def main() -> None:
    if NUM_REPETITIONS <= 0:
        raise ValueError("NUM_REPETITIONS must be positive.")
    if len(SEEDS) != NUM_REPETITIONS:
        raise ValueError(
            f"SEEDS has length {len(SEEDS)}, expected {NUM_REPETITIONS}."
        )
    if len(set(SEEDS)) != len(SEEDS):
        raise ValueError("SEEDS must be distinct for independent repetitions.")
    if NUM_READS <= 0 or NUM_SWEEPS <= 0:
        raise ValueError("NUM_READS and NUM_SWEEPS must be positive.")

    for path in (QUBO_PATH, INFO_PATH):
        if not path.exists():
            raise FileNotFoundError(path)

    info = read_info_file(INFO_PATH)
    q_matrix = np.loadtxt(QUBO_PATH)
    qubo = dense_qubo_to_dict(q_matrix)
    variable_count = int(q_matrix.shape[0])

    if info.total_variables is not None and info.total_variables != variable_count:
        raise ValueError(
            f"Info.txt reports {info.total_variables} variables, but QUBO_W.txt "
            f"contains {variable_count}."
        )

    bqm = dimod.BinaryQuadraticModel.from_qubo(qubo)
    # Preserve zero-bias variables that may be absent from the sparse QUBO dict.
    for variable in range(variable_count):
        if variable not in bqm.variables:
            bqm.add_variable(variable, 0.0)

    layers: list[BinaryLinearLayer] | None = None
    pixel_to_qubo_index: dict[int, int] | None = None
    mapping_mode: str | None = None
    clean_prediction: int | None = None

    validation_active = False
    if ENABLE_FORWARD_VALIDATION:
        try:
            for path in (VARIABLES_PATH, CHECKPOINT_PATH):
                if not path.exists():
                    raise FileNotFoundError(
                        f"Forward validation requires the missing file: {path}"
                    )

            labels = load_index_to_variable(VARIABLES_PATH, variable_count)
            pixel_to_qubo_index, mapping_mode = detect_pixel_to_qubo_index(
                labels,
                info.pixels_to_perturb,
            )
            layers = load_binary_linear_layers(CHECKPOINT_PATH)
            clean_prediction, clean_logits = forward_binary_network(
                info.input_boolean,
                layers,
            )
            if clean_prediction != info.target_label:
                raise ValueError(
                    f"Checkpoint predicts {clean_prediction}, while Info.txt label is "
                    f"{info.target_label}. Files do not describe the same instance."
                )

            validation_active = True
            print("=== Forward-validation setup ===")
            print(f"Clean prediction       : {clean_prediction}")
            print(f"Clean logits           : {clean_logits.tolist()}")
            print(f"Perturbation map mode  : {mapping_mode}")
            print(f"Mapped tau variables   : {len(pixel_to_qubo_index)}")
        except Exception as error:
            if STRICT_FORWARD_VALIDATION:
                raise
            layers = None
            pixel_to_qubo_index = None
            mapping_mode = None
            clean_prediction = None
            print("WARNING: Forward validation could not be initialized.")
            print(f"Reason: {error}")
            print("Continuing with repeated-run energy statistics only.")

    print("\n=== Repeated Ocean SA ===")
    print(f"QUBO                  : {QUBO_PATH}")
    print(f"Variables             : {variable_count}")
    print(f"Target energy         : {info.minimum_energy}")
    print(f"Repetitions           : {NUM_REPETITIONS}")
    print(f"Reads / sweeps        : {NUM_READS} / {NUM_SWEEPS}")
    print(f"Schedule              : {BETA_SCHEDULE_TYPE}")

    results: list[RunResult] = []
    for repetition, seed in enumerate(SEEDS, start=1):
        sampleset, runtime = run_dwave_sa(
            bqm,
            num_reads=NUM_READS,
            num_sweeps=NUM_SWEEPS,
            beta_schedule_type=BETA_SCHEDULE_TYPE,
            seed=int(seed),
        )
        result = analyze_sampleset(
            sampleset=sampleset,
            runtime_seconds=runtime,
            repetition=repetition,
            seed=int(seed),
            variable_count=variable_count,
            target_energy=info.minimum_energy,
            info=info if validation_active else None,
            pixel_to_qubo_index=pixel_to_qubo_index,
            layers=layers,
        )
        results.append(result)

        target_text = (
            "N/A"
            if result.target_energy_batch_success is None
            else str(result.target_energy_batch_success)
        )
        valid_text = (
            "N/A"
            if result.valid_counterexample_found is None
            else str(result.valid_counterexample_found)
        )
        print(
            f"Run {repetition:02d}/{NUM_REPETITIONS} | seed={seed:>4} | "
            f"best={result.best_energy:>12.6f} | "
            f"target={target_text:<5} | valid_CE={valid_text:<5} | "
            f"time={runtime:.3f}s"
        )

    summary = build_summary(
        results,
        info=info,
        variable_count=variable_count,
        mapping_mode=mapping_mode,
        clean_prediction=clean_prediction,
    )
    write_run_outputs(results, summary)
    print_summary(summary)


if __name__ == "__main__":
    main()
