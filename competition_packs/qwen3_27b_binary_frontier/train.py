from __future__ import annotations

from typing import Any

import numpy as np

from competition_packs.qwen3_27b_shared import shared


DISTILLATION_PASSES = 4
MIN_BINARY_SCALE = 1e-12


def _initial_binary_scales(matrix: np.ndarray, *, basis_count: int, scale_multiplier: float) -> np.ndarray:
    residual = matrix.reshape(-1).copy()
    scales: list[float] = []
    for _ in range(int(basis_count)):
        scale = (float(np.mean(np.abs(residual))) or 1.0) * max(0.25, float(scale_multiplier))
        signs = np.where(residual >= 0.0, 1.0, -1.0)
        residual -= signs * scale
        scales.append(float(scale))
    return np.asarray(scales, dtype=np.float64)


def _binary_basis_matrices(matrix: np.ndarray, scales: np.ndarray) -> list[np.ndarray]:
    residual = matrix.reshape(-1).copy()
    bases: list[np.ndarray] = []
    for raw_scale in scales:
        scale = max(MIN_BINARY_SCALE, float(raw_scale))
        signs = np.where(residual >= 0.0, 1.0, -1.0)
        bases.append(signs.reshape(matrix.shape))
        residual -= signs * scale
    return bases


def _solve_layer_scales(
    *,
    inputs: np.ndarray,
    matrix: np.ndarray,
    basis_count: int,
    scale_multiplier: float,
) -> list[float]:
    scales = _initial_binary_scales(matrix, basis_count=basis_count, scale_multiplier=scale_multiplier)
    target = (inputs @ matrix).reshape(-1)
    for _ in range(DISTILLATION_PASSES):
        bases = _binary_basis_matrices(matrix, scales)
        features = np.stack([(inputs @ basis).reshape(-1) for basis in bases], axis=1)
        gram = features.T @ features
        rhs = features.T @ target
        ridge = 1e-9 * max(1.0, float(np.trace(gram)) / max(1, int(basis_count)))
        try:
            solved = np.linalg.solve(gram + np.eye(int(basis_count), dtype=np.float64) * ridge, rhs)
        except np.linalg.LinAlgError:
            solved = np.linalg.lstsq(features, target, rcond=1e-8)[0]
        solved = np.asarray(solved, dtype=np.float64)
        solved = np.where(np.isfinite(solved), solved, scales)
        scales = np.maximum(solved, MIN_BINARY_SCALE)
    return [float(f"{value:.12g}") for value in scales]


def _relative_mse(reference: np.ndarray, candidate: np.ndarray) -> float:
    denom = max(float(np.mean(np.square(reference))), MIN_BINARY_SCALE)
    return float(np.mean(np.square(candidate - reference)) / denom)


def _annotate_layerwise_distillation(artifact: dict[str, Any]) -> None:
    """Fit per-layer binary scales against teacher outputs on public proxy activations."""

    rows = {str(row["name"]): row for row in artifact["layers"]}
    inputs = shared._document_matrix(list(shared.CORPUS_TEXTS))
    diagnostics: list[dict[str, Any]] = []

    for spec in shared._sorted_layer_specs():
        row = rows[spec.name]
        matrix = shared.BASE_WEIGHTS[spec.name]
        basis_count = max(1, int(row.get("binary_basis_count", 1)))
        if basis_count > 1:
            row["binary_basis_scales"] = _solve_layer_scales(
                inputs=inputs,
                matrix=matrix,
                basis_count=basis_count,
                scale_multiplier=float(row.get("scale_multiplier", 1.0)),
            )
        shared.attach_q4_rescue_values_for_layer(row, matrix, mode="binary")
        quantized, _summary = shared._quantize_layer(
            matrix,
            mode="binary",
            high_precision_fraction=float(row.get("high_precision_fraction", 0.0)),
            rescue_bits_per_parameter=shared.DEFAULT_RESCUE_BITS_PER_PARAMETER,
            scale_multiplier=float(row.get("scale_multiplier", 1.0)),
            threshold_multiplier=float(row.get("threshold_multiplier", 1.0)),
            rescue_values_q4=row.get("rescue_values_q4"),
            rescue_scales=row.get("rescue_scales"),
            rescue_scale_group_size=row.get("rescue_scale_group_size"),
            rescue_quantization=row.get("rescue_quantization"),
            layer_name=spec.name,
            binary_basis_count=basis_count,
            binary_basis_scales=row.get("binary_basis_scales"),
        )
        teacher_output = inputs @ matrix
        student_output = inputs @ quantized
        rel_mse = _relative_mse(teacher_output, student_output)
        row["distill_calibration_rel_mse"] = float(f"{rel_mse:.12g}")
        diagnostics.append(
            {
                "name": spec.name,
                "basis_count": int(basis_count),
                "calibration_rel_mse": float(f"{rel_mse:.12g}"),
            }
        )
        if spec.name != "lm_head":
            inputs = np.tanh(student_output)

    artifact["distillation"] = {
        "name": "public_proxy_layerwise_output_matching",
        "passes": int(DISTILLATION_PASSES),
        "calibration_rows": int(len(shared.CORPUS_TEXTS)),
        "diagnostics": diagnostics,
    }


def build_submission(*, seed: int, time_budget_seconds: int, debug_dataset_name: str | None = None) -> dict[str, Any]:
    del seed, time_budget_seconds, debug_dataset_name
    artifact = shared.default_submission(quant_mode="binary", kernel_task=False)
    for row in artifact["layers"]:
        row["binary_basis_count"] = 1
        row["high_precision_fraction"] = 0.02
        row["scale_multiplier"] = 1.0
    _annotate_layerwise_distillation(artifact)
    artifact["strategy"] = {
        "name": "layerwise_distilled_binary_q4_rescue_v1",
        "summary": (
            "Fit each binary layer against teacher linear outputs on public proxy activations, roll the "
            "quantized student forward layer by layer, then spend a small q4 rescue allowance on the "
            "largest remaining residuals while staying inside the mostly-binary compressed-size budget."
        ),
    }
    return artifact
