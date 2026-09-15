"""Dataset-cohort applicability certificate for the inference/data-processing repository.

N/A is not a static declaration here.  The repository's retained-source authority
AST-parses every Python implementation and fails closed on optimizer construction,
backpropagation, estimator fit/partial_fit, Trainer/train calls and other common
training primitives.  This certificate executes that same live audit; a future
training surface therefore invalidates N/A automatically and must receive a real
OPF-visible dataset-cohort DAG.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

SCHEMA = "opf-dataset-cohort-not-applicable/v1"
REPOSITORY = "Anurag9000/Text-and-Emotion-Analysis-Tool-with-Visualization"
APPLICABLE = False
REASON = "live retained-source AST authority certifies inference/data processing only and no authored optimizer transaction"


def certificate(root: str | Path | None = None) -> dict[str, Any]:
    repository_root = Path(root or Path(__file__).resolve().parents[1]).resolve()
    launcher = repository_root / "run_all_training.py"
    control = repository_root / "training_control"
    authority = control / "no_trainable_surface_v1.py"
    if not launcher.is_file() or not authority.is_file():
        raise RuntimeError("inference-only N/A authority inputs are incomplete")

    launcher_source = launcher.read_text(encoding="utf-8", errors="strict")
    for marker in (
        "no_trainable_surface_v1.py",
        "scientific_authority",
        "strict_coverage",
        "require_literal_opf_mechanism_parity",
        "require_all_retained_trainable_source_reachability",
    ):
        if marker not in launcher_source:
            raise RuntimeError(f"root authority no longer proves required invariant: {marker}")

    authority_source = authority.read_text(encoding="utf-8", errors="strict")
    for marker in (
        "TRAINING_ATTRS",
        "TRAINING_NAMES",
        "TRAINING_MODULE_PREFIXES",
        "ast.parse",
        "retained_training_primitives_detected",
        "require_no_trainable_surface",
    ):
        if marker not in authority_source:
            raise RuntimeError(f"retained-source AST authority lost fail-closed invariant: {marker}")

    if str(control) not in sys.path:
        sys.path.insert(0, str(control))
    import no_trainable_surface_v1 as live_authority  # noqa: E402

    payload = live_authority.require_no_trainable_surface()
    if str(payload.get("repository")) != REPOSITORY:
        raise RuntimeError("inference-only authority repository identity drifted")
    if str(payload.get("classification")) != "inference_and_data_processing_only":
        raise RuntimeError(f"unexpected inference-only classification: {payload.get('classification')!r}")
    if payload.get("training_findings") or payload.get("parse_errors") or payload.get("unresolved"):
        raise RuntimeError("live retained-source audit is not clean")

    return {
        "schema": SCHEMA,
        "repository": REPOSITORY,
        "applicable": APPLICABLE,
        "reason": REASON,
        "authority": "run_all_training.py + training_control/no_trainable_surface_v1.py",
        "live_source_classification": payload["classification"],
        "retained_python_files": payload["retained_python_files"],
        "training_findings": payload["training_findings"],
        "parse_errors": payload["parse_errors"],
        "require_literal_opf_mechanism_parity": True,
        "require_all_retained_trainable_source_reachability": True,
        "source_configuration_only": True,
        "training_executed": False,
    }


if __name__ == "__main__":
    print(json.dumps(certificate(), indent=2, sort_keys=True))
