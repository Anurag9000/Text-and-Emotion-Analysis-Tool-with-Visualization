"""Dataset-cohort applicability certificate for the inference/data-processing repository."""
from __future__ import annotations
from pathlib import Path
from typing import Any

SCHEMA = "opf-dataset-cohort-not-applicable/v1"
REPOSITORY = "Anurag9000/Text-and-Emotion-Analysis-Tool-with-Visualization"
APPLICABLE = False
REASON = "retained-source AST authority certifies inference/data processing only and no authored optimizer transaction"


def certificate(root: str | Path | None = None) -> dict[str, Any]:
    repository_root = Path(root or Path(__file__).resolve().parents[1]).resolve()
    launcher = repository_root / "run_all_training.py"
    if not launcher.is_file():
        raise RuntimeError("root scientific authority launcher is missing")
    source = launcher.read_text(encoding="utf-8", errors="strict")
    for marker in (
        "no_trainable_surface_v1.py",
        "strict_coverage",
        "require_literal_opf_mechanism_parity",
        "require_all_retained_trainable_source_reachability",
    ):
        if marker not in source:
            raise RuntimeError(f"root authority no longer proves required invariant: {marker}")
    return {
        "schema": SCHEMA,
        "repository": REPOSITORY,
        "applicable": APPLICABLE,
        "reason": REASON,
        "authority": "run_all_training.py",
        "require_literal_opf_mechanism_parity": True,
        "require_all_retained_trainable_source_reachability": True,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(certificate(), sort_keys=True, separators=(",", ":")))
