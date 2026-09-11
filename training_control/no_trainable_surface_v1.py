#!/usr/bin/env python3
"""Fail-closed retained-source audit for repositories with inference-only ML code.

The repository uses pretrained inference models (Transformers pipelines, VADER,
spaCy and locally hosted LLM inference) but does not currently declare an authored
optimizer/training transaction.  This module makes that a *provable invariant*
rather than an assumption: every retained Python file is parsed and any optimizer,
backpropagation, estimator fit, trainer.train, or common training-loop primitive
causes the central authority to fail closed.

The certificate is source/configuration evidence only.  It never claims model
training, benchmark execution, or dataset validation occurred.
"""
from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PARTS = {".git", ".training_control", "__pycache__", ".venv", "venv", "build", "dist"}
EXCLUDED_FILES = {"run_all_training.py", "training_control/no_trainable_surface_v1.py"}

TRAINING_ATTRS = {
    "fit", "fit_generator", "partial_fit", "train_on_batch", "backward", "zero_grad",
    "step", "train", "training_step", "optimizer_step", "manual_backward",
}
TRAINING_NAMES = {
    "Trainer", "Seq2SeqTrainer", "TrainingArguments", "Optimizer", "SGD", "Adam", "AdamW",
    "Adagrad", "Adadelta", "RMSprop", "LBFGS", "SparseAdam",
}
TRAINING_MODULE_PREFIXES = (
    "torch.optim", "tensorflow.keras.optimizers", "keras.optimizers",
    "pytorch_lightning", "lightning.pytorch",
)
FIT_ALLOWLIST_RECEIVERS = {
    # Inference/data transformations whose APIs happen to contain 'fit' are not
    # present in the current repository; the set is intentionally empty so a new
    # fit call must be reviewed explicitly rather than silently accepted.
}


@dataclass(frozen=True, slots=True)
class Finding:
    path: str
    line: int
    kind: str
    symbol: str


class Scanner(ast.NodeVisitor):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.findings: list[Finding] = []

    def add(self, node: ast.AST, kind: str, symbol: str) -> None:
        self.findings.append(
            Finding(
                path=self.path.relative_to(ROOT).as_posix(),
                line=int(getattr(node, "lineno", 0) or 0),
                kind=kind,
                symbol=symbol,
            )
        )

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name.startswith(TRAINING_MODULE_PREFIXES):
                self.add(node, "training_import", alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        if module.startswith(TRAINING_MODULE_PREFIXES):
            self.add(node, "training_import", module)
        for alias in node.names:
            if alias.name in TRAINING_NAMES:
                self.add(node, "training_symbol_import", f"{module}.{alias.name}".strip("."))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Attribute):
            if func.attr in TRAINING_ATTRS:
                receiver = ""
                if isinstance(func.value, ast.Name):
                    receiver = func.value.id
                if receiver not in FIT_ALLOWLIST_RECEIVERS:
                    self.add(node, "training_call", func.attr)
        elif isinstance(func, ast.Name) and func.id in TRAINING_NAMES:
            self.add(node, "training_constructor", func.id)
        self.generic_visit(node)


def retained_python_files() -> tuple[Path, ...]:
    files: list[Path] = []
    for path in ROOT.rglob("*.py"):
        relative = path.relative_to(ROOT).as_posix()
        if relative in EXCLUDED_FILES:
            continue
        if any(part in EXCLUDED_PARTS for part in path.relative_to(ROOT).parts):
            continue
        files.append(path)
    return tuple(sorted(files))


def audit() -> dict[str, object]:
    findings: list[Finding] = []
    parse_errors: list[dict[str, object]] = []
    files = retained_python_files()
    for path in files:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
        except SyntaxError as exc:
            parse_errors.append(
                {
                    "path": path.relative_to(ROOT).as_posix(),
                    "line": int(exc.lineno or 0),
                    "message": str(exc),
                }
            )
            continue
        scanner = Scanner(path)
        scanner.visit(tree)
        findings.extend(scanner.findings)

    unresolved: list[dict[str, object]] = []
    if parse_errors:
        unresolved.append({"type": "python_parse_errors", "values": parse_errors})
    if findings:
        unresolved.append(
            {
                "type": "retained_training_primitives_detected",
                "values": [asdict(row) for row in findings],
            }
        )

    return {
        "schema_version": 1,
        "repository": "Anurag9000/Text-and-Emotion-Analysis-Tool-with-Visualization",
        "classification": "inference_and_data_processing_only" if not unresolved else "training_surface_detected",
        "retained_python_files": [path.relative_to(ROOT).as_posix() for path in files],
        "training_findings": [asdict(row) for row in findings],
        "parse_errors": parse_errors,
        "unresolved": unresolved,
        "complete": not unresolved,
        "source_configuration_only": True,
        "training_executed_by_audit": False,
        "benchmark_claim_emitted": False,
    }


def require_no_trainable_surface() -> dict[str, object]:
    payload = audit()
    if not bool(payload["complete"]):
        raise RuntimeError(
            "repository is no longer provably inference-only; retained training primitives require "
            "a real central training DAG"
        )
    return payload
