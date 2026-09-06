#!/usr/bin/env python3
"""One-command exhaustive training entrypoint for this repository."""
from __future__ import annotations
import hashlib, json, os, subprocess, sys, urllib.request
from pathlib import Path

REPOSITORY = "Anurag9000/Text-and-Emotion-Analysis-Tool-with-Visualization"
CONTROLLER_COMMIT = "7b9ceb12d6c5fdef33eefd73eaea4c027b941737"
CONTROLLER_BLOB = "4ecb86674c3baa91c88ff57a8699decce26c528d"
SOURCE = f"https://raw.githubusercontent.com/Anurag9000/RigorousRAG/{CONTROLLER_COMMIT}/tools/universal_training_controller_entry.py"
ROOT = Path(__file__).resolve().parent
PROFILE = {
    "repository": REPOSITORY,
    "preferred_training_entrypoints": ["train.py","training.py","run_training.py","scripts/train.py","scripts/train_all.py","scripts/run_training.py"],
    "preferred_dataset_entrypoints": ["prepare_data.py","scripts/prepare_data.py","scripts/download_data.py","scripts/materialize_datasets.py","scripts/dataset_setup.py"],
    "dynamic_registry_covers": [], "extra_jobs": [], "ignore_entrypoints": ["run_all_training.py"],
    "strict_coverage": True, "require_native_resume": True, "require_exact_resume": True,
    "require_training_exact_resume": True, "require_training_early_stopping": True,
    "require_dag_enforcement": True, "require_model_surface_accounting": True,
    "require_literal_opf_mechanism_parity": True, "require_well_formed_training_exemptions": True,
}

def _sha(data: bytes) -> str: return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
def main() -> int:
    cache = ROOT / ".training_control" / "universal_training_controller_entry.py"
    if not cache.is_file() or _sha(cache.read_bytes()) != CONTROLLER_BLOB:
        cache.parent.mkdir(parents=True, exist_ok=True)
        data = urllib.request.urlopen(SOURCE, timeout=60).read()
        if _sha(data) != CONTROLLER_BLOB: raise RuntimeError("Pinned training controller checksum mismatch")
        tmp = cache.with_suffix(".tmp"); tmp.write_bytes(data); os.replace(tmp, cache)
    env = os.environ.copy(); env["TRAINING_CONTROL_PROFILE"] = json.dumps(PROFILE, separators=(",",":")); env["TRAINING_CONTROL_REPO_ROOT"] = str(ROOT); env.setdefault("TRAINING_CONTROL_TERMINATION_GRACE_SEC", "30")
    return subprocess.call([sys.executable, str(cache), *sys.argv[1:]], cwd=ROOT, env=env)
if __name__ == "__main__": raise SystemExit(main())