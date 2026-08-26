#!/usr/bin/env python3
"""One-command exhaustive training entrypoint for this repository."""
from __future__ import annotations
import hashlib, json, os, subprocess, sys, urllib.request
from pathlib import Path

REPOSITORY = "Anurag9000/Text-and-Emotion-Analysis-Tool-with-Visualization"
CONTROLLER_COMMIT = "294af1b35689a2eefb9453a96eec3ed9c66b68ec"
CONTROLLER_SHA256 = "6043605115ddf934433380e892f1f238eb9e4af236c4063350f477bc5cb0d4dc"
SOURCE = f"https://raw.githubusercontent.com/Anurag9000/RigorousRAG/{CONTROLLER_COMMIT}/tools/universal_training_controller.py"
ROOT = Path(__file__).resolve().parent
PROFILE = {
    "repository": REPOSITORY,
    "preferred_training_entrypoints": ["train.py","training.py","run_training.py","scripts/train.py","scripts/train_all.py","scripts/run_training.py"],
    "preferred_dataset_entrypoints": ["prepare_data.py","scripts/prepare_data.py","scripts/download_data.py","scripts/materialize_datasets.py","scripts/dataset_setup.py"],
    "dynamic_registry_covers": [], "extra_jobs": [], "ignore_entrypoints": ["run_all_training.py"],
}

def _sha(data: bytes) -> str: return hashlib.sha256(data).hexdigest()
def main() -> int:
    cache = ROOT / ".training_control" / "universal_training_controller.py"
    local = ROOT / "tools" / "universal_training_controller.py"
    controller = local if local.is_file() and _sha(local.read_bytes()) == CONTROLLER_SHA256 else cache
    if not controller.is_file() or _sha(controller.read_bytes()) != CONTROLLER_SHA256:
        cache.parent.mkdir(parents=True, exist_ok=True)
        data = urllib.request.urlopen(SOURCE, timeout=60).read()
        if _sha(data) != CONTROLLER_SHA256: raise RuntimeError("Pinned training controller checksum mismatch")
        tmp = cache.with_suffix(".tmp"); tmp.write_bytes(data); os.replace(tmp, cache); controller = cache
    env = os.environ.copy(); env["TRAINING_CONTROL_PROFILE"] = json.dumps(PROFILE, separators=(",",":")); env["TRAINING_CONTROL_REPO_ROOT"] = str(ROOT)
    return subprocess.call([sys.executable, str(controller), *sys.argv[1:]], cwd=ROOT, env=env)
if __name__ == "__main__": raise SystemExit(main())
