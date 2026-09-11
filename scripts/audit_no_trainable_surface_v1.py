#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training_control.no_trainable_surface_v1 import audit  # noqa: E402

OUTPUT = ROOT / "artifacts" / "training_control" / "no_trainable_surface_v1.json"


def main() -> int:
    payload = audit()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(OUTPUT.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(OUTPUT)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if bool(payload["complete"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
