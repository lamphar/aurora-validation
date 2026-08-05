#!/usr/bin/env python3
from __future__ import annotations
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
try:
    freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
except Exception as exc:
    freeze = f"pip freeze failed: {exc}\n"
(out / "pip-freeze.txt").write_text(freeze, encoding="utf-8")
provenance = {
    "finalized_at_utc": datetime.now(timezone.utc).isoformat(),
    "python": platform.python_version(),
    "platform": platform.platform(),
    "github_sha": __import__("os").environ.get("GITHUB_SHA"),
    "github_run_id": __import__("os").environ.get("GITHUB_RUN_ID"),
    "github_repository": __import__("os").environ.get("GITHUB_REPOSITORY"),
}
(out / "workflow_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
lines = []
for path in sorted(p for p in out.rglob("*") if p.is_file() and p.name != "SHA256SUMS.txt"):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    lines.append(f"{digest}  {path.relative_to(out).as_posix()}")
(out / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Finalized {len(lines)} artifact files")
