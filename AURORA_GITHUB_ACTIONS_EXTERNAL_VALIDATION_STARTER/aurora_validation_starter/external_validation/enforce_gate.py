#!/usr/bin/env python3
from __future__ import annotations
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.exists():
    print("FAIL: benchmark_summary.json was not produced", file=sys.stderr)
    raise SystemExit(2)
data = json.loads(path.read_text(encoding="utf-8"))
if data.get("validation_pass") is True:
    print("PASS: recognized external-solver benchmark passed its preregistered criteria")
    raise SystemExit(0)
print("FAIL: recognized external-solver benchmark did not pass")
print(json.dumps(data, indent=2))
raise SystemExit(2)
