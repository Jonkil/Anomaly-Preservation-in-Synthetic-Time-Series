#!/usr/bin/env python3
"""Run "scripts/validate_env.py" from the repository root."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    script = Path(__file__).resolve().parent / "scripts" / "validate_env.py"
    r = subprocess.run([sys.executable, str(script)], check=False)
    raise SystemExit(r.returncode)
