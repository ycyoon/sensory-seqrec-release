"""Verify the downloaded facet banks against the checksums fixed at release.

The banks are distributed outside the repository because of file-size limits, so
integrity is not guaranteed by the version control system. A silently truncated
or mismatched bank would change every reported number, which is exactly the kind
of failure a reproduction attempt must not absorb quietly.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    manifest = json.loads((ROOT / "configs" / "manifest.json").read_text())
    missing, bad = [], []
    for rel, want in sorted(manifest.items()):
        path = ROOT / rel
        if not path.exists():
            missing.append(rel)
            continue
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != want["sha256"]:
            bad.append((rel, want["sha256"], got))
        else:
            print("ok       %s" % rel)
    for rel in missing:
        print("MISSING  %s" % rel, file=sys.stderr)
    for rel, want, got in bad:
        print("MISMATCH %s\n  expected %s\n  got      %s" % (rel, want, got),
              file=sys.stderr)
    if missing or bad:
        print("\nSee banks/README.md for the download location.", file=sys.stderr)
        return 1
    print("\nAll %d banks verified." % len(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
