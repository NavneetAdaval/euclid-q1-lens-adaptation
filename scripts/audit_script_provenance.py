"""
audit_script_provenance.py
==========================
REPORT ONLY (brief_dilute10_part2 section B). For every run record under ``runs/`` (and
the Kaggle run records under ``kaggle/*/runs/``, where CORAL x nb3's confirmatory records
live) that stores script hashes, compare each recorded hash with the file in ``scripts/``
today. Edits nothing.

For a differing script it tries, in order:
  1. APPEND-ONLY: is the recorded version an exact LINE PREFIX of today's file? (the shape
     PROJECTv7.md 7.21 proved for ``da_losses.py``)
  2. EXECUTED VERSION ON DISK: every ``*.py`` of that name under the repo and its sibling
     directories in ``C:\\Users\\krris\\ML`` is hashed; matches are listed.

    python -u scripts/audit_script_provenance.py --out audit/script_provenance/report.json
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ML = os.path.dirname(REPO)

KNOWN = {
    "da_losses.py": "append-only: BNM was appended in round 6 (PROJECTv7.md 7.21)",
    "run_dann_r5.py": ("three versions recorded: arm 1 fold 0 (9b6e38669c), arm 2 fold 0 "
                       "(4fd5964f37), DANN folds 1-4 (320ed27d74) (PROJECTv7.md 7.21)"),
}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def records():
    pats = ["runs/**/run_config.json", "runs/**/run.json",
            "kaggle/**/runs/**/run_config.json", "kaggle/**/runs/**/run.json"]
    seen = set()
    for pat in pats:
        for p in glob.glob(os.path.join(REPO, pat), recursive=True):
            p = os.path.normpath(p)
            if p in seen:
                continue
            seen.add(p)
            try:
                j = json.load(open(p, encoding="utf-8"))
            except Exception:                                            # noqa: BLE001
                continue
            if isinstance(j, dict) and isinstance(j.get("script_sha256_16"), dict):
                yield p, j["script_sha256_16"]


def line_prefix_match(path, want):
    """N such that sha256(first N lines) starts with ``want``, else None."""
    data = open(path, "rb").read()
    lines = data.splitlines(keepends=True)
    h = hashlib.sha256()
    for n, ln in enumerate(lines, 1):
        h.update(ln)
        if h.copy().hexdigest().startswith(want):
            return n, len(lines)
    return None


_INDEX = {}


def copies_of(name):
    if name not in _INDEX:
        hits = []
        for root in [REPO] + [os.path.join(ML, d) for d in os.listdir(ML)
                              if os.path.isdir(os.path.join(ML, d))
                              and os.path.join(ML, d) != REPO]:
            for p in glob.glob(os.path.join(root, "**", name), recursive=True):
                try:
                    hits.append((os.path.normpath(p), sha(p)))
                except OSError:
                    pass
        _INDEX[name] = hits
    return _INDEX[name]


_ARCH = None


def _archives():
    global _ARCH
    if _ARCH is None:
        _ARCH = []
        for root in [REPO] + [os.path.join(ML, d) for d in os.listdir(ML)
                              if os.path.isdir(os.path.join(ML, d))
                              and os.path.join(ML, d) != REPO]:
            for ext in ("*.tar", "*.tar.gz", "*.tgz", "*.zip"):
                _ARCH += glob.glob(os.path.join(root, "**", ext), recursive=True)
    return _ARCH


def archive_hits(name, want):
    """Members named ``name`` inside tar/zip archives whose sha256 starts with ``want``."""
    import tarfile
    import zipfile
    hits = []
    for a in _archives():
        try:
            if a.endswith(".zip"):
                with zipfile.ZipFile(a) as z:
                    for m in z.namelist():
                        if os.path.basename(m) == name and                                 hashlib.sha256(z.read(m)).hexdigest().startswith(want):
                            hits.append(f"{os.path.relpath(a, ML)}!{m}".replace("\\", "/"))
            else:
                with tarfile.open(a) as t:
                    for m in t.getmembers():
                        if m.isfile() and os.path.basename(m.name) == name:
                            b = t.extractfile(m).read()
                            if hashlib.sha256(b).hexdigest().startswith(want):
                                hits.append(f"{os.path.relpath(a, ML)}!{m.name}"
                                            .replace("\\", "/"))
        except Exception:                                                # noqa: BLE001
            continue
    return sorted(hits)


def describe_diff(found, disk):
    """Line-level summary of how the executed copy differs from today's file."""
    import difflib
    if "!" in found:
        return None
    a = open(os.path.join(ML, found), encoding="utf-8", errors="replace").read().splitlines()
    b = open(disk, encoding="utf-8", errors="replace").read().splitlines()
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    ops = [(t, i1, i2, j1, j2) for t, i1, i2, j1, j2 in sm.get_opcodes() if t != "equal"]
    return {"executed_lines": len(a), "disk_lines": len(b),
            "hunks": [{"op": t, "executed": [i1 + 1, i2], "disk": [j1 + 1, j2]}
                      for t, i1, i2, j1, j2 in ops[:20]],
            "n_hunks": len(ops)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    if os.path.exists(a.out) and not a.force:
        raise SystemExit(f"refusing to overwrite {a.out}")
    rows = []
    for p, rec in sorted(records()):
        for script, want in sorted(rec.items()):
            disk = os.path.join(HERE, script)
            dsha = sha(disk) if os.path.isfile(disk) else None
            status = "match" if dsha and dsha.startswith(want) else (
                "differs" if dsha else "missing on disk")
            row = {"run": os.path.relpath(os.path.dirname(p), REPO).replace("\\", "/"),
                   "record": os.path.basename(p), "script": script, "recorded_sha": want,
                   "disk_sha": dsha[:16] if dsha else None, "status": status,
                   "explanation": None, "executed_version_at": []}
            if status != "match":
                pm = line_prefix_match(disk, want) if dsha else None
                if pm:
                    row["explanation"] = (f"VERIFIED append-only: the recorded version is the "
                                          f"first {pm[0]} of today's {pm[1]} lines")
                elif script in KNOWN and not KNOWN[script].startswith("append-only"):
                    row["explanation"] = KNOWN[script] + " (record note; not re-verified here)"
                row["executed_version_at"] = sorted(
                    os.path.relpath(c, ML).replace("\\", "/")
                    for c, s in copies_of(script) if s.startswith(want))
                if not row["executed_version_at"]:
                    row["executed_version_at"] = archive_hits(script, want)
                if not pm and row["executed_version_at"] and dsha:
                    row["diff_vs_disk"] = describe_diff(row["executed_version_at"][0], disk)
            rows.append(row)
    differ = [r for r in rows if r["status"] != "match"]
    unexplained = [r for r in differ if not r["explanation"]]
    out = {"what": "script-provenance audit, report only; nothing edited",
           "n_records": len({(r['run'], r['record']) for r in rows}),
           "n_rows": len(rows), "n_match": len(rows) - len(differ),
           "n_differs": len(differ), "n_unexplained": len(unexplained),
           "unexplained_executed_versions": {
               f"{r['run']} :: {r['script']} ({r['recorded_sha']})":
                   r["executed_version_at"] or "NOT FOUND" for r in unexplained},
           "rows": rows}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
