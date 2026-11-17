"""
integrity.py
============
The protected set: every artifact whose hash PROJECT.md records, and the check that it
still hashes to what PROJECT.md says.

WHY THIS EXISTS. Running the Task-A detector for round 6 silently overwrote
``runs/coral_r4_budget_analysis.json`` with BNM numbers, and the repair re-wrote
``runs/coral_r4_budget.json`` with a fresh timestamp, moving its self-certifying hash away
from the ``65c20810...`` that PROJECT.md 7.14 records. Both were recoverable only because a
backup happened to exist and the detector happened to be deterministic. **Neither is a
control.** Folds 1-4 run the same scripts four more times.

Two mechanisms, and they are different in kind:

  * :func:`protect` sets the protected files read-only, so an overwrite FAILS rather than
    succeeding quietly -- the same principle as ``stats_only_row`` not taking the scorer.
  * :func:`verify` re-derives each hash and compares it to PROJECT.md, so a file that was
    changed while writable is still caught. Analysis scripts call it at START and END: the
    start call refuses to analyse against a corrupted record, and the end call catches a
    script that corrupted one while running.

Two hash FLAVOURS are handled, because the project uses both:

  * WHOLE-FILE  -- sha256 of the bytes (e.g. ``runs/bnm_r6_prereg.json``).
  * SELF-CERTIFYING -- sha256 of the JSON body with the
    ``sha256_of_body_without_this_field`` key removed (e.g. ``runs/coral_r4_budget.json``).
    A whole-file hash of these does NOT appear in PROJECT.md, so checking the wrong flavour
    would report a false failure.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(_HERE)

# THE RECORD IS DISCOVERED, NOT NAMED -- and that is a fix, not a preference.
#
# This constant used to read ``PROJECTv6.md``. When the record was renamed to
# ``PROJECTv7.md`` the constant went on naming a file that no longer existed, so every
# caller of ``verify()`` raised ``FileNotFoundError`` at start-up and **the guard stopped
# running**. It failed in the worst available way: silently as far as anyone watching was
# concerned, because the two scripts that call it do so as a start-of-run check nobody
# reads when it passes, and because ``test_bnm_round6.T22`` treats an absent record as
# "this is the training venue" and SKIPS. A dead guard and a guard on a machine that
# holds no artifacts looked identical.
#
# So the filename is now derived from what is on disk. A rename cannot disable the guard
# again, and an ambiguous record is refused rather than guessed at (see ``verify``).
_RECORD_RE = re.compile(r"^PROJECT(?:v(\d+))?\.md$")


def project_record_candidates(repo=REPO):
    """Every plausible project record in ``repo``, highest version first.

    Matches ``PROJECT.md`` and ``PROJECTvN.md`` exactly -- not ``PROJECT_STATE v2.md``
    and not a draft -- so a stray file cannot become the record by accident.
    """
    found = []
    try:
        names = os.listdir(repo)
    except OSError:
        return []
    for name in names:
        m = _RECORD_RE.match(name)
        if m:
            found.append((int(m.group(1)) if m.group(1) else -1, name))
    return [os.path.join(repo, n) for _v, n in sorted(found, reverse=True)]


PROJECT_MD_CANDIDATES = project_record_candidates()
# Absent-but-conventional when nothing is found: callers already treat a missing record
# as "not the analysis machine", and that behaviour is deliberate (T22's skip).
PROJECT_MD = (PROJECT_MD_CANDIDATES[0] if PROJECT_MD_CANDIDATES
              else os.path.join(REPO, "PROJECT.md"))


def whole_file_sha(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def self_certifying_sha(path):
    """sha256 of the body with the self-hash field removed, or ``None`` if not that shape."""
    try:
        d = json.load(open(path, encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(d, dict) or "sha256_of_body_without_this_field" not in d:
        return None
    body = {k: v for k, v in d.items() if k != "sha256_of_body_without_this_field"}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _record(md_path):
    """Resolve the record at CALL time.

    ``md_path=PROJECT_MD`` as a default argument would bind at import time, so a later
    reassignment of the module constant would be ignored while still looking honoured --
    the same shape of bug as the stale filename this module just had. ``None`` means
    "whatever the record is now"; anything else is an explicit caller choice.
    """
    return PROJECT_MD if md_path is None else md_path


# Where the project record is not distributed, its hash table is: public_inputs/record_extract.json
# (written by scripts/export_public_inputs.py) holds the record's digests of the files under
# runs/. PUBLIC_INPUTS=1 reads it even when the record is present -- the equivalence check.
RECORD_EXTRACT = os.path.join(REPO, "public_inputs", "record_extract.json")


def recorded_hashes(md_path=None) -> set:
    """Every 64-hex string the project record contains (or its extracted hash table)."""
    import re
    path = _record(md_path)
    if os.environ.get("PUBLIC_INPUTS") == "1" or not os.path.isfile(path):
        return set(json.load(open(RECORD_EXTRACT, encoding="utf-8"))["recorded_hashes"])
    txt = open(path, encoding="utf-8", errors="replace").read()
    return set(re.findall(r"\b[0-9a-f]{64}\b", txt))


#: where --build looks for hashed artifacts. ``data/splits`` added 2026-10-13 (PROJECTv7.md
#: 7.37): the diluted pool ``data/splits/dilute10_pool.json`` is protected, and a rebuild that
#: scanned ``runs/`` alone would have dropped it from the inventory.
DISCOVERY_DIRS = ("runs", os.path.join("data", "splits"))


def protected_files(md_path=None, scan_dirs=DISCOVERY_DIRS):
    """Files under ``scan_dirs`` whose hash (either flavour) appears in PROJECT.md.

    Returns ``[(relpath, flavour, sha), ...]`` sorted by path.
    """
    want = recorded_hashes(md_path)
    found = []
    for root, _dirs, files in (w for d in scan_dirs
                               for w in os.walk(os.path.join(REPO, d))):
        for name in files:
            p = os.path.join(root, name)
            rel = os.path.relpath(p, REPO).replace("\\", "/")
            try:
                if os.path.getsize(p) > 64 << 20:      # skip weights/tars
                    continue
            except OSError:
                continue
            sc = self_certifying_sha(p)
            if sc and sc in want:
                found.append((rel, "self_certifying", sc))
                continue
            wf = whole_file_sha(p)
            if wf in want:
                found.append((rel, "whole_file", wf))
    return sorted(found)


def verify(md_path=None, *, where="", quiet=False) -> dict:
    """Re-derive every protected file's hash and confirm PROJECT.md still records it.

    Raises SystemExit on any mismatch -- an analysis run against a corrupted record is not
    an analysis. Returns a summary dict.

    It also refuses when the record is AMBIGUOUS. Two files matching ``PROJECT*.md`` means
    nobody can say which one the hashes below were checked against, and guessing is how
    this guard died the first time.
    """
    if md_path is None and len(PROJECT_MD_CANDIDATES) > 1:
        raise SystemExit(
            "INTEGRITY CHECK REFUSED: more than one project record is present, so it is "
            "undecidable which one records the hashes:\n"
            + "".join(f"  {os.path.relpath(p, REPO)}\n"
                      for p in PROJECT_MD_CANDIDATES)
            + "  Remove or rename the superseded one, or pass md_path explicitly.")
    md_path = _record(md_path)
    want = recorded_hashes(md_path)
    inv = _load_inventory()
    bad, missing, ok = [], [], []
    for rel, flavour, sha in inv:
        p = os.path.join(REPO, rel)
        if not os.path.isfile(p):
            missing.append(rel)
            continue
        got = self_certifying_sha(p) if flavour == "self_certifying" else whole_file_sha(p)
        if got != sha:
            bad.append((rel, sha, got))
        elif got not in want:
            bad.append((rel, sha, got + " (no longer in PROJECT.md)"))
        else:
            ok.append(rel)

    if bad or missing:
        lines = [f"INTEGRITY CHECK FAILED{(' ' + where) if where else ''}."]
        for rel, exp, got in bad:
            lines += [f"  CHANGED  {rel}", f"    recorded {exp}", f"    now      {got}"]
        for rel in missing:
            lines.append(f"  ABSENT   {rel}")
        lines.append("  A hashed artifact no longer matches what PROJECT.md records.")
        lines.append("  Do not proceed: restore it before analysing anything.")
        raise SystemExit("\n".join(lines))

    if not quiet:
        print(f"[integrity] {len(ok)} protected artifact(s) verified"
              + (f" {where}" if where else ""))
    return {"verified": ok, "n": len(ok)}


INVENTORY = os.path.join(REPO, "runs", "protected_artifacts.json")


def _load_inventory():
    if not os.path.isfile(INVENTORY):
        raise SystemExit(
            f"{os.path.relpath(INVENTORY, REPO)} is absent. Build it first:\n"
            "    python scripts/integrity.py --build")
    d = json.load(open(INVENTORY, encoding="utf-8"))
    return [(r["path"], r["flavour"], r["sha256"]) for r in d["artifacts"]]


def protect(readonly=True, *, quiet=False):
    """chmod 444 (or back to 644) every protected file."""
    n = 0
    for rel, _f, _s in _load_inventory():
        p = os.path.join(REPO, rel)
        if not os.path.isfile(p):
            continue
        mode = (stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH) if readonly else 0o644
        os.chmod(p, mode)
        n += 1
    if not quiet:
        print(f"[integrity] {'protected' if readonly else 'unprotected'} {n} file(s)")
    return n


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--build", action="store_true",
                   help="scan runs/ for files PROJECT.md hashes, write the inventory")
    g.add_argument("--verify", action="store_true")
    g.add_argument("--protect", action="store_true")
    g.add_argument("--unprotect", action="store_true")
    ap.add_argument("--out", default=None,
                    help="--build only: write the inventory here instead of the live one")
    a = ap.parse_args(argv)

    if a.build:
        found = protected_files()
        dst = os.path.abspath(a.out) if a.out else INVENTORY
        with open(dst, "w", encoding="utf-8") as fh:
            json.dump({"note": ("every artifact whose hash PROJECT.md records. Built by "
                                "scripts/integrity.py --build; verified at the start and "
                                "end of every analysis script."),
                       "n": len(found),
                       "artifacts": [{"path": p, "flavour": f, "sha256": s}
                                     for p, f, s in found]},
                      fh, indent=1)
        print(f"[integrity] inventory: {len(found)} protected artifact(s)"
              f" -> {os.path.relpath(dst, REPO) if dst.startswith(REPO) else dst}")
        for p, f, s in found:
            print(f"    {f:<16} {s[:16]}  {p}")
        return 0
    if a.verify:
        verify()
        return 0
    protect(readonly=a.protect)
    return 0


if __name__ == "__main__":
    sys.exit(main())
