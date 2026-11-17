r"""
write_record_extract.py -- write public_inputs/record_extract.json from the lab log.
===================================================================================

    python scripts/write_record_extract.py            # (re)write the extract
    python scripts/write_record_extract.py --check    # exit 1 if the committed extract is stale

The extract is everything the paper build needs from ``PROJECTv7.md`` (see
``scripts/record_extract.py``), so that a release without the log rebuilds the paper exactly:

  * ``recorded_hashes`` -- the digests ``scripts/integrity.py`` reads when the log is absent:
    every protected-inventory digest, ``runs/round7/window_prereg.json``'s, and every other
    digest that discovery on the log resolves to a file, each in every flavour the log records
    for that file, so ``integrity.protected_files`` finds the same files over the extract as
    over the log; plus the superseded FreeMatch text's digest, which matches no file
    (``superseded_text_no_file``). ``hash_sources`` says which file each digest belongs to;
  * ``values`` -- each number with the log line it came from and that line's section number;
  * ``checks`` and ``prose_claims`` -- verdicts; for a prose claim also the sha256 of each phrase
    it tests, never the phrase;
  * ``log_sha256``, ``log_lines`` and ``extracted_utc``.

No prose of the log is written. ``extracted_utc`` is kept when nothing else changed, so
rewriting an up-to-date extract changes no byte. Writing the extract also rewrites its entry in
``public_inputs/MANIFEST.json`` (source, licence, contents, bytes, sha256, rows), so the entry
cannot go stale. ``--check`` fails if the extract differs from what the log gives now (ignoring
``extracted_utc``), if the manifest entry's sha256 or bytes differ from the extract's, or if
discovery over the extract finds other files than discovery over the log;
``paper/scripts/make_a2_hashes.py`` runs it whenever the log is present.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import integrity          # noqa: E402
import record_extract as R  # noqa: E402

WINDOW_PREREG = "runs/round7/window_prereg.json"
_HEAD = re.compile(r"^#{1,4} (\d+(?:\.\d+)*[a-z]?)[ .]")
# The superseded FreeMatch pre-registration text (the original budget of 2000): its digest is in
# the log and the paper (Appendix B.2) but matches no file, by design.
SUPERSEDED = "superseded_text_no_file"
_SUPERSEDED_LINE = re.compile(r"hash of the ORIGINAL §7\.11\*\*\s*\|\s*`([0-9a-f]{64})`")
MANIFEST = os.path.join(REPO, "public_inputs", "MANIFEST.json")
MANIFEST_KEY = "record_extract.json"


def superseded_digest(lines: list) -> str:
    hit = [m.group(1) for m in map(_SUPERSEDED_LINE.search, lines) if m]
    assert len(hit) == 1, f"expected one superseded-text digest line in the log, found {len(hit)}"
    assert hit[0].startswith("7356acda"), hit[0]          # the prefix Appendix B.2 prints
    return hit[0]


def discovery_matches() -> tuple:
    """(on the log, over the extract): integrity.protected_files() each way. They must agree."""
    on_log = integrity.protected_files(md_path=R.LOG)
    prev = os.environ.get("PUBLIC_INPUTS")
    os.environ["PUBLIC_INPUTS"] = "1"             # integrity.recorded_hashes() -> the extract
    try:
        on_extract = integrity.protected_files(md_path=R.LOG)
    finally:
        if prev is None:
            os.environ.pop("PUBLIC_INPUTS")
        else:
            os.environ["PUBLIC_INPUTS"] = prev
    return on_log, on_extract


def manifest_entry(extract_text: str, d: dict) -> dict:
    return {"source": "the author's lab log PROJECTv7.md (not distributed), via "
                      "scripts/write_record_extract.py",
            "licence": "this work (GPL-3.0)",
            "contents": (f"{len(d['hash_sources'])} digest rows ({len(d['recorded_hashes'])} distinct "
                         f"recorded_hashes: the protected inventory, runs/round7/window_prereg.json, "
                         f"every other file discovery on the log resolves, and one superseded text "
                         f"with no file); {len(d['values'])} values with their log line and section "
                         f"number; {len(d['checks'])} check verdicts; {len(d['prose_claims'])} "
                         f"prose-claim verdicts with the sha256 of each phrase; the log's sha256 and "
                         f"line count; the extraction time"),
            "bytes": len(extract_text.encode("utf-8")),
            "sha256": hashlib.sha256(extract_text.encode("utf-8")).hexdigest(),
            "rows": len(d["hash_sources"])}


def write_manifest_entry(entry: dict) -> bool:
    """Rewrite the record_extract.json entry of public_inputs/MANIFEST.json (indent 1, no final
    newline, as export_public_inputs.py writes it). Returns True if the file changed."""
    raw = open(MANIFEST, encoding="utf-8").read()
    m = json.loads(raw)
    assert json.dumps(m, indent=1, ensure_ascii=False) == raw, "MANIFEST.json format changed"
    m[MANIFEST_KEY] = entry
    out = json.dumps(m, indent=1, ensure_ascii=False)
    if out != raw:
        open(MANIFEST, "w", encoding="utf-8", newline="\n").write(out)
        return True
    return False


def section_of(lines: list, n: int) -> str:
    """The number of the closest numbered heading at or above log line n ('' if none)."""
    for i in range(n - 1, -1, -1):
        m = _HEAD.match(lines[i])
        if m:
            return m.group(1)
    return ""


def build() -> dict:
    if not R.log_present():
        raise SystemExit(f"{R.LOG_NAME} is not present; the extract can only be made from the log")
    raw = R.log_raw()
    lines = raw.split("\n")
    in_log = integrity.recorded_hashes(R.LOG)
    inv = json.load(open(os.path.join(REPO, "runs", "protected_artifacts.json"),
                         encoding="utf-8"))["artifacts"]
    paths = [a["path"] for a in inv]
    for a in inv:
        assert a["sha256"] in in_log, f"inventory digest of {a['path']} is not in the log"
    if WINDOW_PREREG not in paths:
        paths.append(WINDOW_PREREG)
    # every file that discovery on the log resolves (integrity.protected_files), so discovery
    # over the extract finds the same files; the inventory adds the files outside its scan dirs
    for rel, _f, _s in integrity.protected_files(md_path=R.LOG):
        if rel not in paths:
            paths.append(rel)
    sources = []
    for rel in paths:
        p = os.path.join(REPO, *rel.split("/"))
        for flav, sha in (("self_certifying", integrity.self_certifying_sha(p)),
                          ("whole_file", integrity.whole_file_sha(p))):
            if sha and sha in in_log:
                sources.append({"path": rel, "flavour": flav, "sha256": sha})
    assert {s["path"] for s in sources} == set(paths), "a protected file has no digest in the log"
    sources.append({"path": None, "flavour": SUPERSEDED, "sha256": superseded_digest(lines)})

    values = {}
    for k in R.VALUES:
        v, n = R.eval_value(k)
        values[k] = {"value": v, "line": n, "section": section_of(lines, n)}
    checks = {k: {"verdict": R.eval_check(k)} for k in R.CHECKS}
    prose = {cid: {"verdict": R.eval_prose(cid), "form": f,
                   "phrases_sha256": [R.sha256_text(p) for p in ps]}
             for cid, (f, ps) in R.PROSE.items()}
    return {
        "what": ("Everything the paper build reads from the author's lab log PROJECTv7.md, which is "
                 "not distributed: digests, numbers with their log line and section number, and "
                 "verdicts. No text of the log. Written by scripts/write_record_extract.py; read "
                 "through scripts/record_extract.py and scripts/integrity.py."),
        "log": R.LOG_NAME,
        "log_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "log_lines": len(lines),
        "extracted_utc": None,
        "recorded_hashes": sorted({s["sha256"] for s in sources}),
        "hash_sources": sources,
        "values": values,
        "checks": checks,
        "prose_claims": prose,
    }


def _comparable(d: dict) -> dict:
    return {k: v for k, v in d.items() if k != "extracted_utc"}


def dumps(d: dict) -> str:
    return json.dumps(d, indent=1, ensure_ascii=False) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="fail if the committed extract is stale")
    a = ap.parse_args(argv)
    new = build()
    old = None
    if os.path.isfile(R.EXTRACT):
        try:
            old = json.load(open(R.EXTRACT, encoding="utf-8"))
        except ValueError:
            old = None
    same = old is not None and _comparable(old) == _comparable(new)
    if a.check:
        if not same:
            diff = sorted(k for k in set(new) | set(old or {}) if k != "extracted_utc"
                          and (old or {}).get(k) != new.get(k))
            print(f"[record-extract] STALE: {os.path.relpath(R.EXTRACT, REPO)} differs from what "
                  f"{R.LOG_NAME} gives now in {diff}. Run python scripts/write_record_extract.py")
            return 1
        text = open(R.EXTRACT, encoding="utf-8").read()
        want = manifest_entry(text, old)
        have = json.load(open(MANIFEST, encoding="utf-8")).get(MANIFEST_KEY, {})
        if (have.get("sha256"), have.get("bytes")) != (want["sha256"], want["bytes"]):
            print(f"[record-extract] STALE: public_inputs/MANIFEST.json's {MANIFEST_KEY} entry "
                  f"(sha256 {str(have.get('sha256'))[:16]}..., {have.get('bytes')} bytes) does not "
                  f"match the extract ({want['sha256'][:16]}..., {want['bytes']} bytes). Run python "
                  f"scripts/write_record_extract.py")
            return 1
        on_log, on_extract = discovery_matches()
        if on_log != on_extract:
            print(f"[record-extract] DISCOVERY DIFFERS: on the log {len(on_log)} files, over the "
                  f"extract {len(on_extract)}; only on the log: "
                  f"{sorted(set(on_log) - set(on_extract))[:5]}")
            return 1
        print(f"[record-extract] up to date with {R.LOG_NAME} (sha256 {new['log_sha256'][:16]}...); "
              f"manifest entry matches; discovery finds the same {len(on_log)} files over the "
              f"extract as over the log")
        return 0
    new["extracted_utc"] = (old["extracted_utc"] if same else
                            datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    s = dumps(new)
    if old is None or open(R.EXTRACT, encoding="utf-8").read() != s:
        open(R.EXTRACT, "w", encoding="utf-8", newline="\n").write(s)
    changed = write_manifest_entry(manifest_entry(s, new))
    print(f"[record-extract] {os.path.relpath(R.EXTRACT, REPO)}: {len(new['recorded_hashes'])} digests "
          f"({len(new['hash_sources'])} rows), {len(new['values'])} values, "
          f"{len(new['checks'])} checks, {len(new['prose_claims'])} prose claims; log sha256 "
          f"{new['log_sha256'][:16]}...; MANIFEST.json entry {'rewritten' if changed else 'unchanged'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
