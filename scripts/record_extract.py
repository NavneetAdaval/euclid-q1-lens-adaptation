r"""
record_extract.py -- what the paper build takes from the author's lab log, read from the log or
from its extract.
===============================================================================================

The lab log (``PROJECTv7.md``) is not distributed. The build needs four kinds of thing from it:

  * **digests** -- the hash table ``scripts/integrity.py`` checks protected files against;
  * **numbers** -- values the generators print as macros or the cross-check compares against
    (``VALUES`` below), each found by a regular expression on one form of the log's text;
  * **checks** -- yes/no facts about the log that the generators assert (``CHECKS``);
  * **prose claims** -- eight procedural claims of ``make_m_prose.py`` that test whether given
    phrases occur in the log (``PROSE``).

``scripts/write_record_extract.py`` evaluates all of them on the log and writes
``public_inputs/record_extract.json``: digests, numbers with the log line they came from, check
and claim verdicts, the sha256 of each phrase (never the phrase), the log's own sha256 and the
extraction time. No text of the log goes into the extract.

With the log present every accessor below reads the log, exactly as the generators did before;
with it absent they read the extract; with neither present they raise. The registry here is
the single definition both paths use, and ``write_record_extract.py --check`` (run by
``make_a2_hashes.py`` in every build that has the log) fails if the committed extract differs
from what the log gives now.
"""
from __future__ import annotations

import hashlib
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
LOG_NAME = "PROJECTv7.md"
LOG = os.path.join(REPO, LOG_NAME)
EXTRACT = os.path.join(REPO, "public_inputs", "record_extract.json")

# ---------------------------------------------------------------------------- text forms
# Each form is the exact transformation a generator applied to the log before matching; the
# steps run in order, and every step keeps track of which log character each output character
# came from, so a match can be traced back to its log line.
FORMS = {
    "raw": [],
    # crosscheck.py: re.sub(r"\s+", " ", proj.replace("**", ""))
    "flat": [(r"\*\*", ""), (r"\s+", " ")],
    # make_m_prose.py rewrite v2 / v4 claims
    "prose_v2": [(r"\n> ?", "\n"), (r"\*\*", ""), (r"\s+", " ")],
    # make_m_prose.py rewrite v3 claims (backticks dropped as well)
    "prose_v3": [(r"\n> ?", "\n"), (r"\*\*", ""), (r"`", ""), (r"\s+", " ")],
}

# ---------------------------------------------------------------------------- numbers
# key -> (form, pattern, group). form "line:N" applies the pattern to log line N (1-based);
# "unique_line" requires exactly one matching line and returns that line's number.
VALUES = {
    # crosscheck.py, rewrite-v2 block
    "cc.defl_pool": ("raw", r"deflector_cutouts_clean\.npz\s+([\d,]+) clean deflector galaxies", 1),
    "cc.defl_arc": ("flat", r"~([\d,]+) of them can produce an arc with Einstein radius θ_E ≥ 0\.75", 1),
    "cc.defl_reuse": ("flat", r"reused for up to (\d+) different lenses", 1),
    "cc.opt_lr": ("flat", r"learning_rate=([\de.-]+[\d])", 1),
    "cc.opt_lr_decay": ("flat", r"lr_decay=([\d.]+)", 1),
    "cc.opt_wd": ("flat", r"weight_decay=([\d.]+)", 1),
    "cc.fm_lr": ("flat", r"FreeMatch x nb3 kept round-1's lr ([\de-]+\d) / wd ([\de-]+\d)", 1),
    "cc.fm_wd": ("flat", r"FreeMatch x nb3 kept round-1's lr ([\de-]+\d) / wd ([\de-]+\d)", 2),
    "cc.coral_argmax": ("flat", r"Explicitly NOT the argmax \((\d+)\)", 1),
    "cc.fm_budget_old": ("flat", r"old value \| (\d+) steps", 1),
    "cc.fm_budget_new": ("flat", r"new value \| (\d+) steps", 1),
    # crosscheck.py, review-macro block (fixed lines)
    "cc.early_noda": ("line:494", r"~(0\.\d+) no-DA", 1),
    "cc.early_iw": ("line:494", r"IW-CDAN \(`--iw on`\) to ~(0\.\d+)", 1),
    "cc.early_dann": ("line:4143", r"--iw off`\) to ~(0\.\d+)", 1),
    "cc.early_probe": ("line:3181", r"same pool \| ~(0\.\d+)", 1),
    # make_review_macros.py project_early()
    "rv.early_iw": ("line:494", r"IW-CDAN \(`--iw on`\) to ~(0\.\d+)\*\*, from a \*\*~(0\.\d+) no-DA", 1),
    "rv.early_noda": ("line:494", r"IW-CDAN \(`--iw on`\) to ~(0\.\d+)\*\*, from a \*\*~(0\.\d+) no-DA", 2),
    "rv.early_dann": ("line:4143", r"plain DANN \(`--iw off`\) to ~(0\.\d+)\*\*", 1),
    "rv.early_probe": ("line:3181", r"frozen-feature linear probe\*\*, same pool \| ~(0\.\d+) \|", 1),
    # make_rewrite_v2_macros.py deflectors()
    "rw.defl_pool": ("unique_line", r"\(([\d,]+) clean massive ellipticals; ~([\d,]+) of them can produce", 1),
    "rw.defl_arc": ("unique_line", r"\(([\d,]+) clean massive ellipticals; ~([\d,]+) of them can produce", 2),
    "rw.defl_reuse": ("unique_line_reuse", r"reused for up to (\d+) different lenses", 1),
}

# ---------------------------------------------------------------------------- checks
# key -> function(lines) -> bool: assertions the generators make about the log.
def _rw_arc_subset(L):
    hit = [(i, l) for i, l in enumerate(L, 1) if re.search(VALUES["rw.defl_pool"][1], l)]
    i, line = hit[0]
    return "0.75" in L[i] and "Einstein" in line


def _rw_budget_list(_L):
    text = form("flat")[0]
    return ("Selection rule R1 (last checkpoint, step 8000), pre-registered and FROZEN before "
            "results" in text and "with a ceiling declared before the curve exists" in text)


CHECKS = {
    "rv.early_table_header": lambda L: "| instrument | no-DA | plain DANN | IW-CDAN |" in L[3177],
    "rw.arc_subset_theta_e_075": _rw_arc_subset,
    "rw.budget_list_lines": _rw_budget_list,
}

# ---------------------------------------------------------------------------- prose claims
# claim id -> (form, phrases). The claim's log part holds when every phrase occurs in the form.
PROSE = {
    "RW2-1": ("prose_v2", ["30,000 simulated lens images"]),
    "RW2-2": ("prose_v2", ["THE BUDGET — 4000, DECIDED"]),
    "RW2-7": ("prose_v2", ["old value | 2000 steps", "new value | 1500 steps",
                           "The argmax sits at 1250 and 1500, both 0.5204"]),
    "RW2-8": ("prose_v2", ["freematch_fold0_standard/` (TPR 0.357) and `freematch_fold0_floor/` (TPR 0.174)"]),
    "RW3-2": ("prose_v3", ["γ = 0.00787 from a metric-blind fold-0 sweep",
                           "measures the BNM term's gradient norm against source CE at step 0",
                           "GATES Stage 2's γ selection",
                           "off its 9.77 init"]),
    "RW3-3": ("prose_v3", ["generation runs in WSL2 Ubuntu, conda env lenssim (Python 3.11, galsim + "
                           "lenstronomy==1.14.1 pinned + astropy)"]),
    "RW3-4": ("prose_v3", ["Selection rule R1 (last checkpoint, step 8000), pre-registered and FROZEN before "
                           "results", "with a ceiling declared before the curve exists"]),
    "RW4-5": ("prose_v2", ["SCREEN 1 — inherited hyperparameters (head_lr 1e-3, backbone_lr 1e-4, wd 1e-4), fold 0",
                           "SCREEN 2 — Zoobot published defaults",
                           "screen 1 (inherited: head_lr 1e-3, backbone_lr 1e-4, wd 1e-4) gave nb3 0.2143",
                           "screen 2 (Zoobot defaults) gave 0.1939",
                           "It was not chosen because it looked better — it demonstrably did not."]),
}


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------- log access
_CACHE: dict = {}


def log_present() -> bool:
    return os.path.isfile(LOG)


def _require_source():
    if not log_present() and not os.path.isfile(EXTRACT):
        raise SystemExit(f"neither the lab log ({LOG_NAME}) nor its extract "
                         f"({os.path.relpath(EXTRACT, REPO)}) is present; the build cannot run")


def log_raw() -> str:
    if "raw" not in _CACHE:
        _CACHE["raw"] = open(LOG, encoding="utf-8").read()
    return _CACHE["raw"]


def log_lines() -> list:
    return log_raw().split("\n")


def form(name: str):
    """(text, source_index) for a text form of the log; source_index[k] is the log offset
    that output character k came from."""
    key = ("form", name)
    if key not in _CACHE:
        raw = log_raw()
        chars, idx = raw, list(range(len(raw)))
        for pat, rep in FORMS[name]:
            out, oidx, pos = [], [], 0
            for m in re.finditer(pat, chars):
                out.append(chars[pos:m.start()]); oidx.extend(idx[pos:m.start()])
                out.append(rep); oidx.extend([idx[m.start()]] * len(rep))
                pos = m.end()
            out.append(chars[pos:]); oidx.extend(idx[pos:])
            chars, idx = "".join(out), oidx
        _CACHE[key] = (chars, idx)
    return _CACHE[key]


def _line_of(offset: int) -> int:
    return log_raw().count("\n", 0, offset) + 1


def eval_value(key: str):
    """(value string, log line) for a VALUES key, computed on the log."""
    kind, pat, g = VALUES[key]
    L = log_lines()
    if kind.startswith("line:"):
        n = int(kind.split(":")[1])
        m = re.search(pat, L[n - 1])
        assert m, f"{LOG_NAME}:{n} no longer matches {key}"
        return m.group(g), n
    if kind == "unique_line":
        hit = [(i, l) for i, l in enumerate(L, 1) if re.search(pat, l)]
        assert len(hit) == 1, (key, hit)
        return re.search(pat, hit[0][1]).group(g), hit[0][0]
    if kind == "unique_line_reuse":
        hit = [(i, l) for i, l in enumerate(L, 1) if "reused for up to" in l and "lenses" in l]
        assert len(hit) == 1, (key, hit)
        return re.search(pat, hit[0][1]).group(g), hit[0][0]
    text, idx = form(kind)
    m = re.search(pat, text)
    assert m, f"{LOG_NAME} no longer matches {key}"
    return m.group(g), _line_of(idx[m.start(g)])


def eval_check(key: str) -> bool:
    return bool(CHECKS[key](log_lines()))


def eval_prose(cid: str) -> bool:
    name, phrases = PROSE[cid]
    text = form(name)[0]
    return all(p in text for p in phrases)


def extract() -> dict:
    if "extract" not in _CACHE:
        _CACHE["extract"] = json.load(open(EXTRACT, encoding="utf-8"))
    return _CACHE["extract"]


# ---------------------------------------------------------------------------- the accessors
def value(key: str):
    """(value string, log line): from the log if present, else from the extract."""
    _require_source()
    if log_present():
        return eval_value(key)
    e = extract()["values"][key]
    return e["value"], e["line"]


def check(key: str) -> bool:
    _require_source()
    if log_present():
        return eval_check(key)
    return bool(extract()["checks"][key]["verdict"])


def prose(cid: str) -> bool:
    """The log part of a prose claim. Without the log, the extract's verdict, after checking
    that the phrases tested now are the ones whose hashes were recorded."""
    _require_source()
    if log_present():
        return eval_prose(cid)
    e = extract()
    rec = e["prose_claims"][cid]
    now = [sha256_text(p) for p in PROSE[cid][1]]
    assert now == rec["phrases_sha256"], (
        f"{cid}: the phrases this build tests differ from those recorded at extraction")
    print(f"  {cid}: verified against the author's log (sha256 {e['log_sha256']}) at extraction "
          f"({e['extracted_utc']})")
    return bool(rec["verdict"])


def source_label() -> str:
    """How a generator should name its source in printed provenance."""
    return LOG_NAME
