r"""
make_archive.py -- build the arXiv/journal source package behind a submission gate.
====================================================================================

    python paper/scripts/make_archive.py            # gate, then write <repo>/paper.zip
    python paper/scripts/make_archive.py --draft    # write <repo>/paper_DRAFT.zip anyway
    python paper/scripts/make_archive.py --list     # report contents and gate, write nothing

``paper.zip`` is what goes to arXiv, which publishes uploaded sources next to the PDF. The
principle: working state must FAIL LOUDLY IN A CHECK, never ship or print.

**THE SUBMISSION GATE.** By default the script refuses to write ``paper.zip`` while any
of the conditions in :func:`gate` holds, and lists every violation rather than the first:

  * a cited entry has an ``annote`` carrying an unresolved marker (``ANNOTE_MARKERS``);
  * a cited entry's ``citation_register.md`` row has evidence ``none``, an empty claims
    column, or a per-site mark that is still open (``REGISTER_OPEN_MARKS``);
  * a cited entry is missing ``author`` or ``title``, or has neither a year nor an eprint;
  * a ``\todo`` or ``\PLACEHOLDER`` remains in the source (outside comments and the macro
    definitions themselves);
  * a ``\section`` has no body at all (a stub whose only content was a planning comment).

When it refuses, it also deletes any existing ``paper.zip``, so a stale, ungated package
cannot be uploaded by mistake. ``--draft`` builds anyway, prints the same list as a
warning, and names the output ``paper_DRAFT.zip``.

**WHAT GOES IN: an allow-list derived from the build.** From ``main.tex`` the script
follows every ``\input``/``\include`` (resolved against the document root, as pdflatex
does), ``\includegraphics`` (honouring ``\graphicspath``), ``\bibliography`` and
``\bibliographystyle``, and ``\documentclass``/``\usepackage`` where the ``.cls``/``.sty``
is vendored. Nothing is copied because it sits in a directory; an unresolved reference is
an error.

**WHAT IS REWRITTEN IN THE ARCHIVED COPY ONLY** (the working files keep everything):

  * ``.tex`` -- comments are stripped. Text after an unescaped ``%`` is removed, but the
    ``%`` itself is kept when content precedes it on the line, because it suppresses the
    end-of-line space. Comment-only lines are deleted. ``\%`` is literal. The contents of
    ``verbatim``/``lstlisting`` environments, ``\verb``, ``\url{}`` and the URL argument of
    ``\href{}`` are left untouched.
  * ``.bib`` -- every ``annote`` field and every line outside an entry (comments) is
    removed. ``annote`` holds working notes; plainnat does not print it, and the source
    package should not carry it either.

**CHECKS ON THE WRITTEN ARCHIVE.** Zip integrity; ``main.tex`` at the root; no backslash
in a path; no ``.md``, ``.json``, ``.py``, ``.sha256`` or ``refs_removed*`` file; every file
referenced by the build; every extension an allowed LaTeX-source type; no archived ``.tex``
line carries an unescaped ``%`` followed by text; no archived ``.bib`` carries an
``annote`` field or a comment; the bib's entries and every other field survive
unchanged. Finally the archive is unzipped into a clean temporary directory and the whole
reference graph is re-resolved there. No TeX distribution is assumed.
"""
from __future__ import annotations

import argparse
import fnmatch
import os
import re
import shutil
import sys
import tempfile
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
REPO = os.path.dirname(PAPER)
OUT = os.path.join(REPO, "paper.zip")
OUT_DRAFT = os.path.join(REPO, "paper_DRAFT.zip")
REGISTER = os.path.join(PAPER, "citation_register.md")
MAIN = "main.tex"

ALLOWED_EXT = {".tex", ".bib", ".bst", ".cls", ".sty", ".pdf", ".png", ".jpg", ".jpeg",
               ".eps"}
GRAPHIC_EXT = (".pdf", ".png", ".jpg", ".jpeg", ".eps")
FORBIDDEN_GLOBS = ("*.md", "*.json", "*.py", "*.sha256", "refs_removed*")

# Unresolved-work markers in a cited entry's annote. Whole-word, case-insensitive, so
# "CHECK VOLUME" matches and "Checked against the full text" does not. The brief's five,
# plus three that express the same thing in notes the five would miss:
#   VERIFY          -- zoobot: "VERIFY before submission"
#   NOT YET CHECKED -- belfiore2024: "NOT YET CHECKED"
#   COMPLETE THE    -- discoveryengineA/E, lines2025discoveryengineC: "complete the list"
ANNOTE_MARKERS = ("UNVERIFIED", "INCOMPLETE", "TODO", "CHECK", "NEEDED",
                  "VERIFY", "NOT YET CHECKED", "COMPLETE THE")

# Open per-site marks in the register's "What we claim with it" column. The brief's gate
# reads only the evidence and claims columns, which misses claims still open at a site
# of an otherwise-summarised entry (Musgrave's correlation claim, Belfiore's dropout
# rate). "title-level" and "second-hand" are deliberate states, not open work.
REGISTER_OPEN_MARKS = ("body needed", "not covered", "identifier-level",
                       "summarise before use")

VERBATIM_ENVS = ("verbatim", "verbatim*", "Verbatim", "lstlisting", "minted", "comment")


# =====================================================================================
#  LaTeX comment handling
# =====================================================================================
def _protected_spans(line: str):
    """Character spans in which '%' is literal: \\verb, \\url{}, \\href's URL argument."""
    spans = []
    for m in re.finditer(r"\\verb\*?(.)", line):
        d = m.group(1)
        end = line.find(d, m.end())
        spans.append((m.start(), len(line) if end < 0 else end + 1))
    for m in re.finditer(r"\\(url|href)\s*\{", line):
        depth, i = 1, m.end()
        while i < len(line) and depth:
            depth += {"{": 1, "}": -1}.get(line[i], 0)
            i += 1
        spans.append((m.start(), i))
    return spans


def first_comment(line: str) -> int | None:
    r"""Index of the first unescaped, unprotected '%', or None. '\\' is an escape pair."""
    spans = _protected_spans(line)
    i = 0
    while i < len(line):
        hit = next((e for s, e in spans if s <= i < e), None)
        if hit is not None:
            i = hit
            continue
        if line[i] == "\\":
            i += 2
            continue
        if line[i] == "%":
            return i
        i += 1
    return None


def _verbatim_toggle(line: str, inside: str | None) -> str | None:
    if inside is None:
        m = re.search(r"\\begin\{(" + "|".join(map(re.escape, VERBATIM_ENVS)) + r")\}",
                      line)
        return m.group(1) if m else None
    return None if ("\\end{" + inside + "}") in line else inside


def strip_tex_comments(text: str) -> str:
    """The archive-copy transform. See the module docstring for the rules."""
    out, inside = [], None
    for line in text.split("\n"):
        if inside is not None:
            out.append(line)
            inside = _verbatim_toggle(line, inside)
            continue
        p = first_comment(line)
        if p is None:
            out.append(line)
        elif line[:p].strip() == "":
            pass                                   # comment-only line: delete it
        else:
            out.append(line[:p] + "%")             # keep the whitespace-suppressing %
        inside = _verbatim_toggle(line, None)
    return "\n".join(out)


def comment_text_lines(text: str):
    """Lines (1-based) where an unescaped % is followed by text -- must be none."""
    bad, inside = [], None
    for n, line in enumerate(text.split("\n"), 1):
        if inside is not None:
            inside = _verbatim_toggle(line, inside)
            continue
        p = first_comment(line)
        if p is not None and line[p + 1:].strip():
            bad.append(n)
        inside = _verbatim_toggle(line, None)
    return bad


def strip_comments_for_scan(text: str) -> str:
    """Remove comments entirely, for reading the build graph and the gate."""
    return "\n".join(line if first_comment(line) is None else line[:first_comment(line)]
                     for line in text.split("\n"))


# =====================================================================================
#  BibTeX handling
# =====================================================================================
def _match_brace(s: str, i: int) -> int:
    """s[i] == '{'; return the index just past its matching '}'."""
    depth = 0
    while i < len(s):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    raise ValueError("unbalanced braces in .bib")


def bib_entries(text: str):
    """[(type, key, start, end, body)] for every @entry, brace-matched."""
    out = []
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", text):
        end = _match_brace(text, text.index("{", m.start()))
        out.append((m.group(1).lower(), m.group(2), m.start(), end, text[m.end():end - 1]))
    return out


def bib_fields(body: str) -> dict:
    """field -> raw value (braces stripped) for one entry body."""
    fields, i = {}, 0
    for m in re.finditer(r"(\w+)\s*=\s*", body):
        if m.start() < i:
            continue
        j = m.end()
        if j < len(body) and body[j] == "{":
            k = _match_brace(body, j)
            fields[m.group(1).lower()] = body[j + 1:k - 1]
        else:
            k = j
            while k < len(body) and body[k] not in ",\n":
                k += 1
            fields[m.group(1).lower()] = body[j:k].strip().strip('"')
        i = k
    return fields


def strip_bib_for_archive(text: str) -> str:
    """Keep only the entries, each without its annote field."""
    kept = []
    for _typ, _key, start, end, _body in bib_entries(text):
        entry = text[start:end]
        m = re.search(r"\n[ \t]*annote\s*=\s*\{", entry, re.I)
        if m:
            k = _match_brace(entry, entry.index("{", m.end() - 1))
            tail = re.match(r"[ \t]*,?", entry[k:]).end()     # never eat the newline
            entry = entry[:m.start()] + entry[k + tail:]
            entry = re.sub(r",(\s*)\}\s*$", r"\1}", entry)   # no dangling comma
        kept.append(entry.rstrip())
    return "\n\n".join(kept) + "\n"


# =====================================================================================
#  the build graph
# =====================================================================================
def resolve(root: str):
    """Follow every build reference from main.tex under ``root``. -> (found, missing)."""
    found, missing = {MAIN: "document root"}, []
    graphic_dirs = [""]
    queue, seen = [MAIN], set()

    def add(rel: str, why: str):
        rel = os.path.normpath(rel).replace(os.sep, "/")
        found.setdefault(rel, why)
        if rel.endswith(".tex") and rel not in seen:
            queue.append(rel)

    while queue:
        cur = queue.pop(0)
        if cur in seen:
            continue
        seen.add(cur)
        src = strip_comments_for_scan(open(os.path.join(root, cur), encoding="utf-8").read())

        m = re.search(r"\\graphicspath\{(.+?)\}\s*$", src, re.M)
        if m:
            for d in re.findall(r"\{(.*?)\}", m.group(1)):
                if d not in graphic_dirs:
                    graphic_dirs.append(d)
        for cmd, target in re.findall(r"\\(input|include)\{(.+?)\}", src):
            rel = target if target.endswith(".tex") else target + ".tex"
            (add(rel, f"\\{cmd}{{{target}}} in {cur}")
             if os.path.isfile(os.path.join(root, rel))
             else missing.append(f"\\{cmd}{{{target}}} in {cur}"))
        for target in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{(.+?)\}", src):
            cands = [target] if os.path.splitext(target)[1] else \
                [target + e for e in GRAPHIC_EXT]
            hit = next((d + c for d in graphic_dirs for c in cands
                        if os.path.isfile(os.path.join(root, d + c))), None)
            (add(hit, f"\\includegraphics{{{target}}} in {cur}") if hit
             else missing.append(f"\\includegraphics{{{target}}} in {cur}"))
        for group in re.findall(r"\\bibliography\{(.+?)\}", src):
            for b in (x.strip() for x in group.split(",")):
                rel = b if b.endswith(".bib") else b + ".bib"
                (add(rel, f"\\bibliography{{{b}}} in {cur}")
                 if os.path.isfile(os.path.join(root, rel))
                 else missing.append(f"\\bibliography{{{b}}} in {cur}"))
        for style in re.findall(r"\\bibliographystyle\{(.+?)\}", src):
            if os.path.isfile(os.path.join(root, style + ".bst")):
                add(style + ".bst", f"\\bibliographystyle{{{style}}} in {cur}")
        for kind, group in re.findall(
                r"\\(documentclass|usepackage)(?:\[[^\]]*\])?\{(.+?)\}", src):
            ext = ".cls" if kind == "documentclass" else ".sty"
            for pkg in (x.strip() for x in group.split(",")):
                if os.path.isfile(os.path.join(root, pkg + ext)):
                    add(pkg + ext, f"\\{kind}{{{pkg}}} in {cur}")
    return found, missing


# =====================================================================================
#  the submission gate
# =====================================================================================
def _tex_sources(root: str, found) -> dict:
    return {rel: open(os.path.join(root, rel), encoding="utf-8").read()
            for rel in found if rel.endswith(".tex")}


def cited_keys(tex: dict) -> set:
    keys = set()
    for text in tex.values():
        for group in re.findall(r"\\cite\w*\*?(?:\[[^\]]*\])*\{([^}]+)\}",
                                strip_comments_for_scan(text)):
            keys |= {k.strip() for k in group.split(",") if k.strip()}
    return keys


def register_rows(text: str) -> dict:
    """key -> (claims, evidence, what-we-claim) from the register's four-column table."""
    rows = {}
    for line in text.split("\n"):
        m = re.match(r"^\|\s*`([^`]+)`\s*\|(.*)\|\s*$", line)
        if not m:
            continue
        cells = [c.strip() for c in m.group(2).split(" | ")]
        if len(cells) >= 3:
            rows[m.group(1)] = (cells[0], cells[1], " | ".join(cells[2:]))
    return rows


def _has_marker(text: str, markers) -> list:
    # A hyphen counts as part of the word, so "incomplete-ring" is not INCOMPLETE.
    return [mk for mk in markers
            if re.search(r"(?<![A-Za-z-])" + re.escape(mk) + r"(?![A-Za-z-])", text, re.I)]


def gate(tex: dict, bib_text: str, register_text: str) -> list:
    """Every reason this source is not ready to submit. Empty list == pass."""
    v = []
    cited = cited_keys(tex)
    entries = {key: (typ, bib_fields(body)) for typ, key, _s, _e, body in bib_entries(bib_text)}
    reg = register_rows(register_text)

    for key in sorted(cited):
        if key not in entries:
            v.append(f"[bib]      {key}: cited but has no .bib entry")
            continue
        _typ, f = entries[key]
        marks = _has_marker(f.get("annote", ""), ANNOTE_MARKERS)
        if marks:
            v.append(f"[annote]   {key}: unresolved marker(s) {', '.join(marks)}")
        missing = [x for x in ("author", "title") if not f.get(x, "").strip()]
        if not f.get("year", "").strip() and not f.get("eprint", "").strip():
            missing.append("year and eprint")
        if missing:
            v.append(f"[fields]   {key}: missing {', '.join(missing)}")
        if key not in reg:
            v.append(f"[register] {key}: cited but has no register row")
            continue
        claims, evidence, sites = reg[key]
        ev = evidence.strip().lower()
        if ev in ("", "—", "-") or ev.startswith("none"):
            v.append(f"[register] {key}: evidence is none ('{evidence[:40]}')")
        if claims.strip() in ("", "—", "-") or claims.strip().startswith("✗"):
            v.append(f"[register] {key}: claims column is empty (marked x)")
        open_marks = _has_marker(sites, REGISTER_OPEN_MARKS)
        if open_marks:
            v.append(f"[register] {key}: open per-site mark(s): {', '.join(open_marks)}")

    for rel, text in sorted(tex.items()):
        body = strip_comments_for_scan(text)
        for n, line in enumerate(body.split("\n"), 1):
            if re.search(r"\\newcommand\s*\{?\\(todo|PLACEHOLDER)\b", line):
                continue
            for macro in ("todo", "PLACEHOLDER"):
                for m in re.finditer(r"\\" + macro + r"\s*\{", line):
                    arg = line[m.end():m.end() + 60].split("}")[0]
                    v.append(f"[source]   {rel}:{n}: \\{macro}{{{arg}...}}")
        sec = list(re.finditer(r"\\section\*?\{([^}]*)\}", body))
        for i, m in enumerate(sec):
            nxt = sec[i + 1].start() if i + 1 < len(sec) else len(body)
            rest = re.sub(r"\\label\{[^}]*\}", "", body[m.end():nxt])
            rest = rest.replace(r"\appendix", "").replace(r"\end{document}", "")
            if not rest.strip():
                ln = body[:m.start()].count("\n") + 1
                v.append(f"[source]   {rel}:{ln}: section '{m.group(1)}' has no body")
    return v


# =====================================================================================
#  archive checks
# =====================================================================================
def check_archive(names, referenced):
    assert MAIN in names, "main.tex is not at the archive root"
    assert not any("\\" in n for n in names), "a backslash reached an archive path"
    bad = [n for n in names
           if any(fnmatch.fnmatch(os.path.basename(n), g) for g in FORBIDDEN_GLOBS)]
    assert not bad, f"forbidden working files in the archive: {bad}"
    unref = [n for n in names if n != MAIN and n not in referenced]
    assert not unref, f"archived files the build never references: {unref}"
    badext = [n for n in names if os.path.splitext(n)[1].lower() not in ALLOWED_EXT]
    assert not badext, f"archived files with a disallowed extension: {badext}"


def check_archived_text(rel: str, text: str, original: str | None = None):
    if rel.endswith(".tex"):
        bad = comment_text_lines(text)
        assert not bad, f"{rel}: comment text survives on line(s) {bad[:10]}"
    if rel.endswith(".bib"):
        assert not re.search(r"^\s*annote\s*=", text, re.M | re.I), f"{rel}: annote shipped"
        spans, pos, outside = bib_entries(text), 0, ""
        for _t, _k, s, e, _b in spans:
            outside += text[pos:s]
            pos = e
        outside += text[pos:]
        assert not outside.strip(), f"{rel}: text outside entries survives"
        if original is not None:
            a = {k: (t, {f: x for f, x in bib_fields(b).items() if f != "annote"})
                 for t, k, _s, _e, b in bib_entries(original)}
            b = {k: (t, bib_fields(bd)) for t, k, _s, _e, bd in bib_entries(text)}
            assert a == b, f"{rel}: an entry or field changed beyond annote removal"


# =====================================================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="report only; write nothing")
    ap.add_argument("--draft", action="store_true",
                    help="build despite gate violations, as paper_DRAFT.zip")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    found, missing = resolve(PAPER)
    if missing:
        print("UNRESOLVED BUILD REFERENCES -- refusing to write an archive:")
        for m in missing:
            print("   ", m)
        return 1
    kept = sorted(found)

    print("=" * 78)
    print("paper.zip -- allow-list resolved from main.tex")
    print("=" * 78)
    for rel in kept:
        print(f"  {rel:<42}<- {found[rel]}")
    print(f"\n  {len(kept)} files")

    tex = _tex_sources(PAPER, found)
    bib_rel = next(r for r in kept if r.endswith(".bib"))
    bib_text = open(os.path.join(PAPER, bib_rel), encoding="utf-8").read()
    violations = gate(tex, bib_text, open(REGISTER, encoding="utf-8").read())

    print("\n" + "=" * 78)
    print(f"SUBMISSION GATE -- {len(violations)} violation(s)")
    print("=" * 78)
    for x in violations:
        print("  " + x)

    if a.list:
        print("\n  --list given: nothing written.")
        return 0
    if violations and not a.draft:
        if os.path.exists(OUT):
            os.remove(OUT)
            print(f"\n  removed stale {os.path.relpath(OUT, REPO)} -- it did not pass this gate")
        print("\n  REFUSED: paper.zip not written. Fix the list above, or pass --draft.")
        return 1

    out = OUT_DRAFT if a.draft else OUT
    if a.draft and violations:
        print(f"\n  WARNING: --draft given; writing {os.path.basename(out)} despite "
              f"{len(violations)} violation(s). Do not upload it.")

    before_after = {}
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for rel in kept:
            path = os.path.join(PAPER, rel)
            if rel.endswith(".tex"):
                orig = open(path, encoding="utf-8").read()
                new = strip_tex_comments(orig)
                before_after[rel] = (len(orig.splitlines()), len(new.splitlines()))
                zf.writestr(rel, new)
            elif rel.endswith(".bib"):
                orig = open(path, encoding="utf-8").read()
                new = strip_bib_for_archive(orig)
                before_after[rel] = (len(orig.splitlines()), len(new.splitlines()))
                zf.writestr(rel, new)
            else:
                zf.write(path, arcname=rel)

    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
        assert zf.testzip() is None, "archive failed its own integrity check"
        check_archive(names, set(found))
        for rel in names:
            if rel.endswith((".tex", ".bib")):
                orig = open(os.path.join(PAPER, rel), encoding="utf-8").read()
                check_archived_text(rel, zf.read(rel).decode("utf-8"),
                                    orig if rel.endswith(".bib") else None)
        tmp = tempfile.mkdtemp(prefix="paper_zip_verify_")
        try:
            zf.extractall(tmp)
            refound, remissing = resolve(tmp)
            assert not remissing, f"references unresolved inside the archive: {remissing}"
            assert set(refound) == set(names), \
                "the unzipped archive's build graph does not match its contents"
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    print("\n  lines before -> after (archived copy only):")
    for rel, (b, c) in before_after.items():
        print(f"    {rel:<42}{b:>5} -> {c:<5}")
    print(f"\n  wrote {os.path.relpath(out, REPO)} ({os.path.getsize(out) / 1e6:.2f} MB, "
          f"verified: every build reference resolves inside the unzipped archive)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
