r"""
make_arxiv_abstract.py -- the plain-text abstract for arXiv, with its length checked.
====================================================================================

    python paper/scripts/make_arxiv_abstract.py

Reads the abstract from ``paper/main.tex``, expands every number macro from the generated
macro files (``paper/tables/M_*.tex``, which must already be built), converts the LaTeX to
plain text, and fails if the result is longer than arXiv's limit of 1,920 characters for the
abstract field.

Writes ``paper/data/arxiv_abstract.txt`` and ``paper/data/arxiv_abstract.json`` (the
character count and the limit) and nothing else.
"""
from __future__ import annotations

import glob
import json
import os
import re

import _common as C

LIMIT = 1920


def macros() -> dict:
    out = {}
    for p in sorted(glob.glob(os.path.join(C.TABLES, "M_*.tex"))):
        txt = open(p, encoding="utf-8").read()
        for name, body in re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}\s*$", txt, re.M):
            out[name] = body
    return out


def plain(tex: str, mac: dict) -> str:
    s = tex
    for _ in range(3):                                   # macros may expand to macros
        s = re.sub(r"\\(\w+)\{\}", lambda m: mac.get(m.group(1), m.group(0)), s)
    unresolved = re.findall(r"\\[A-Za-z]+\{\}", s)
    assert not unresolved, f"unexpanded macros in the abstract: {unresolved}"
    s = s.replace(r"\%", "%").replace(r"\,", " ").replace("~", " ")
    s = re.sub(r"\$10\^\{-3\}\$", "10^-3", s)
    s = s.replace("$", "")
    s = s.replace("---", "-").replace("--", "-")
    s = re.sub(r"\\text\{n/a\}", "n/a", s)
    assert "\\" not in s, f"LaTeX left in the abstract: {s[s.index(chr(92)) - 20:][:60]}"
    return re.sub(r"\s+", " ", s).strip()


def main() -> int:
    main_tex = open(os.path.join(C.PAPER, "main.tex"), encoding="utf-8").read()
    body = main_tex.split(r"\begin{abstract}", 1)[1].split(r"\end{abstract}", 1)[0]
    text = plain(body, macros())
    n = len(text)
    os.makedirs(C.DATA, exist_ok=True)
    with open(os.path.join(C.DATA, "arxiv_abstract.txt"), "w", encoding="utf-8",
              newline="\n") as fh:
        fh.write(text + "\n")
    with open(os.path.join(C.DATA, "arxiv_abstract.json"), "w", encoding="utf-8",
              newline="\n") as fh:
        json.dump({"characters": n, "limit": LIMIT, "within_limit": n <= LIMIT}, fh, indent=1)
    print(f"  arXiv abstract: {n} characters (limit {LIMIT})")
    if n > LIMIT:
        raise SystemExit(f"the abstract is {n} characters, over arXiv's {LIMIT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
