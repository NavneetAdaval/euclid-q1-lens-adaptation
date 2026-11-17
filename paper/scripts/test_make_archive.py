"""test_make_archive.py -- known-bad tests for make_archive.py.

    python paper/scripts/test_make_archive.py

Known-bad tests for make_archive.py: every gate condition and the comment assertion.

Each case builds a minimal source (tex dict, bib text, register text) that is clean
except for ONE defect, and checks that exactly the intended condition fires. A clean
baseline must produce zero violations, so a condition firing is attributable.
"""
import sys

import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import make_archive as m  # noqa: E402

MAIN = r"""\documentclass{article}
\newcommand{\todo}[1]{#1}
\newcommand{\PLACEHOLDER}[1]{#1}
\begin{document}
\section{Intro}
Text citing \citep{good}.
\end{document}
"""
BIB = """@article{good,
  author = {A, B},
  title  = {T},
  year   = {2020},
  annote = {Checked against the full text; incomplete-ring morphology noted.}
}
"""
REG = """| Key | What the paper claims | Evidence | What we claim with it |
|---|---|---|---|
| `good` | A claim. | body | **(§1)** a use — **body, checked** |
"""


def run(tex=None, bib=BIB, reg=REG):
    return m.gate({"main.tex": tex or MAIN}, bib, reg)


results = []


def case(name, violations, expect):
    ok = (any(expect in v for v in violations) if expect else not violations)
    results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<48} -> {violations if violations else 'clean'}")


print("GATE CONDITIONS (each on a known-bad case)")
case("clean baseline", run(), None)
for mk in ("UNVERIFIED", "INCOMPLETE", "TODO", "CHECK", "needed"):
    case(f"annote marker {mk}", run(bib=BIB.replace("Checked against the full text",
                                                    f"venue {mk} here")), "[annote]")
case("uncited entry's annote is ignored",
     run(bib=BIB + "@misc{other,\n author={X}, title={Y}, year={1}, annote={UNVERIFIED}\n}\n"),
     None)
case("register evidence none", run(reg=REG.replace("| body |", "| none — summarise |")),
     "evidence is none")
case("register claims empty", run(reg=REG.replace("| A claim. |", "| ✗ |")),
     "claims column is empty")
case("register open per-site mark", run(reg=REG.replace("body, checked", "body needed")),
     "open per-site")
case("cited key with no register row", run(reg=REG.replace("`good`", "`other`")),
     "no register row")
case("missing author", run(bib=BIB.replace("  author = {A, B},\n", "")), "missing author")
case("missing title", run(bib=BIB.replace("  title  = {T},\n", "")), "missing title")
case("neither year nor eprint", run(bib=BIB.replace("  year   = {2020},\n", "")),
     "year and eprint")
case("eprint without year is fine",
     run(bib=BIB.replace("  year   = {2020},\n", "  eprint = {1234.5678},\n")), None)
case("\\todo in source", run(MAIN.replace("Text citing", r"Text\todo{fix} citing")),
     r"\todo")
case("\\PLACEHOLDER in source", run(MAIN.replace("Text citing", r"\PLACEHOLDER{NAME} citing")),
     r"\PLACEHOLDER")
case("\\todo inside a comment is ignored", run(MAIN.replace("Text citing", "Text % \\todo{x}\nciting")),
     None)
case("empty section", run(MAIN.replace(r"\end{document}", "\\section{Stub}\n\\label{s}\n% WILL ARGUE: x\n\\end{document}")),
     "has no body")

print("\nCOMMENT STRIPPING AND ITS ASSERTION")
bad = "Text % a comment\nMore\n"
case("assertion fires on known-bad .tex", [f"bad lines {m.comment_text_lines(bad)}"]
     if m.comment_text_lines(bad) else [], "bad lines")
src = ("A line % remove me\n% whole comment line\n50\\% literal\n\\\\% after a linebreak\n"
       "\\url{http://x.org/a%20b}\n\\href{http://x.org/%7E}{link} % c\n\\verb|%|\n"
       "\\begin{verbatim}\n% kept verbatim\n\\end{verbatim}\n")
out = m.strip_tex_comments(src)
expect = ("A line %\n50\\% literal\n\\\\%\n\\url{http://x.org/a%20b}\n"
          "\\href{http://x.org/%7E}{link} %\n\\verb|%|\n\\begin{verbatim}\n% kept verbatim\n"
          "\\end{verbatim}\n")
case("strip rules (%, \\%, \\\\%, url, href, verb, verbatim)", [] if out == expect else
     [f"got {out!r}"], None)
case("assertion passes on the stripped text", [f"bad {m.comment_text_lines(out)}"]
     if m.comment_text_lines(out) else [], None)

print("\nBIB SANITISER")
bib_out = m.strip_bib_for_archive("% header comment\n" + BIB)
case("annote and comments removed", [] if ("annote" not in bib_out and "%" not in bib_out)
     else [bib_out], None)
try:
    m.check_archived_text("refs.bib", bib_out, BIB)
    case("other fields unchanged (assertion passes)", [], None)
except AssertionError as e:
    case("other fields unchanged (assertion passes)", [str(e)], None)
try:
    m.check_archived_text("refs.bib", bib_out.replace("{T}", "{U}"), BIB)
    case("assertion fires if a field changes", [], "fired")
except AssertionError as e:
    case("assertion fires if a field changes", ["fired: " + str(e)], "fired")

print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
