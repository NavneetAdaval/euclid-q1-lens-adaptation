r"""
check_latex.py -- structural checks on the LaTeX project, with no LaTeX installed.
==================================================================================

    python paper/scripts/check_latex.py

**THIS IS NOT A COMPILER AND DOES NOT CLAIM TO BE ONE.** No TeX distribution exists on
the machine this stage was built on, so ``main.tex`` has NOT been compiled and
``STAGE1_REPORT.md`` says so plainly. What this script does instead is catch the failure
modes a generator can introduce mechanically, which are most of the ones that would
actually break a first build:

  1. every ``\input{...}`` resolves to a file that exists;
  2. every ``\includegraphics{...}`` resolves, honouring ``\graphicspath``;
  3. ``\begin``/``\end`` environments balance, in order, per file;
  4. braces balance per file;
  5. every ``tabular`` row has the column count its preamble declares (a row carrying
     ``\multicolumn`` is counted by its span; rule lines are skipped);
  6. no **unescaped ``%``** appears after text on a line -- in LaTeX that silently eats
     the rest of the line, which is exactly the error that turns a percentage into a
     missing table cell;
  7. no bare ``_`` outside maths and ``\texttt``, and no bare ``&`` outside a tabular;
  8. every ``\ref`` has a ``\label``, and no label is defined twice.

A clean run means the project is structurally sound. **It does not mean it compiles.**
"""
from __future__ import annotations

import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)

PROBLEMS = []


def bad(where, msg):
    PROBLEMS.append(f"{where}: {msg}")


def strip_comments(text: str) -> str:
    r"""Remove LaTeX comments, respecting ``\%``."""
    out = []
    for line in text.split("\n"):
        i, n, cut = 0, len(line), None
        while i < n:
            if line[i] == "\\":
                i += 2
                continue
            if line[i] == "%":
                cut = i
                break
            i += 1
        out.append(line if cut is None else line[:cut])
    return "\n".join(out)


def tex_files():
    files = [os.path.join(PAPER, "main.tex")]
    for sub in ("sections", "tables"):
        d = os.path.join(PAPER, sub)
        if os.path.isdir(d):
            files += [os.path.join(d, f) for f in sorted(os.listdir(d))
                      if f.endswith(".tex")]
    return files


def check_inputs_and_graphics():
    gp = [PAPER]
    main = open(os.path.join(PAPER, "main.tex"), encoding="utf-8").read()
    m = re.search(r"\\graphicspath\{(.+?)\}\s*$", main, re.M)
    if m:
        gp += [os.path.join(PAPER, p) for p in re.findall(r"\{(.*?)\}", m.group(1))]

    for f in tex_files():
        rel = os.path.relpath(f, PAPER)
        src = strip_comments(open(f, encoding="utf-8").read())
        for target in re.findall(r"\\input\{(.+?)\}", src):
            # pdflatex resolves \input against the WORKING DIRECTORY -- the directory
            # main.tex is compiled from -- NOT against the file doing the including. A
            # path like ../tables/T1 written inside sections/ looks right and is wrong.
            p = os.path.normpath(os.path.join(PAPER, target))
            if not (os.path.isfile(p) or os.path.isfile(p + ".tex")):
                bad(rel, rf"\input{{{target}}} does not resolve from the document root "
                         rf"(looked for {p}.tex)")
        for target in re.findall(r"\\includegraphics(?:\[.*?\])?\{(.+?)\}", src):
            if not any(os.path.isfile(os.path.join(d, target)) for d in gp):
                bad(rel, rf"\includegraphics{{{target}}} is not on the graphics path")


def check_balance():
    for f in tex_files():
        rel = os.path.relpath(f, PAPER)
        src = strip_comments(open(f, encoding="utf-8").read())
        depth = 0
        for i, ch in enumerate(src):
            if i and src[i - 1] == "\\":
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth < 0:
                    bad(rel, "a closing brace with no opener")
                    break
        if depth > 0:
            bad(rel, f"{depth} unclosed brace(s)")

        stack = []
        for kind, name in re.findall(r"\\(begin|end)\{(.+?)\}", src):
            if kind == "begin":
                stack.append(name)
            elif not stack:
                bad(rel, rf"\end{{{name}}} with no matching \begin")
            elif stack[-1] != name:
                bad(rel, rf"\end{{{name}}} closes \begin{{{stack[-1]}}}")
                stack.pop()
            else:
                stack.pop()
        if stack:
            bad(rel, "unclosed environment(s): " + ", ".join(stack))


def check_tabular_columns():
    # The column spec is brace-matched rather than read up to the first '}', so a p{...}
    # width such as p{0.3\textwidth} counts as one column instead of truncating the spec.
    col = re.compile(r"\\begin\{tabular\}\{")
    rule_start = ("\\toprule", "\\midrule", "\\bottomrule", "\\cmidrule", "\\hline",
                  "\\addlinespace")
    for f in tex_files():
        rel = os.path.relpath(f, PAPER)
        src = strip_comments(open(f, encoding="utf-8").read())
        pos = 0
        while True:
            m = col.search(src, pos)
            if not m:
                break
            depth, i = 1, m.end()
            while i < len(src) and depth:
                depth += {"{": 1, "}": -1}.get(src[i], 0)
                i += 1
            spec, body_start = src[m.end():i - 1], i
            end = src.find("\\end{tabular}", body_start)
            if end < 0:
                bad(rel, "a tabular with no end")
                break
            preamble = re.sub(r"[pmb]\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", "p", spec)
            ncol = len(re.sub(r"[^lcrp]", "", preamble))
            for raw in src[body_start:end].split("\\\\"):
                line = raw.strip()
                while line.startswith(rule_start):
                    line = re.sub(r"^\\(?:top|mid|bottom)rule\s*", "", line)
                    line = re.sub(r"^\\cmidrule(?:\(.*?\))?(?:\{.*?\})?\s*", "", line)
                    line = re.sub(r"^\\hline\s*", "", line)
                    line = re.sub(r"^\\addlinespace(?:\[[^\]]*\])?\s*", "", line)
                if not line:
                    continue
                span = sum(int(s) - 1 for s in re.findall(r"\\multicolumn\{(\d+)\}", line))
                cells = len(re.findall(r"(?<!\\)&", line)) + 1 + span
                if cells != ncol:
                    bad(rel, f"tabular declares {ncol} columns, a row has {cells}: "
                             f"{line[:70]}")
            pos = end


def check_escapes():
    tt = re.compile(r"\\texttt\{[^{}]*\}")
    # Package options and file names legitimately carry & and _ ("sort&compress",
    # "F6_completeness_purity.pdf"), so those constructs are stripped before probing.
    named = re.compile(r"\\(?:label|ref|pageref|cite\w*|input|includegraphics"
                       r"|begin|end|graphicspath|bibliography\w*|usepackage"
                       r"|documentclass|newcommand)"
                       r"(?:\[[^\]]*\])?\{[^{}]*\}")
    # Display maths is maths: `_`, `^` and `&` are legitimate inside it, and a line-by-line
    # probe that only understands $...$ reports every display equation as an error. The
    # unescaped-% probe still applies inside these -- a stray % comments out the line there
    # exactly as it does anywhere else.
    MATHS_ENVS = ("equation", "align", "gather", "multline", "displaymath", "eqnarray",
                  "split", "alignat", "flalign")
    for f in tex_files():
        rel = os.path.relpath(f, PAPER)
        raw = open(f, encoding="utf-8").read()
        in_tabular = False
        in_maths = False
        for n, line in enumerate(raw.split("\n"), 1):
            s = line.strip()
            if s.startswith("\\begin{tabular}"):
                in_tabular = True
            if s.startswith("\\end{tabular}"):
                in_tabular = False
            if any(s.startswith("\\begin{" + e) for e in MATHS_ENVS) or s.startswith("\\["):
                in_maths = True
            was_maths = in_maths
            if any(s.startswith("\\end{" + e) for e in MATHS_ENVS) or s.startswith("\\]"):
                in_maths = False

            i, cut = 0, None
            while i < len(line):
                if line[i] == "\\":
                    i += 2
                    continue
                if line[i] == "%":
                    cut = i
                    break
                i += 1
            if cut is not None and line[:cut].strip():
                bad(rel, f"line {n}: unescaped '%' after text; it eats the rest of the "
                         f"line: {line[:70]}")

            probe = strip_comments(line)
            probe = re.sub(r"\$[^$]*\$", "", probe)
            probe = tt.sub("", probe)
            probe = named.sub("", probe)
            if was_maths:
                continue
            if re.search(r"(?<!\\)_", probe):
                bad(rel, f"line {n}: bare '_' outside maths/texttt: {probe.strip()[:70]}")
            if not in_tabular and re.search(r"(?<!\\)&", probe):
                bad(rel, f"line {n}: bare '&' outside a tabular: {probe.strip()[:70]}")


def check_labels():
    labels, refs = {}, []
    for f in tex_files():
        rel = os.path.relpath(f, PAPER)
        src = strip_comments(open(f, encoding="utf-8").read())
        for lab in re.findall(r"\\label\{(.+?)\}", src):
            if lab in labels:
                bad(rel, f"label '{lab}' is already defined in {labels[lab]}")
            labels[lab] = rel
        refs += [(rel, r) for r in re.findall(r"\\(?:page)?ref\{(.+?)\}", src)]
    for rel, r in refs:
        if r not in labels:
            bad(rel, rf"\ref{{{r}}} has no \label")


# The house style bans verdict words. "confirm" is banned as a whole verb family
# (confirms, confirmed, confirming, confirmation). Two uses are terms, not verdicts, and
# are allow-listed: "confirmatory" (the design term for folds 1--4) and "confirmation
# bias" (the named phenomenon in the cited pseudo-labelling literature).
BANNED = re.compile(r"\b(?:significant(?:ly)?|proves|demonstrates that|clearly|"
                    r"dramatic(?:ally)?|substantially better|confirm\w*)\b", re.I)
BANNED_ALLOW = re.compile(r"\bconfirmatory\b|\bconfirmation bias\b", re.I)


def paragraphs(raw: str):
    """Yield ``(text, line_of)`` per paragraph, comments stripped, whitespace normalised.

    A paragraph ends at a blank line of the SOURCE (a comment-only line does not end one,
    as in LaTeX). Every run of whitespace, newlines included, collapses to one space, so a
    phrase split across lines ("demonstrates\\nthat", "confirmation\\nbias") is seen whole.
    ``line_of[i]`` is the 1-based source line of character ``i`` of ``text``, so a hit is
    reported at the line where the phrase starts.
    """
    text, line_of = [], []
    for n, (orig, line) in enumerate(zip(raw.split("\n"),
                                         strip_comments(raw).split("\n")), 1):
        if not orig.strip():
            if text:
                yield "".join(text), line_of
            text, line_of = [], []
            continue
        for ch in line + " ":
            if ch.isspace():
                if not text or text[-1] == " ":
                    continue
                ch = " "
            text.append(ch)
            line_of.append(n)
    if text:
        yield "".join(text), line_of


def check_banned_words():
    for f in tex_files():
        rel = os.path.relpath(f, PAPER)
        for para, line_of in paragraphs(open(f, encoding="utf-8").read()):
            masked = BANNED_ALLOW.sub(lambda m: " " * len(m.group(0)), para)
            for m in BANNED.finditer(masked):
                ctx = para[max(0, m.start() - 20):m.end() + 40].strip()
                bad(rel, f"line {line_of[m.start()]}: banned word '{m.group(0)}': {ctx[:70]}")


CHECKS = (check_inputs_and_graphics, check_balance, check_tabular_columns,
          check_escapes, check_labels, check_banned_words)

# A scan that has never been seen to fire is not known to work. This is the known-null:
# one deliberately broken document in which every check has something to find. Each entry
# is (a substring the corresponding PROBLEM message must contain).
BROKEN = (
    "\\section{Broken}\n"
    "\\input{sections/does_not_exist}\n"
    "\\includegraphics{no_such_figure.pdf}\n"
    "\\begin{tabular}{ll}\n"
    "a & b & c \\\\\n"
    "\\end{tabular}\n"
    # Display maths: the `_` and `&` here are legitimate and must NOT be reported. An
    # exemption that over-reaches blinds the checker silently, so the known-null tests
    # that this is skipped AND that the bare `_` further down is still caught.
    "\\begin{align}\n"
    "  x_* &= \\sigma_* + 1 \\\\\n"
    "  y_* &= \\sigma_* - 1\n"
    "\\end{align}\n"
    "\\begin{figure}\n"
    "\\end{table}\n"
    "50% of the sample, unescaped\n"
    "bare_underscore in text\n"
    "Smith & Jones, an ampersand in running text\n"
    # Banned-word scan: exactly ONE report is expected from these two lines -- the
    # "confirms" -- while "confirmatory" and "confirmation bias" are terms and must pass.
    "The run confirms it on the confirmatory folds.\n"
    "Pseudo-labels suffer from confirmation bias.\n"
    # The scan is per paragraph with whitespace normalised, so a phrase split by a line
    # break is seen whole: the split banned phrase MUST fire, reported at the line where
    # it starts (SPLIT_LINE below); the split allow-listed phrase must NOT.
    "\n"
    "The adapted arm demonstrates\n"
    "that it works, and the pseudo-labels suffer from confirmation\n"
    "bias less often.\n"
    "\n"
    "\\ref{no_such_label}\n"
    "{ unclosed brace\n"
)
SPLIT_LINE = next(i for i, ln in enumerate(BROKEN.split("\n"), 1)
                  if ln.endswith("demonstrates"))
MUST_FIRE = ["does not resolve", "not on the graphics path",
             "a row has 3", "closes", "unescaped '%'", "bare '_'",
             "has no \\label", "unclosed brace", "banned word 'confirms'",
             f"line {SPLIT_LINE}: banned word 'demonstrates that'"]
# The complement: an exemption that fires anyway, or over-reaches and swallows a real
# problem, is as bad as a check that never fires. Exactly one bare-underscore report is
# expected from BROKEN -- the one in running text, not the two inside the align block.
# Exactly two banned-word reports: "confirms" and the split "demonstrates that"; the
# split "confirmation bias" must stay exempt.
MUST_NOT_FIRE = [("bare '_'", 1), ("bare '&'", 1), ("banned word", 2)]


def self_test() -> int:
    """Run every check against a document that breaks all of them."""
    import shutil
    import tempfile

    global PAPER, PROBLEMS
    tmp = tempfile.mkdtemp(prefix="check_latex_selftest_")
    try:
        os.makedirs(os.path.join(tmp, "sections"))
        with open(os.path.join(tmp, "main.tex"), "w", encoding="utf-8") as fh:
            fh.write(BROKEN)
        keep_paper, keep_problems = PAPER, list(PROBLEMS)
        PAPER, PROBLEMS[:] = tmp, []
        for fn in CHECKS:
            fn()
        fired = list(PROBLEMS)
        PAPER, PROBLEMS[:] = keep_paper, keep_problems
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("=" * 90)
    print("KNOWN-NULL: every check run against a document that breaks all of them")
    print("=" * 90)
    missed = []
    for want in MUST_FIRE:
        hit = any(want in p for p in fired)
        print(f"  {'FIRED  ' if hit else 'SILENT '} {want}")
        if not hit:
            missed.append(want)
    miscounted = []
    for want, n_expected in MUST_NOT_FIRE:
        n = sum(1 for p in fired if want in p)
        ok = (n == n_expected)
        print(f"  {'COUNT  ' if ok else 'WRONG  '} {want}: {n} "
              f"(expected exactly {n_expected} -- maths must be exempt, text must not)")
        if not ok:
            miscounted.append(f"{want}: {n} != {n_expected}")
    for p in fired:
        print("      | " + p)
    if missed or miscounted:
        if missed:
            print(f"\n  {len(missed)} check(s) did NOT fire on the known-null: "
                  + ", ".join(missed))
        if miscounted:
            print(f"  {len(miscounted)} check(s) fired the wrong number of times: "
                  + ", ".join(miscounted))
        return 1
    print(f"\n  all {len(MUST_FIRE)} checks fire on the known-null, and the maths "
          f"exemption neither over- nor under-reaches.")
    return 0


def main() -> int:
    if "--self-test" in os.sys.argv:
        return self_test()
    for fn in CHECKS:
        fn()
    print("=" * 90)
    print("STRUCTURAL CHECK of the LaTeX project "
          "(NOT a compile -- no TeX distribution on this machine)")
    print("=" * 90)
    if not PROBLEMS:
        print(f"  {len(tex_files())} .tex files: inputs and graphics resolve, braces and")
        print("  environments balance, tabular rows match their preambles, no unescaped")
        print("  '%' or bare '_' / '&', every reference has a label, no banned verdict")
        print("  word (confirm-family included; 'confirmatory', 'confirmation bias' allowed).")
        print("  THIS DOES NOT ESTABLISH THAT THE DOCUMENT COMPILES.")
        return 0
    for p in PROBLEMS:
        print("  PROBLEM  " + p)
    print(f"\n  {len(PROBLEMS)} problem(s).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
