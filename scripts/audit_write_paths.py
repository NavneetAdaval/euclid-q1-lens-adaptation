"""
audit_write_paths.py
====================
Find WRITE sites whose destination is hardcoded rather than derived from ``--out``.

WHY. ``analyze_coral_dev8000_budget.py`` wrote its analysis to a path fixed in the source.
Running it for round 6 with ``--out runs/bnm_r6_budget.json`` therefore put BNM numbers into
``runs/coral_r4_budget_analysis.json`` and destroyed CORAL's. The instance is fixed; this
finds the class, the same way the surviving ``run_dann_r5.py`` literal was found by AST scan
rather than by re-reading the file.

A site is REPORTED when its path expression is built only from literals and module-level
constants -- nothing traceable to a command-line argument or a function parameter. That is a
heuristic, deliberately tuned to over-report rather than under-report: a write that is
genuinely fixed (a cache, a log) is cheap to read past, a write that should have followed
``--out`` and did not is what cost CORAL's file.

    python -u scripts/audit_write_paths.py
"""
from __future__ import annotations

import ast
import io
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(_HERE)

WRITE_FUNCS = {"open", "savefig", "to_csv", "to_json", "write_text", "write_bytes",
               "imwrite", "save", "savez", "savez_compressed", "dump"}
# names that mean "this came from outside the source"
EXTERNAL = ("args", "out", "dest", "dst", "target", "path", "outdir", "out_dir",
            "results_out", "o", "p", "fp", "where", "prefix", "run_dir")


def _is_write_call(node):
    """A call that creates or truncates a file, plus the arg index of its path."""
    f = node.func
    name = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute)
                                                 else None)
    if name not in WRITE_FUNCS:
        return None
    if name == "open":
        mode = None
        if len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
            mode = node.args[1].value
        for kw in node.keywords:
            if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                mode = kw.value.value
        if not (isinstance(mode, str) and ("w" in mode or "a" in mode or "x" in mode)):
            return None
        return 0
    if name == "dump":
        # json.dump(obj, open(path, "w")) -- the inner open is caught on its own
        return None
    return 0


def audit_file(path):
    src = io.open(path, encoding="utf-8", errors="replace").read()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []

    # names bound from argparse results anywhere in the module
    argish = set(EXTERNAL)
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and len(n.targets) == 1 \
                and isinstance(n.targets[0], ast.Name):
            seg = ast.get_source_segment(src, n.value) or ""
            if "args." in seg or "parse_args" in seg or ".out" in seg:
                argish.add(n.targets[0].id)

    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        idx = _is_write_call(node)
        if idx is None or len(node.args) <= idx:
            continue
        seg = ast.get_source_segment(src, node.args[idx]) or ""
        if not seg:
            continue
        # does the path reference anything external?
        names = {m.id for m in ast.walk(node.args[idx]) if isinstance(m, ast.Name)}
        attrs = seg.replace(" ", "")
        external = bool(names & argish) or "args." in attrs or "self." in attrs
        has_literal = any(isinstance(m, ast.Constant) and isinstance(m.value, str)
                          and ("." in m.value or "/" in m.value)
                          for m in ast.walk(node.args[idx]))
        if not external and has_literal:
            hits.append((node.lineno, seg.strip().replace("\n", " ")[:100]))
    return hits


def main():
    scripts = sorted(f for f in os.listdir(_HERE) if f.endswith(".py"))
    total, files_with = 0, 0
    print("=" * 78)
    print("HARDCODED WRITE PATHS -- not derived from --out or a parameter")
    print("=" * 78)
    for name in scripts:
        hits = audit_file(os.path.join(_HERE, name))
        if not hits:
            continue
        files_with += 1
        total += len(hits)
        print(f"\n  {name}")
        for ln, seg in hits:
            print(f"    line {ln:>5}: {seg}")
    print()
    print("=" * 78)
    print(f"  {total} hardcoded write site(s) across {files_with} of {len(scripts)} scripts")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
