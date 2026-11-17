"""
build_release_v3.py -- assemble the curated public release from the project.
============================================================================

    python scripts/build_release_v3.py [--out DIR] [--classification CSV]
                                       [--assets-from DIR] [--logs DIR] [--all-tests]

Reads the project; writes only ``--out`` (default ``../lens_paper_release``, which must not
exist) and ``--logs`` (default ``../lens_paper_release_build_logs``). Loads no model.

WHAT IT DOES
  1. Takes the keep set of ``release_curation/classification.csv`` (every row whose new_path is
     not "(drop)") and copies each file from the project to its new path, with fresh mtimes.
     Rows whose content is generated here, not copied, are listed in GENERATED.
  2. Applies the curation decisions (release_curation_apply_brief.md):
       * Decision 2 -- adds the five small files the hashed pre-registrations cite (CITED_EXTRA);
       * Decision 4 -- keeps in tests/ only the tests in TESTS_RUN_IN_RELEASE (each was run in an
         isolated copy of the release); the others stay in the project;
       * Decisions 1 and 3 -- nothing to add: the candidate probes and the unreported experiments
         are already in the classification's drop set.
  3. Writes data/splits/splits.json with the bare-deflector IDs exact, and data/ids/ (the exact-ID
     export, by the export script kept in experiments/05_audit/).
  4. Cleans the copies (local paths, Kaggle slugs, names in decision comments) with the same
     replacement table as the previous builds. Every changed .py must be ast-equivalent to its
     project original modulo the replaced string constants, every changed .json parse-equivalent;
     hashed, self-certifying and inventory files are never touched.
  5. Applies the path edits that let reproduce/ run in the new layout (PATH_EDITS), each asserted
     to match exactly once, and logs every one.
  6. Writes LICENSE (GPL-3.0, from --assets-from, digest checked), README.md and tests/README.md,
     then runs reproduce/release_check.py --repo to write paper/data/release_manifest.json.

The logs (copy manifest, cleaning log with the original strings, path edits) go to --logs,
OUTSIDE the release, because the cleaning log quotes what was removed.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ML = os.path.dirname(REPO)
BS = chr(92)
GPL3_SHA256 = "3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986"

# -------------------------------------------------------------------- decisions
CITED_EXTRA = [  # Decision 2: cited by runs/dann_r5_prereg.json (+ evidence copy) and runs/bnm_r6_prereg.json
    "runs/bakeoff_baseline_A/curve.csv",
    "runs/freematch_fold0_nb3_unfrozen/curve.csv",
    "runs/bnm_r6_sweep/bnm_r6_sweep_g0.00787/checkpoints.csv",
    "runs/bnm_r6_sweep/bnm_r6_sweep_g0.039351/checkpoints.csv",
    "runs/bnm_r6_sweep/bnm_r6_sweep_g0.157406/checkpoints.csv",
]
# Decision 4: the tests that ran and passed inside an isolated copy of the release (see
# paper/RELEASE_CURATION_REPORT.md for the run); every other test stays in the project only.
TESTS_RUN_IN_RELEASE = [
    "selftest_exact_vs_rounded.py", "test_checkpoint_retention.py", "test_compare_methods.py",
    "test_da_bnm.py", "test_lens_dataset.py", "test_make_archive.py", "test_metrics.py",
    "test_package_screen_results.py", "test_round7.py", "test_validators.py",
]
# rows of the classification whose content this script generates instead of copying
GENERATED = {"README.md", "LICENSE", "data/splits/splits.json", "paper/data/release_manifest.json"}
GENERATED_PREFIX = ("ids/",)          # data/ids/*: the exact-ID export (export_ids.py is copied)

# -------------------------------------------------------------------- cleaning (same table as v2)
REPL = []
for _sep in (BS * 2, BS, "/"):
    _base = _sep.join(["C:", "Users", "krris", "ML", "lens_paper"])
    REPL.append((_base + "_backup_pre_v2", _sep.join(["..", "lens_paper_backup_pre_v2"])))
    REPL.append((_base, "."))
    REPL.append((_sep.join(["C:", "Users", "krris", "Downloads", "q1_discovery_engine_lens_catalog_old.csv"]),
                 _sep.join(["data", "q1_lenses", "q1_discovery_engine_lens_catalog_old.csv"])))
    REPL.append((_sep.join(["C:", "Users", "krris", "DeepLenseSim"]), "DeepLenseSim"))
REPL.append(("/mnt/c/Users/krris/ML/lens_paper", "."))
REPL += [
    ("navneet78/lens-data", "KAGGLE_OWNER/dataset-lens-data"),
    ("navneet78/lens-paper/", "KAGGLE_OWNER/dataset-lens-paper/"),
    ("lens-paper-coral-r4", "dataset-coral-r4"),
    ("lens-paper-dann-r5", "dataset-dann-r5"),
    ("lens-paper-freematch-nb3", "dataset-freematch-nb3"),
    ("navneet78", "KAGGLE_OWNER"),
    ("Navneet decides.", "author decision."),
    ("Navneet hashes", "The author hashes"),
    ("Navneet's later decision", "the author's later decision"),
    ("trajectory Navneet declares", "trajectory the author declares"),
    ("Design notes (see chat for the reasoning):", "Design notes:"),
]
TEXT_EXT = (".py", ".sh", ".json", ".md", ".csv", ".tex", ".bib", ".txt", ".log")
NEVER_CLEAN = {"paper/main.tex"}      # the author byline stays


def repl(s: str) -> str:
    for a, b in REPL:
        s = s.replace(a, b)
    return s


# -------------------------------------------------------------------- reproduce/ path edits
# (file, exact old text, new text, why). Applied after cleaning; each must match exactly once.
PATH_EDITS = [
    ("reproduce/_common.py", "PAPER = os.path.dirname(HERE)\n",
     'PAPER = os.path.join(os.path.dirname(HERE), "paper")\n',
     "the generators moved from paper/scripts/ to reproduce/; paper/ is now a sibling"),
    ("reproduce/_common.py", 'REPO_SCRIPTS = os.path.join(REPO, "scripts")\n',
     "REPO_SCRIPTS = HERE\n",
     "the analysis modules the build imports now sit beside _common.py in reproduce/"),
    ("reproduce/_common.py", "    sys.path.insert(0, REPO_SCRIPTS)\n",
     "    sys.path.insert(0, REPO_SCRIPTS)\n"
     'sys.path.append(os.path.join(REPO, "experiments", "03_training"))  # lens_dataset, heads, continuation_state\n',
     "the one sys.path line: fig_f2 (lens_dataset), make_a5_restart (continuation_state) and a "
     "lazy import in analyze_window_average (heads) resolve from experiments/03_training"),
    ("reproduce/_common.py", 'PUBLIC_DIR = os.path.join(REPO, "public_inputs")\n',
     'PUBLIC_DIR = os.path.join(REPO, "data", "public_inputs")\n', "public_inputs/ moved to data/"),
    ("reproduce/integrity.py",
     'RECORD_EXTRACT = os.path.join(REPO, "public_inputs", "record_extract.json")\n',
     'RECORD_EXTRACT = os.path.join(REPO, "data", "public_inputs", "record_extract.json")\n',
     "public_inputs/ moved to data/"),
    ("reproduce/make_all.py", "PAPER = os.path.dirname(HERE)\nREPO = os.path.dirname(PAPER)\n",
     'PAPER = os.path.join(os.path.dirname(HERE), "paper")\nREPO = os.path.dirname(HERE)\n',
     "make_all.py moved from paper/scripts/ to reproduce/"),
    ("reproduce/check_latex.py", "PAPER = os.path.dirname(HERE)\n",
     'PAPER = os.path.join(os.path.dirname(HERE), "paper")\n', "moved to reproduce/"),
    ("reproduce/make_archive.py", "PAPER = os.path.dirname(HERE)\nREPO = os.path.dirname(PAPER)\n",
     'PAPER = os.path.join(os.path.dirname(HERE), "paper")\nREPO = os.path.dirname(HERE)\n',
     "moved to reproduce/"),
    ("reproduce/make_m_sim_source.py",
     'GEN = os.path.join(C.REPO_SCRIPTS, "generate_arc_on_real.py")\n',
     'GEN = C.R("experiments", "02_simulation", "generate_arc_on_real.py")\n',
     "the simulator moved to experiments/02_simulation/"),
    ("reproduce/make_m_prose.py", 'QUERY_SCRIPT = ("scripts", "01_query_catalog.py")\n',
     'QUERY_SCRIPT = ("experiments", "01_data", "01_query_catalog.py")\n',
     "the MER query moved to experiments/01_data/"),
    ("reproduce/make_m_prose.py", 'src = open(C.R("scripts", "generate_arc_on_real.py"), encoding="utf-8").read()\n',
     'src = open(C.R("experiments", "02_simulation", "generate_arc_on_real.py"), encoding="utf-8").read()\n',
     "the simulator moved to experiments/02_simulation/"),
    ("reproduce/crosscheck.py",
     '    qs = C.R("scripts", "01_query_catalog.py")\n'
     '    lo, hi = (float(x) for x in re.findall(r"flux_vis_psf\\s+BETWEEN\\s+([\\d.]+)\\s+AND\\s+([\\d.]+)",\n'
     '                                             open(qs, encoding="utf-8").read())[0])\n',
     '    qs = "scripts/01_query_catalog.py"         # provenance label, as in the project\n'
     '    lo, hi = (float(x) for x in re.findall(r"flux_vis_psf\\s+BETWEEN\\s+([\\d.]+)\\s+AND\\s+([\\d.]+)",\n'
     '                                             open(C.R("experiments", "01_data", "01_query_catalog.py"),\n'
     '                                                  encoding="utf-8").read())[0])\n',
     "read the query from experiments/01_data/; keep the provenance label so crosscheck.json is identical"),
    ("reproduce/prose_macros.py", "REPO = os.path.dirname(os.path.dirname(HERE))\n",
     "REPO = os.path.dirname(HERE)\n", "prose_macros.py moved from audit/decontam/ to reproduce/"),
    ("reproduce/prose_macros.py", 'os.path.join(PAPER, "scripts", "make_m_prose.py")',
     'os.path.join(HERE, "make_m_prose.py")', "make_m_prose.py is now beside it in reproduce/"),
    ("reproduce/release_check.py", "REPO_DEFAULT = os.path.dirname(os.path.dirname(HERE))\n",
     "REPO_DEFAULT = os.path.dirname(HERE)\n", "moved to reproduce/ (pass --repo to be explicit)"),
]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)                 # fresh mtime


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ML, "lens_paper_release"))
    ap.add_argument("--classification", default=os.path.join(ML, "release_curation", "classification.csv"))
    ap.add_argument("--assets-from", default=os.path.join(ML, "lens_paper_release_OLD2"),
                    help="previous release: source of LICENSE and ids/export_ids.py")
    ap.add_argument("--logs", default=os.path.join(ML, "lens_paper_release_build_logs"))
    ap.add_argument("--all-tests", action="store_true",
                    help="stage every classified test (used once, to find which run in the release)")
    a = ap.parse_args(argv)
    OUT = os.path.abspath(a.out)
    assert not os.path.exists(OUT), f"{OUT} exists -- refusing to overwrite"
    assert "Temp" not in OUT.split(os.sep) and "TEMP" not in OUT.upper().split(os.sep), "build outside Temp"
    os.makedirs(a.logs, exist_ok=True)

    rows = list(csv.DictReader(open(a.classification, encoding="utf-8")))
    keep = [r for r in rows if r["new_path"] != "(drop)"]
    plan, gen_rows, dropped_tests = [], [], []
    for r in keep:
        old, new = r["old_path"], r["new_path"]
        if old in GENERATED or old.startswith(GENERATED_PREFIX) and not old.endswith("export_ids.py"):
            gen_rows.append(r)
            continue
        if new.startswith("tests/") and not a.all_tests and os.path.basename(new) not in TESTS_RUN_IN_RELEASE:
            dropped_tests.append(old)
            continue
        if old == "ids/export_ids.py":
            src = os.path.join(a.assets_from, "ids", "export_ids.py")
        else:
            src = os.path.join(REPO, old)
        plan.append((src, new, old))
    for p in CITED_EXTRA:
        plan.append((os.path.join(REPO, p), p, p))
    # ---- copy
    for src, new, _old in plan:
        assert os.path.isfile(src), src
        copy(src, os.path.join(OUT, new))

    # ---- exact-ID splits.json (bare-deflector IDs rebuilt from RA/Dec)
    sys.path.insert(0, HERE)
    from exact_ids import rebuild
    import numpy as np
    txt = open(os.path.join(REPO, "data", "splits", "splits.json"), encoding="utf-8").read()
    sp = json.loads(txt)
    assert json.dumps(sp) == txt
    with np.load(os.path.join(REPO, "data_old", "sim_deflectors", "deflector_cutouts_clean.npz")) as d:
        r2e = {int(s): rebuild(float(x), float(y)) for s, x, y in
               zip(np.asarray(d["object_id"], np.int64), d["ra"], d["dec"])}
    for node in (sp["source_train"]["neg_bare_deflector"], sp["val"]["neg_bare_deflector"]):
        node["deflector_object_ids"] = [r2e[int(x)] for x in node["deflector_object_ids"]]
    os.makedirs(os.path.join(OUT, "data", "splits"), exist_ok=True)
    open(os.path.join(OUT, "data", "splits", "splits.json"), "w", encoding="utf-8", newline="").write(json.dumps(sp))

    # ---- data/ids: the exact-ID export
    tmp_ids = os.path.join(a.logs, "_ids_tmp")
    shutil.rmtree(tmp_ids, ignore_errors=True)
    idsnpz = [os.path.join(REPO, old) for _s, new, old in plan if os.path.basename(new) == "ids.npz"]
    subprocess.run([sys.executable, os.path.join(OUT, "experiments", "05_audit", "export_ids.py"),
                    REPO, tmp_ids, *idsnpz], check=True, stdout=subprocess.DEVNULL)
    for f in os.listdir(os.path.join(tmp_ids, "ids")):
        copy(os.path.join(tmp_ids, "ids", f), os.path.join(OUT, "data", "ids", f))
    shutil.rmtree(tmp_ids, ignore_errors=True)

    # ---- cleaning, with equivalence proofs against the project originals
    src_of = {new: src for src, new, _o in plan}
    inv = {x["path"] for x in json.load(open(os.path.join(REPO, "runs", "protected_artifacts.json"),
                                             encoding="utf-8"))["artifacts"]}
    clean_log, skipped, fail = {}, {}, []

    def ast_mapped(s):
        t = ast.parse(s)
        for n in ast.walk(t):
            if isinstance(n, ast.Constant) and isinstance(n.value, str):
                n.value = repl(n.value)
        return ast.dump(t)

    def json_mapped(o):
        if isinstance(o, str): return repl(o)
        if isinstance(o, list): return [json_mapped(x) for x in o]
        if isinstance(o, dict): return {repl(k): json_mapped(v) for k, v in o.items()}
        return o
    for new, src in sorted(src_of.items()):
        if not new.endswith(TEXT_EXT) or new in NEVER_CLEAN:
            continue
        p = os.path.join(OUT, new)
        try:
            old = open(p, "rb").read().decode("utf-8")
        except UnicodeDecodeError:
            continue
        if new in inv or new == "runs/protected_artifacts.json":
            skipped[new] = "integrity inventory"; continue
        if new.endswith(".json"):
            try:
                if "sha256_of_body_without_this_field" in json.loads(old):
                    skipped[new] = "self-certifying"; continue
            except Exception:
                pass
        cleaned = repl(old)
        if cleaned == old:
            continue
        open(p, "wb").write(cleaned.encode("utf-8"))
        e = {"changed_lines": [(i + 1, x.strip()[:160], y.strip()[:160]) for i, (x, y) in
                               enumerate(zip(old.splitlines(), cleaned.splitlines())) if x != y]}
        assert len(old.splitlines()) == len(cleaned.splitlines()), new
        if new.endswith(".py"):
            e["ast_equivalent"] = ast_mapped(old) == ast.dump(ast.parse(cleaned))
            ok = e["ast_equivalent"]
        elif new.endswith(".json"):
            e["json_equivalent"] = json_mapped(json.loads(old)) == json.loads(cleaned)
            ok = e["json_equivalent"]
        else:
            ok = True
        if not ok:
            fail.append(new)
        clean_log[new] = e
    assert not fail, f"cleaning proof failed: {fail}"

    # ---- path edits in reproduce/
    edits = []
    for f, old, new_txt, why in PATH_EDITS:
        p = os.path.join(OUT, f)
        t = open(p, encoding="utf-8").read()
        assert t.count(old) == 1, (f, old[:60], t.count(old))
        open(p, "w", encoding="utf-8", newline="").write(t.replace(old, new_txt))
        edits.append({"file": f, "before": old, "after": new_txt, "why": why})

    # ---- LICENSE, READMEs, release manifest
    lic = os.path.join(a.assets_from, "LICENSE")
    assert sha(lic) == GPL3_SHA256, "LICENSE is not the gnu.org GPL-3.0 text"
    copy(lic, os.path.join(OUT, "LICENSE"))
    kept_tests = sorted(os.path.basename(n) for _s, n, _o in plan if n.startswith("tests/"))
    open(os.path.join(OUT, "tests", "README.md"), "w", encoding="utf-8", newline="\n").write(TESTS_README)
    open(os.path.join(OUT, "README.md"), "w", encoding="utf-8", newline="\n").write(README)
    subprocess.run([sys.executable, os.path.join(OUT, "reproduce", "release_check.py"), "--repo", OUT],
                   check=True, stdout=subprocess.DEVNULL)

    # ---- logs (outside the release)
    json.dump({"copied": [{"from": os.path.relpath(s, REPO).replace(BS, "/"), "to": n} for s, n, _o in plan],
               "generated": [r["old_path"] + " -> " + r["new_path"] for r in gen_rows],
               "decision2_added": CITED_EXTRA, "decision4_tests_kept": kept_tests,
               "decision4_tests_dropped": dropped_tests},
              open(os.path.join(a.logs, "manifest_v3.json"), "w", encoding="utf-8"), indent=1)
    json.dump({"replacements": REPL, "skipped": skipped, "files": clean_log},
              open(os.path.join(a.logs, "cleaning_v3.json"), "w", encoding="utf-8"), indent=1)
    json.dump(edits, open(os.path.join(a.logs, "path_edits_v3.json"), "w", encoding="utf-8"), indent=1)
    n = sum(len(fs) for _d, _s, fs in os.walk(OUT))
    print(f"built {OUT}: {n} files; copied {len(plan)}; cleaned {len(clean_log)} "
          f"({sum(1 for k in clean_log if k.endswith('.py'))} .py, all proofs pass); "
          f"path edits {len(edits)}; tests kept {len(kept_tests)}, dropped {len(dropped_tests)}")
    return 0


TESTS_README = """# Tests

These tests check the metric family (`test_metrics.py`), the paired method comparison (`test_compare_methods.py`), the BNM loss (`test_da_bnm.py`), the data loader on fabricated shards (`test_lens_dataset.py`), the round-7 analyses on the released artifacts (`test_round7.py`), the label-free validators (`test_validators.py`) and checkpoint retention (`test_checkpoint_retention.py`, both skipping only their real-feature-cache part), the split-archive packaging (`test_package_screen_results.py`), the arXiv packager (`test_make_archive.py`) and the exact-ID comparison (`selftest_exact_vs_rounded.py`); the suites that need the feature cache, checkpoints, image shards, model weights or the original flat layout are kept in the project only.
Run each from the repository root with the code folders on the path, for example: `PYTHONPATH="reproduce:experiments/03_training:experiments/04_evaluation" python tests/test_metrics.py` (use `;` instead of `:` on Windows).
"""

README = r"""# Label-free sim-to-real strong-lens finding on Euclid Q1 — code and numbers

This folder holds the code, the stored results and the identifier lists behind the paper
*Label-free sim-to-real strong-lens finding on Euclid Q1: what a target-domain training signal
buys, and what it does not*. Every table, number macro and figure except Figure 2 regenerates
from this folder alone. It contains **no Euclid pixels**: no cutouts, image shards, simulated lens
images or model weights.

Licence: **GPL-3.0** (`LICENSE`). The Euclid data and the Discovery Engine catalogues keep their
own licences (below).

## Folder guide

| Folder | What it holds |
|---|---|
| `reproduce/` | everything the paper build runs: `make_all.py`, `crosscheck.py`, `_common.py` (paths, loaders, the test-pool filter), every table/macro/figure generator, `integrity.py`, the analysis modules they import, `prose_macros.py`, and the checking tools (`check_latex.py`, `make_archive.py`, `release_check.py`) |
| `experiments/` | the code that produced the runs, by stage: `01_data/` (catalogue query, downloads, cutouts, splits), `02_simulation/` (the arc-on-real simulator), `03_training/` (libraries, runners, Kaggle bundles, pre-registration writers), `04_evaluation/` (the analyses whose outputs the paper uses), `05_audit/` (leakage verifier, code scans, the ID and public-input exports) |
| `data/ids/` | exact Euclid object IDs for every object the splits use, and a map from every positional ID in the score files to its Euclid ID |
| `data/public_inputs/` | pixel-free extracts of the inputs that are not redistributed in full (`MANIFEST.json` lists source, licence, rows and sha256 of each) |
| `data/splits/` | `splits.json`, the split definitions (bare-deflector IDs exact) |
| `runs/` | the stored per-object test scores (`logits/step_*.npz`, `scores/*.npz`) with their `ids.npz`, the run records of the reported runs, and the hashed pre-registration and budget files |
| `kaggle/coral_r4/runs/` | the CORAL × nb3 run files and scores |
| `audit/decontam/` | the exact-ID exclusion rule (`rule.json`, hashed) and the audit's diagnostics |
| `evidence/` | the pre-fold copy of the DANN pre-registration |
| `paper/` | the paper source (`main.tex`, `sections/`, `refs.bib`) and its generated `tables/`, `figures/` and `data/` |
| `tests/` | the tests that run here (see `tests/README.md`) |

`runs/`, `kaggle/coral_r4/runs/`, `audit/decontam/` and `evidence/` keep their original paths
because the hashed files and the integrity inventory record those paths.

## Reproducing the paper

From this folder, with Python 3 and `numpy`, `pandas`, `scipy` and `matplotlib`:

```bash
python reproduce/make_all.py
python reproduce/crosscheck.py
python reproduce/integrity.py --verify
```

`make_all.py` regenerates every table, number macro and figure; `crosscheck.py` re-derives 1,310
numbers independently and reports any mismatch; `integrity.py --verify` checks the seven protected
pre-registration and budget files against their recorded digests. No checkpoint and no GPU is
needed (the two tables that read a checkpoint are kept as committed unless `--with-checkpoints` is
given). The inputs that cannot be redistributed in full are read from `data/public_inputs/`; with
the original files present the same loaders read those instead (`PUBLIC_INPUTS=1` forces the
extracts), and both paths give byte-identical outputs.

**Figure 2 is the one exception.** It shows Euclid cutouts, so it needs the pixels: the Discovery
Engine cutout archives (`lens.zip`, `group.zip`, `recenter.zip`, `unsuccess.zip`,
doi:10.5281/zenodo.15025832, processed by `experiments/01_data/build_lens_cutouts.py`), the
grade-X cutouts and the simulated lenses, cut from the Q1 MER mosaics in the ESA Euclid Science
Archive (`experiments/01_data/02_download_cutouts.py`, `07_download_ml_fps.py`,
`experiments/02_simulation/generate_arc_on_real.py`). Without them `make_all.py` keeps the
committed `paper/figures/F2_cutouts.pdf`.

## The hashed files

These files are kept byte for byte and at their original paths; `integrity.py` and Table A2 check
them.

| File | What it is |
|---|---|
| `runs/protected_artifacts.json` | the integrity inventory |
| `runs/dann_r5_prereg.json`, `runs/dann_r5_budget.json` | DANN × nb3 pre-registration and budget |
| `runs/coral_r4_budget.json` | CORAL × nb3 budget |
| `runs/lambda0_r6_prereg.json` | λ = 0 control pre-registration |
| `runs/bnm_r6_prereg.json`, `runs/bnm_r6_budget.json` | BNM × nb3 pre-registration and budget |
| `runs/bnm_r6_step0_gradnorms_t4fp16_REMEASURED.json` | BNM step-0 gradient-norm measurement (hashed) |
| `runs/round7/window_prereg.json` | window-averaging declaration (self-certifying) |
| `runs/coral_r4_gamma_selection.json` | CORAL γ selection (self-certifying) |
| `audit/decontam/rule.json` | the exact-ID exclusion rule (self-certifying) |
| `evidence/prefold/dann_r5_prereg_kaggle_v2_20260902.json` | pre-fold copy of the DANN pre-registration |
| `data/public_inputs/record_extract.json` | the project record's hash table, which `integrity.py` checks against |

The files these records cite by path are kept at those paths too.

### Scripts the hashed records cite by their old path

The hashed records cite these scripts as `scripts/…`, the original flat layout. They cannot be
edited, so here is where each one is now:

| Cited as | Now at |
|---|---|
| `scripts/bnm_round6.py` | `experiments/03_training/bnm_round6.py` |
| `scripts/run_bnm_r6.py` | `experiments/03_training/run_bnm_r6.py` |
| `scripts/da_losses.py` | `experiments/03_training/da_losses.py` |
| `scripts/run_dann_r5.py` | `experiments/03_training/run_dann_r5.py` |
| `scripts/integrity.py` | `reproduce/integrity.py` |
| `scripts/exact_ids.py` (`contaminant_sets`) | `reproduce/exact_ids.py` |
| `scripts/test_da_bnm.py` | `tests/test_da_bnm.py` |
| `scripts/test_bnm_round6.py` | not in this release (it resolves `runs/` from the original flat `scripts/` layout); kept in the project |
| `scripts/test_da_dann.py` | not in this release (it needs model weights and the original layout); kept in the project |

## About `experiments/`

The files in `experiments/` are archived as they were run. Their paths, `sys.path` handling and
imports refer to the original flat `scripts/` layout of the project (the modules they import by
bare name are now spread over `reproduce/` and the five stage folders), and they need the Euclid
data and, for training, a GPU. They are here so every reported run can be traced to the code that
produced it; they are not part of the reproduction above. Development code that produced no
reported result (connectivity checks, early probes, timing benchmarks, superseded split and
cutout scripts) is omitted.

## Identifiers

Every Euclid object ID in this folder is the **exact** Q1 `object_id`. The Q1 ID encodes position,
`id = sign(Dec) × (floor(RA·10⁷)·10⁹ + floor(|Dec|·10⁷))`, and each exported ID was asserted equal
to the rebuild from that object's own coordinates (`data/ids/ID_PROVENANCE.json`,
`experiments/05_audit/export_ids.py`).

| File | Rows | Source of the exact ID |
|---|---:|---|
| `data/ids/negatives_random.csv` | 50,299 | as stored in `splits.json`; rebuilt from the MER RA/Dec and asserted equal |
| `data/ids/hard_negatives_gradeX.csv` | 4,455 | Discovery Engine v0.0.1 `id_str`; rebuilt from its RA/Dec and asserted equal |
| `data/ids/positives.csv` | 489 | Discovery Engine v0.0.3 `id_str`; rebuilt from its RA/Dec and asserted equal |
| `data/ids/deflectors.csv` | 3,601 | rebuilt from RA/Dec of the deflector sample (the stored IDs were float-rounded and are not released) |
| `data/ids/positional_id_map.csv` | 40,453 | every positional ID in the released score files, mapped to its exact ID |

## Data sources

- **Euclid Q1:** European Space Agency, Euclid Q1 data, doi:[10.57780/esa-2853f3b](https://doi.org/10.57780/esa-2853f3b).
  Licence CC BY-NC 3.0 IGO. Q1 overview: Euclid Collaboration: Aussel et al. 2026, A&A 711, A1,
  doi:10.1051/0004-6361/202554610.
  - **MER** catalogue and mosaics: Euclid Collaboration: Romelli et al. 2026, A&A 711, A4,
    doi:10.1051/0004-6361/202554586.
  - **PHZ** photometric redshifts and stellar masses: Euclid Collaboration: Tucci et al. 2026,
    A&A 711, A5, doi:10.1051/0004-6361/202554588.
- **Discovery Engine lens catalogue and cutouts, v0.0.3** (the positives):
  doi:[10.5281/zenodo.15025832](https://doi.org/10.5281/zenodo.15025832), CC-BY-4.0.
  Paper: Euclid Collaboration: Walmsley et al. 2026, A&A 711, A26.
- **Discovery Engine catalogue, v0.0.1** (the grade-X hard negatives; later versions publish grades
  A–C only): doi:[10.5281/zenodo.15003117](https://doi.org/10.5281/zenodo.15003117), CC-BY-4.0.

None of these are redistributed in full. `data/public_inputs/` holds only the columns and rows the
build reads: the Discovery Engine extracts are CC-BY-4.0 (credit: Walmsley et al., the records
above); the MER, PHZ and positional extracts are Euclid Q1 data, CC BY-NC 3.0 IGO.

**Credit and acknowledgement.** Images: ESA Euclid/Euclid Consortium/NASA/Q1-2025.
"This work has made use of the Euclid Q1 data from the Euclid mission of the European Space Agency
(ESA), 2025, https://doi.org/10.57780/esa-2853f3b." "This research makes use of ESA Datalabs
(datalabs.esa.int), an initiative by ESA's Data Science and Archives Division in the Science and
Operations Department, Directorate of Science."

**Input stretch.** `zoobot_stretch` in `experiments/03_training/lens_dataset.py` is ported from
`fits_to_jpg_like_scaling` in the **galaxy-datasets** package by Mike Walmsley (GPL-3.0), with a
constant-image guard added.

## Kaggle placeholders

Several runs ran on Kaggle. In this copy the account name is replaced by `KAGGLE_OWNER`, and the
dataset slugs by `dataset-coral-r4`, `dataset-dann-r5`, `dataset-freematch-nb3`,
`dataset-lens-data` and `dataset-lens-paper`. `/kaggle/input` and `/kaggle/working` are Kaggle's
own paths. Absolute local paths were replaced by paths relative to the repository root (`.`).

## Known issues

- **`experiments/02_simulation/generate_arc_on_real.py` docstring is stale.** It says the source
  redshift is drawn so that θ_E is in [0.5″, 2.0″]; the code calls `pair_source_natural`
  (θ_E ≥ 0.75″, no upper cap). It also says "real HST/COSMOS source" and "Euclid VIS PSF"; the
  code draws parametric COSMOS profiles and convolves with a Gaussian PSF of FWHM 0.16″.
- **Rounded-ID comparisons.** The original split audit and leakage checks compared object IDs as
  the files stored them, and the deflector IDs were stored float-rounded. They are superseded by
  the exact-ID exclusion in `audit/decontam/rule.json`, which defines the test pool the paper
  reports.
- **The deflector selection query was not archived.** The deflectors were selected on ESA Datalabs;
  what is recoverable is the stored values (log M* ≥ 10.7, 0.10 ≤ z_phot ≤ 0.79), and
  `data/ids/deflectors.csv` defines the sample.
- **The screening cut is in PSF flux.** The unlabelled pool was selected by VIS PSF-fit flux between
  2.3 and 229 µJy (I_E ≈ 23.0 to 18.0 in PSF magnitude), not by a total-magnitude cut.
- **The parent pool is the first 250,000 rows of an unordered query.** The MER query
  (`experiments/01_data/01_query_catalog.py`) has `SELECT TOP 250000` and no `ORDER BY`, and the
  cap was reached.
- **Three committed tables keep an older header comment.** `paper/tables/A5_gate.tex`,
  `M_batch.tex` and `M_restart.tex` read a checkpoint, so they are not regenerated by default; their
  "% Rebuild" comment still names the old `paper/scripts/` path.
- **`reproduce/release_check.py`** describes the project's layout in its root lists; in this
  folder use it as `python reproduce/release_check.py --repo .` (pixel scan) or `--self-test`.
"""

if __name__ == "__main__":
    raise SystemExit(main())
