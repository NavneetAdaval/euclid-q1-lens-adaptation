r"""
make_all.py -- rebuild every table and every figure, then cross-check against PROJECT.md.
=========================================================================================

    python paper/scripts/make_all.py

Order matters in exactly one place: ``build_round1_comparisons.py`` must have run before
``make_t3_round1.py``, because T3's paired column reads the JSON it produces. That step
is slow (a 1000-replicate class-conditional paired bootstrap over five folds, three
times) so it is skipped when its outputs already exist; pass ``--force-round1`` to redo
it, or run it directly.

NOTHING HERE TRAINS ANYTHING OR TOUCHES A GPU. Every script reads saved logits, saved
score dumps, cached frozen features and JSON artifacts. The heaviest thing that happens
is a paired bootstrap over stored score vectors; the frozen baseline is read from its
stored step-8000 scores.

THE TEST POOL. Every score-based number is computed on the pool ``_common.POOL`` selects
(``PAPER_POOL``, default D1: audit/decontam/rule.json). ``build_pool_recomputes.py`` first
recomputes the pool-dependent JSON artifacts on that pool, after checking that each one
reproduces its original on the pre-registered pool.

The last step is ``crosscheck.py``, which diffs every generated number against what
PROJECT.md says and reports mismatches instead of silently preferring one. It returns
non-zero on a mismatch; this driver reports that as the finding it is and does not hide
it, but also does not treat it as a build failure, because a genuine disagreement between
the document and the artifacts is something to write down rather than something to fix by
rerunning.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
REPO = os.path.dirname(PAPER)
# Reproducible figure PDFs: matplotlib writes SOURCE_DATE_EPOCH as the CreationDate instead
# of the build time, so two builds give byte-identical figures. Every generator inherits it.
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

TABLES = ["make_t1_main.py", "make_t2_contrasts.py", "make_t3_round1.py",
          "make_t4_secondary.py", "make_t5_perfold.py", "make_t6_window.py",
          # rewrite v3: make_a3_orderings.py (protected, reads runs/fold_average.json) no longer
          # runs here; its committed outputs stay, and verify_orig_tables.py still regenerates
          # and checks them. Table 11 comes from make_orderings_v3.py below.
          "make_a2_hashes.py", "make_m_sim_zl.py",
          "make_a4_augprec.py", "make_a5_restart.py", "make_m_venue.py",
          "make_m_batch.py", "make_m_sim_source.py",
          "make_m_zoobot2.py", "make_a7_prereg_pool.py",
          "make_m_prose.py",
          # rewrite v1: review-round macros, main-text display variants, compact tables,
          # appendix tables, and the arXiv abstract (last: it reads every M_*.tex)
          "make_review_macros.py", "make_m_display.py", "make_main_tables_v2.py",
          "make_appendix_tables_v2.py", "make_appendix_relabel_v2.py",
          # rewrite v2: the values it adds to the text (before the abstract reads M_*.tex)
          "make_rewrite_v2_macros.py",
          # rewrite v3: Table 11 and \contrastCount, without runs/fold_average.json
          "make_orderings_v3.py",
          # rewrite v4: the above-threshold count and the D2 lens count as a word
          "make_rewrite_v4_macros.py",
          # rewrite v5: the FreeMatch and CORAL head starts (Section 6.3)
          "make_rewrite_v5_macros.py",
          # rewrite v5 addendum 1: Appendix C's k=1 / k=8 pairs table and its macros
          "make_k8_pairs.py",
          # 7.65: the pre-registration fixes' numbers and Appendix C's window secondaries
          "make_prereg_fixes.py",
          # 7.66: the code-review fixes' numbers (augmentation, method details, provenance)
          "make_code_review_fixes.py",
          # final build: the cross-check total the acknowledgements quote (PROJECTv7 7.61)
          "make_final_macros.py",
          "make_arxiv_abstract.py"]
# These two open a checkpoint file (checkpoint keys; a batch size recorded only there).
# Their outputs do not depend on the test pool, so a default rebuild keeps the committed
# files and opens no checkpoint; pass --with-checkpoints to rerun them.
# (make_a4_augprec.py opens one only when given --count-from-checkpoint.)
OPENS_CHECKPOINTS = {"make_a5_restart.py", "make_m_batch.py"}
# Figure 2 shows Euclid cutouts, so it needs the pixel files themselves. Where they are not
# present (the public release) the committed PDF is kept, exactly as for the
# checkpoint-reading tables above.
NEEDS_PIXELS = {"fig_f2_cutouts.py": [os.path.join("data", "sim_arc_on_real"),
                                      os.path.join("data", "q1_lenses_processed", "lens_cutouts.npz"),
                                      os.path.join("data", "ml_fp_cutouts", "shards")],
                "fig_fy_lens_gallery.py": [os.path.join("data", "q1_lenses_processed",
                                                        "lens_cutouts.npz")]}
FIGURES = ["fig_f1_setup.py", "fig_f2_cutouts.py", "fig_f3_forest.py",
           "fig_f4_fold0_curves.py", "fig_f5_resolution.py",
           "fig_f6_completeness_purity.py",
           # rewrite v1
           "fig_f1_setup_v2.py", "fig_f3_forest_v2.py", "make_F5_resolution_v2.py",
           "fig_fy_lens_gallery.py",
           # rewrite v2
           "fig_f6_completeness_purity_v2.py"]


def run(script, *args, check=True):
    print(f"\n=== {script} " + "=" * (72 - len(script)))
    r = subprocess.run([sys.executable, "-u", os.path.join(HERE, script), *args],
                       cwd=REPO)
    if check and r.returncode != 0:
        raise SystemExit(f"{script} failed with exit code {r.returncode}")
    return r.returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force-round1", action="store_true",
                    help="recompute the slow round-1 paired bootstraps")
    ap.add_argument("--with-checkpoints", action="store_true",
                    help="also rerun the generators that open checkpoint files")
    a = ap.parse_args()

    run("build_round1_comparisons.py", *(["--force"] if a.force_round1 else []))
    run("build_pool_recomputes.py")
    for s in TABLES:
        if s in OPENS_CHECKPOINTS and not a.with_checkpoints:
            print(f"\n=== {s}: kept as committed (opens a checkpoint; pool-independent)")
            continue
        run(s)
    for s in FIGURES:
        missing = [p for p in NEEDS_PIXELS.get(s, []) if not os.path.exists(os.path.join(REPO, p))]
        if missing:
            print(f"\n=== {s}: kept as committed (needs the Euclid cutouts: {', '.join(missing)})")
            continue
        run(s)
    rc = run("crosscheck.py", check=False)

    print("\n" + "=" * 80)
    if rc == 0:
        print("  ALL TABLES AND FIGURES REBUILT. No number disagrees with PROJECT.md.")
    else:
        print("  ALL TABLES AND FIGURES REBUILT. crosscheck.py reports at least one")
        print("  disagreement with PROJECT.md -- see the MISMATCH lines above and")
        print("  paper/STAGE1_REPORT.md. That is a finding, not a build failure.")
    print("=" * 80)
    run("make_v6.py")   # paper v6: the 1:10 generators, then crosscheck.py again (PROJECTv7 7.56)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
