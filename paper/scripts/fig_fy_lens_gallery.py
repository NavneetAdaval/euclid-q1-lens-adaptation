r"""
fig_fy_lens_gallery.py -- the found-and-missed lens gallery (rewrite v1, Section 6.6).
====================================================================================

    python paper/scripts/fig_fy_lens_gallery.py

Two rows of eight confirmatory test lenses:

  * top: recovered by all four adaptation arms at their headline readouts;
  * bottom: missed by every arm (the four adaptation arms, Baseline A and the lambda=0 control).

The sets and their order come from audit/review_r1/out/A3_galleries.json (each set sorted by
exact Euclid object ID; the first eight are drawn), read through make_review_macros.review,
which re-checks every recorded input hash. Each panel is the stored cutout
(data/q1_lenses_processed/lens_cutouts.npz, read only) under the encoder's own input stretch,
``lens_dataset.zoobot_stretch``, as in Figure 2, and is labelled with its grade and expert
score. Writes ``paper/figures/FY_lens_gallery.pdf`` and nothing else.
"""
from __future__ import annotations

import os

import numpy as np

import _common as C
import make_review_macros as RV

SETS = [("a_recovered_by_all_four_DA_arms", "recovered by all four adaptation arms"),
        ("b_missed_by_every_arm", "missed by every arm")]
LABEL = {"a_recovered_by_all_four_DA_arms": "found by all\nfour adaptation\narms",
         "b_missed_by_every_arm": "missed by\nall six\nmodels"}


def main() -> int:
    plt = C.use_paper_style()
    import torch
    from lens_dataset import zoobot_stretch
    gal = RV.review(RV.R1, "A3_galleries.json")["galleries"]
    with np.load(C.R("data", "q1_lenses_processed", "lens_cutouts.npz"), allow_pickle=True) as z:
        imgs = np.asarray(z["images"])
        ids = [str(s) for s in z["id_str"]]
        grade = np.asarray(z["grade"]).astype(str)
        score = np.asarray(z["expert_score"], float)
    fig, axes = plt.subplots(2, 8, figsize=(6.6, 2.15))
    for r, (key, title) in enumerate(SETS):
        shown = gal[key]["shown"][:8]
        assert len(shown) == 8
        assert [x["euclid_id"] for x in shown] == sorted(x["euclid_id"] for x in shown)
        for c, x in enumerate(shown):
            row = x["lens_row"]
            assert ids[row] == x["id_str"] and grade[row] == x["grade"]
            t = zoobot_stretch(torch.from_numpy(imgs[row].astype(np.float32))[None, ...])[0].numpy()
            ax = axes[r, c]
            ax.imshow(t, cmap="gray", origin="lower", interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{grade[row]}, {score[row]:.2f}", fontsize=6.4, pad=1.5)
        axes[r, 0].set_ylabel(LABEL[key], fontsize=6.2, labelpad=3)
    fig.subplots_adjust(left=0.085, right=0.995, top=0.9, bottom=0.02, wspace=0.06, hspace=0.32)
    C.save_fig(fig, os.path.join(C.FIGURES, "FY_lens_gallery.pdf"))
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
