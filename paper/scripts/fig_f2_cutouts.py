r"""
fig_f2_cutouts.py -- F2, one example of each image population.
==============================================================

    python paper/scripts/fig_f2_cutouts.py

Four panels: an arc-on-real simulated positive, **the same deflector galaxy without the
arc** (the matched bare-deflector negative), a real grade-A Euclid lens, and a grade-X
hard negative (version 0.0.1 of the Discovery Engine catalogue).

THE SELECTION RULE IS DECLARED AND DETERMINISTIC, because "a representative example" is
otherwise an invitation to pick the prettiest one:

  * **simulated positive** -- from the first shard, the lens whose Einstein radius is
    closest to that shard's *median* $\theta_E$. Not the largest, not the brightest arc;
  * **bare-deflector negative** -- the matched non-lens for that same index, i.e.
    literally the same real Euclid galaxy with no arc painted on. Showing them side by
    side is the point: it is what forces the classifier onto the arc rather than onto the
    host galaxy, and it is why an earlier fully-synthetic generator's brightness shortcut
    could reach AUROC 0.995 on nothing at all;
  * **real lens** -- among the grade-A lenses, the one with the *median* expert score;
  * **grade-X hard negative** -- the first object in the first shard by sorted filename.

**Every panel uses the same stretch** -- ``lens_dataset.zoobot_stretch``, the per-image
arcsinh the model itself sees (percentile clip at 0/99.7, rescale to [0,1], arcsinh with
$q=1$). It is applied per image, which is what the encoder receives, so the panels are
comparable as *inputs to the model* rather than in absolute flux. The caption says so.

Writes ``paper/figures/F2_cutouts.pdf`` and nothing else.
"""
from __future__ import annotations

import glob
import os

import numpy as np

import _common as C


def main() -> int:
    plt = C.use_paper_style()
    import torch
    from lens_dataset import zoobot_stretch

    shard = sorted(glob.glob(C.R("data", "sim_arc_on_real", "shard_*.npz")))[0]
    z = np.load(shard)
    theta = np.asarray(z["theta_E"], float)
    idx = int(np.argmin(np.abs(theta - np.median(theta))))
    sim_pos = np.asarray(z["lenses"][idx], np.float32)
    sim_neg = np.asarray(z["nonlenses"][idx], np.float32)
    obj = z["object_id"][idx]

    zl = np.load(C.R("data", "q1_lenses_processed", "lens_cutouts.npz"))
    grade = np.asarray(zl["grade"])
    score = np.asarray(zl["expert_score"], float)
    a_rows = np.flatnonzero(grade == "A")
    med_row = int(a_rows[np.argsort(score[a_rows])[len(a_rows) // 2]])
    real = np.asarray(zl["images"][med_row], np.float32)

    mlfp_file = sorted(glob.glob(C.R("data", "ml_fp_cutouts", "shards", "*.npz")))[0]
    zm = np.load(mlfp_file)
    mlfp = np.asarray(zm["images"][0], np.float32)

    panels = [
        (sim_pos, "simulated lens\n(arc on a real galaxy)", C.OKABE_ITO["blue"], "positive"),
        (sim_neg, "the SAME galaxy,\nno arc", C.OKABE_ITO["blue"], "negative"),
        (real, "real Euclid lens\n(grade A)", C.OKABE_ITO["green"], "positive"),
        (mlfp, "expert grade X\n(not a lens)", C.OKABE_ITO["vermillion"], "negative"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(6.6, 2.02))
    for ax, (img, title, colour, kind) in zip(axes, panels):
        t = zoobot_stretch(torch.from_numpy(img).float()[None, ...])[0].numpy()
        ax.imshow(t, cmap="gray", origin="lower", interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color(colour)
            sp.set_linewidth(1.6)
        ax.set_title(title, fontsize=7.2, color=colour, pad=3.5)
        ax.text(0.5, -0.055, kind, transform=ax.transAxes, ha="center", va="top",
                fontsize=6.8, color="#333333",
                fontweight="bold" if kind == "positive" else "normal")

    fig.subplots_adjust(wspace=0.10)
    C.save_fig(fig, os.path.join(C.FIGURES, "F2_cutouts.pdf"))
    plt.close(fig)

    print(f"  sim index {idx} of shard {os.path.basename(shard)}, "
          f"theta_E = {theta[idx]:.3f}\" (shard median {np.median(theta):.3f}\"), "
          f"deflector object_id {obj}")
    print(f"  real lens row {med_row}, grade A, expert score {score[med_row]:.3f} "
          f"(median of {a_rows.size} grade-A lenses)")
    print(f"  ML-FP: first image of {os.path.basename(mlfp_file)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
