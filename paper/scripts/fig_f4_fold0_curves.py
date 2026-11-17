r"""
fig_f4_fold0_curves.py -- F4, the fold-0 controlled comparison.
===============================================================

    python paper/scripts/fig_f4_fold0_curves.py

The $\lambda=0$ control against DANN over the 32 matched checkpoints of the round-5
development fold. **These two arms differ in $\lambda$ and in nothing else** -- same
fold, same seed, same initialisation, same GPU, same 8000 steps, bit-identical at step 0
-- which makes this the cleanest controlled comparison anywhere in the project, and the
only place the paper can show the effect of a target-domain objective with every other
variable held.

TWO THINGS ARE MARKED, AND BOTH ARE COUNTED FROM THE CURVES RATHER THAN ASSERTED:

  * **how many of the matched steps DANN sits above the control** -- the script counts
    them and writes the count into the annotation, so the figure cannot claim 32 of 32
    if the artifact says otherwise;
  * **the frozen LP$\rightarrow$FT head both arms initialise from.** The control ends
    *below* it: three encoder blocks unfrozen with a source-only objective make the model
    worse at finding real lenses than the frozen linear probe it started from. That line
    is Baseline A's own fold-0 value, recomputed from the frozen feature cache, not a
    number typed in.

**FOLD 0 IS THE DEVELOPMENT FOLD.** Nothing here enters a headline, no budget is read off
it, and the confirmatory $\lambda=0$ arm ran on folds 1--4 while DANN's control did not
-- so the *size* of this gap rests on one fold and the figure's caption has to say so.

Writes ``paper/figures/F4_fold0_curves.pdf`` and nothing else.
"""
from __future__ import annotations

import csv
import json
import math
import os

import _common as C


def read_curve(path):
    steps, tpr = [], []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            steps.append(int(row["step"]))
            tpr.append(float(row["tpr_at_1e-3"]))
    return steps, tpr


def main() -> int:
    plt = C.use_paper_style()

    if C.POOL == "orig":
        s1, t1 = read_curve(C.ART["dann_arm1_curve"])
        s2, t2 = read_curve(C.ART["dann_arm2_curve"])
    else:
        # recomputed from the rescored per-checkpoint logits on this pool; on the
        # pre-registered pool they equal the stored curves (build_pool_recomputes.py)
        d = json.load(open(os.path.join(C.DATA, f"fold0_curves_{C.POOL}.json"),
                           encoding="utf-8"))["curves"]
        s1 = [int(s) for s in d["arm1_lambda0"]]
        t1 = [d["arm1_lambda0"][str(s)] for s in s1]
        s2 = [int(s) for s in d["arm2_dann"]]
        t2 = [d["arm2_dann"][str(s)] for s in s2]
    assert s1 == s2, "the two arms are not on the same step grid"

    init = float(C.suite("baseline", 0, 8000)["tpr"])   # the shared LP->FT head, same pool
    above = sum(1 for a, b in zip(t2, t1) if a > b)
    below_init = sum(1 for v in t1 if v < init)

    fig, ax = plt.subplots(figsize=(6.6, 2.95))
    O = C.OKABE_ITO

    ax.fill_between(s1, t1, t2, where=[b > a for a, b in zip(t1, t2)],
                    color=O["vermillion"], alpha=0.10, lw=0, zorder=1)

    ax.axhline(init, color="#555555", lw=1.0, linestyle=(0, (5, 3)), zorder=2)
    ax.annotate(f"frozen LP$\\rightarrow$FT head both arms start from  ({init:.4f})",
                xy=(8000, init), xytext=(7900, init + 0.018), ha="right", va="bottom",
                fontsize=7.0, color="#444444")

    ax.plot(s2, t2, color=O["vermillion"], marker="o", ms=2.8, lw=1.3, zorder=4,
            label=r"DANN ($\lambda$ live)")
    ax.plot(s1, t1, color="#000000", marker="D", ms=2.6, lw=1.3, zorder=4,
            label="control ($\\lambda{=}0$) — unfreezing alone")

    # the vertical range follows the curves (unchanged at 0.66 where they stay below 0.60)
    top = 0.66 if max(t2) <= 0.60 else math.ceil((max(t2) + 0.08) * 50) / 50
    ax.text(4900, top - 0.043, f"DANN above the control at {above} of {len(s1)} matched steps",
            fontsize=7.4, color=O["vermillion"], fontweight="bold", ha="center",
            va="center")
    ax.text(8050, 0.008,
            f"the control ends {'below' if t1[-1] < init else 'at or above'} its own "
            f"initialisation ({t1[-1]:.4f} against {init:.4f}) and is below it at "
            f"{below_init} of {len(s1)} steps",
            ha="right", va="bottom", fontsize=7.0, color="#333333")

    ax.set_xlabel("training step (fold 0, development)")
    ax.set_ylabel(r"TPR at FPR${=}10^{-3}$")
    ax.set_xlim(0, 8300)
    ax.set_ylim(0.0, top)
    ax.grid(color="#EEEEEE", lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, fontsize=7.2, handlelength=2.0,
              borderaxespad=0.3)

    C.save_fig(fig, os.path.join(C.FIGURES, "F4_fold0_curves.pdf"))
    plt.close(fig)

    print(f"  DANN above control at {above}/{len(s1)} matched steps; "
          f"control below its init at {below_init}/{len(s1)}; "
          f"control last = {t1[-1]:.4f}, DANN last = {t2[-1]:.4f}, init = {init:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
