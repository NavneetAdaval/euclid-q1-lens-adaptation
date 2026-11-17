r"""
make_m_batch.py -- macros for Section 4's per-step batch composition.
=====================================================================

    python paper/scripts/make_m_batch.py

Reads, and asserts before emitting:

  * every arm's recorded ``batch_size`` (and ``pos_fraction``), from each fold's own run
    files as listed in ``audit_venue.ARMS``; the round-1 DANN probe records its
    configuration only inside its checkpoints, so it is read from a checkpoint's ``cfg``;
  * that every adaptation runner builds its TARGET sampler with the same batch size as
    its source sampler (a static check on the runner source: the target sampler/loader
    call passes the runner's batch-size variable). No runner has an unlabelled-ratio
    multiplier: FreeMatch's and AdaMatch's ``mu`` in this code base is the consistency
    weight, not a ratio of unlabelled to labelled rows.

Every step of every arm therefore carries one labelled source batch of \batchSourceRows{}
rows; every adaptation arm and the lambda = 0 control carry \batchTargetRows{} unlabelled
target rows beside it (FreeMatch and AdaMatch see each target row in a weak and a strong
view); Baseline A has no target batch.

Writes ``paper/tables/M_batch.tex``.
"""
from __future__ import annotations

import glob
import json
import os
import re

import _common as C
import audit_venue as AV

KEYS = ("batch_size", "pos_fraction")

# runner -> regex that must match the TARGET sampler/loader call, with the runner's own
# batch-size variable (the same one its source sampler uses)
TARGET_CALLS = {
    "run_coral.py":         r"tgt_sampler = ClassBalancedBatchSampler\(\s*tpos_idx, tneg_idx, batch_size=batch_size",
    "run_dann.py":          r"tgt_sampler = ClassBalancedBatchSampler\(\s*tpos_idx, tneg_idx, batch_size=batch_size",
    "run_adamatch.py":      r"tgt_loader = make_balanced_loader\(tgt_ds, batch_size,",
    "run_freematch.py":     r"tgt_loader = make_balanced_loader\(tgt_ds, batch_size,",
    "run_coral_round4.py":  r"tgt_loader = make_balanced_loader\(tgt_ds, batch_size,",
    "run_dann_r5.py":       r"tgt_sampler = ClassBalancedBatchSampler\(holdout\[\"train_pos\"\], holdout\[\"train_neg\"\],\s*batch_size,",
    "run_bnm_r6.py":        r"tgt_sampler = ClassBalancedBatchSampler\(holdout\[\"train_pos\"\], holdout\[\"train_neg\"\],\s*BATCH_SIZE,",
}


def _recorded(d: str) -> dict:
    found = {k: set() for k in KEYS}
    for p in glob.glob(os.path.join(d, "**", "*.json"), recursive=True):
        if os.path.getsize(p) > 2_000_000:
            continue
        try:
            obj = json.load(open(p, encoding="utf-8"))
        except (ValueError, UnicodeDecodeError):
            continue
        stack = [obj]
        while stack:
            o = stack.pop()
            if isinstance(o, dict):
                for k, v in o.items():
                    if k in found and not isinstance(v, (dict, list)):
                        found[k].add(v)
                    stack.append(v)
            elif isinstance(o, list):
                stack.extend(o)
    if not found["batch_size"]:                      # the round-1 DANN probe
        import torch
        ck = sorted(glob.glob(os.path.join(d, "checkpoints", "step_*.pt")))
        assert ck, f"no run file or checkpoint records a batch size under {d}"
        cfg = torch.load(ck[0], map_location="cpu", weights_only=False)["cfg"]
        for k in KEYS:
            found[k].add(cfg[k])
    return found


def main() -> int:
    batch, posf = set(), set()
    for arm, folds, dir_of, _ in AV.ARMS:
        for f in folds:
            rec = _recorded(dir_of(f))
            assert len(rec["batch_size"]) == 1, (arm, f, rec)
            batch |= rec["batch_size"]
            posf |= rec["pos_fraction"]
    assert len(batch) == 1 and len(posf) == 1, (batch, posf)
    bs, pf = int(batch.pop()), float(posf.pop())

    for runner, pat in TARGET_CALLS.items():
        src = open(os.path.join(C.REPO_SCRIPTS, runner), encoding="utf-8").read()
        assert re.search(pat, src), f"{runner}: target sampler not built at the source batch size"
        assert not re.search(r"\buratio\b|unlabeled_ratio", src), f"{runner}: unlabelled ratio"

    macros = "\n".join([
        rf"\newcommand{{\batchSourceRows}}{{{bs}}}",
        rf"\newcommand{{\batchTargetRows}}{{{bs}}}",
        rf"\newcommand{{\batchSourcePositives}}{{{round(bs * pf)}}}",
    ])
    C.write_table(os.path.join(C.TABLES, "M_batch.tex"), macros,
                  header_comment="Macros for Section 4's batch-composition sentence; sources "
                                 "in paper/scripts/make_m_batch.py.", size="")
    print(f"  every arm: {bs} labelled source rows per step (pos_fraction {pf}); "
          f"{bs} unlabelled target rows per step in every adaptation arm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
