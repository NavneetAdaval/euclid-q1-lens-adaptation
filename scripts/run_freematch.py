"""
run_freematch.py
================
FreeMatch -- the THIRD real domain-adaptation method in the bake-off (after Deep CORAL
and AdaMatch). It reuses AdaMatch's scaffolding VERBATIM: image-space training with the
frozen Zoobot encoder in the loop (encoder under no_grad, only the head trains), reading
weak+strong views from LensDataset(n_views=2); ONE MODEL PER FOLD (the target rotates per
fold; PROJECT.md 4.3); run_coral's per-fold evaluator + the compare_methods metric suite;
and the ConfirmationBiasDetector. So FreeMatch is directly comparable to Baseline A /
CORAL / AdaMatch.

WHAT IS DIFFERENT FROM ADAMATCH (the method, not the plumbing)
-------------------------------------------------------------
FreeMatch replaces AdaMatch's source-referenced RELATIVE confidence threshold with a
SELF-ADAPTIVE THRESHOLD estimated purely from the model's own confidence on the unlabeled
target (a global EMA tau_t + a per-class local tau_t(c)), and ADDS a self-adaptive
fairness (SAF) penalty pushing the confident-sample marginal toward a balanced target --
the anti-collapse term AdaMatch lacked. See da_losses.FreeMatchLoss.

  ****  THE MAKE-OR-BREAK KNOB (run this as a first-class experiment)  ****
  FreeMatch's local threshold LOWERS the bar for the under-learned class -- here "lens"
  (rare positives) -- which risks the exact confirmation-bias collapse AdaMatch hit. So
  ``--local-threshold-mode`` is a real experiment, not a detail:
    * "standard": the paper's tau_t(lens) = (pbar(lens)/max_c pbar(c)) * tau_t.
    * "floor":    tau_t(lens) = max(tau_t(lens), tau_t) -- never let the lens bar drop
                  below the global threshold (blocks weak lens pseudo-labels).
  The fold-0 screen runs BOTH and compares acc_pos_frac + TPR@1e-3.

FIXED-BUDGET PROTOCOL (PROJECT.md 7.4)
--------------------------------------
mu_max (peak consistency weight, 1.0), saf_weight (fairness weight, USB default 0.01),
sat_ema_momentum (0.999) and the balanced fairness target ([0.5,0.5]) are all FIXED config
values, NOT tuned on the target. The mu cosine warmup + the consistency-warmup gate are
fixed stability priors, not oracle knobs.

CONFIRMATION-BIAS DIAGNOSTIC (logged, never a selector)
-------------------------------------------------------
Every epoch we log the pseudo-label ACCEPTANCE RATE, the accepted-pseudo-label CLASS
BALANCE (acc_pos_frac), the current global tau_t and BOTH local tau_t(c), and the SAF
penalty value, and run the ConfirmationBiasDetector. The whole point is to SEE whether
SAT+SAF keep acc_pos_frac sane where AdaMatch collapsed.

  ****  LEAKAGE CONTRACT  ****
  Fold k trains on source_train + val (source-domain SIM) + target_fold{k} ONLY. Fold k's
  target pool by construction EXCLUDES fold k's own lenses (PROJECT.md 4.3), and the test
  folds are touched exclusively by the evaluator. No model is trained on a lens it is
  tested on. (The evaluator reuses the SAME leakage-guarded per-fold path as CORAL.)

Run (needs the real splits + the frozen encoder weights; eval needs the feature cache):
    python scripts\\run_freematch.py --data-root data --cache-dir data\\feature_cache \\
        --out-dir runs\\freematch --local-threshold-mode floor
    python scripts\\run_freematch.py --folds 0 --epochs 1,3,5 --local-threshold-mode standard
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

from heads import LINEAR, HEAD_TYPES
from lens_dataset import LensDataset, UNLABELED
from samplers import make_balanced_loader, PairedDomainLoader
from da_losses import ConfirmationBiasDetector, adamatch_mu_at_step, FREEMATCH_LOCAL_MODES
from train_skeleton import Trainer, TrainConfig
from metrics import metric_suite, DEFAULT_TARGET_FPR, DEFAULT_TARGET_COMPLETENESS, NN_NP_Q1
from run_baseline import parse_epochs, set_seed, _cf
import run_coral as rc
import evaluate as ev
import curve_hook


# ---------------------------------------------------------------------------
# checkpoint naming (train_skeleton format, fold+epoch recoverable)
# ---------------------------------------------------------------------------

def ckpt_name(head_type: str, fold: int, epoch: int, width: int) -> str:
    """e.g. freematch_linear_fold0_ep008.pt -- evaluate._epoch_from_name() recovers the
    epoch via '(?:ep|epoch)[_-]?(\\d+)'; the fold is also stored inside cfg."""
    return f"freematch_{head_type}_fold{fold}_ep{epoch:0{width}d}.pt"


# ---------------------------------------------------------------------------
# per-fold FreeMatch training (IMAGE-SPACE; frozen encoder in the loop)
# ---------------------------------------------------------------------------

def train_freematch_fold(splits_path, data_root, cache_dir, fold, ckpt_dir, *,
                         encoder=None, head_type=LINEAR, epochs=(1, 2, 5, 10, 20),
                         batch_size=256, pos_fraction=0.5, lr=1e-3, weight_decay=1e-4,
                         steps_per_epoch=None, mu_max=1.0, saf_weight=0.01,
                         sat_ema_momentum=0.999, consistency_warmup_frac=0.1,
                         local_threshold_mode="standard", pos_class=1,
                         fairness_target=None, cb_patience=None, seed=42, device="cpu",
                         num_workers=0, fp16=False, eval_every_steps=None, curve_csv=None,
                         curve_folds=None, agreement_refs=None,
                         eval_target_fpr=DEFAULT_TARGET_FPR, val_select_metric="auroc",
                         eval_device="cpu", verbose=True,
                         # ---- LEVER-1 x FREEMATCH additions (all default to the round-1
                         # code path, so existing callers are byte-for-byte unchanged) ----
                         n_blocks=0, backbone_lr=None, lr_decay=0.75, init_head_from=None,
                         grad_checkpointing=False, schedule_steps=None, max_steps=None,
                         ckpt_every=None, instrument=None, eval_at_step0=False,
                         step0_check=None,
                         # ---- CONTINUATION-STATE reconstruction (continuation_state.py) ----
                         mu_step_offset=0, rebuild_sat=False, sat_passes=40,
                         sat_ema_iters=5000, adam_warmup_steps=0,
                         adam_probe_steps=(50, 200, 500)):
    """Train the head on source (BCE on the WEAK view) + FreeMatch consistency + SAF for
    one fold, dumping epoch checkpoints. Reads images from LensDataset for source_train
    (n_views=1, weak only) + target_fold{fold} (n_views=2, weak+strong); fits the
    standardizer from the CACHED source features
    (so train/eval standardization is identical). Returns (checkpoints, stability_log).

    PARTIAL ENCODER UNFREEZING (``n_blocks`` > 0)
    ---------------------------------------------
    ``n_blocks`` unfreezes the top n_blocks of Zoobot's own tuneable_blocks and is passed
    straight through to TrainConfig, so the optimizer builds the layer-decayed LR groups
    (PROJECT.md §7.1a). Three consequences are handled HERE and nowhere else:

      * The encoder must be built with the matching ``n_blocks`` -- pass it in via
        ``encoder=``; TrainConfig refuses a mismatch rather than silently building LR
        groups for blocks that are not trainable.
      * The checkpoint must carry the trainable ENCODER DELTA, not just the head, or the
        run is unscorable. At n_blocks>0 we therefore save with ``Trainer.save`` (which
        stores ``encoder_trainable`` + ``encoder_mode_effective``); at n_blocks=0 the
        legacy ``run_coral.save_checkpoint`` path is kept so round-1 checkpoints stay
        byte-compatible.
      * The frozen FEATURE CACHE is invalid the moment the encoder moves (§7.1a: scoring a
        real nb3 checkpoint from the stale cache gave 0.194 against a logged 0.020 -- a
        different model, not a rounding error). The caller supplies a live-encoding
        ``instrument`` whose ``split_loader`` re-encodes the eval splits at every curve
        point; nothing here reads the cache except the standardizer fit.

    **The DA-loss x unfreezing interaction is deliberate, and it is new.**
    ``train_skeleton.encode_features`` flags it as an open item: at n_blocks>0 the encoder
    features carry grad, so the FreeMatch consistency + SAF terms backprop INTO the encoder
    rather than stopping at the head. That is precisely what this run measures, and it is
    still correct FixMatch semantics -- FreeMatchLoss computes the SAT update, the
    pseudo-labels and the mask inside ``torch.no_grad()``, so the weak-view branch cannot
    backprop and the pseudo-label is a constant, exactly as when the encoder was frozen.

    CONTINUATION-STATE RECONSTRUCTION (``mu_step_offset`` / ``rebuild_sat`` /
    ``adam_warmup_steps``)
    ------------------------------------------------------------------------
    A round-1 checkpoint stores the MODEL but not the training state around it, so a naive
    continuation restarts three things at once: the SAT threshold from 1/C, ``mu_eff`` from
    0 (the warmup withholds the target term entirely for the first 10% of the schedule),
    and AdamW's moments from zero. MEASURED on the fold-0 frozen control: a V-shaped
    0.3878 -> 0.2347 -> 0.3367 excursion, spread 0.102, bottoming inside the mu=0 window --
    ten to fifteen times the 1/98 resolution. See continuation_state.py for the evidence
    that this is an artifact rather than continued training.

    ``mu_step_offset`` is added to the step passed to the loss, so a continuation resuming
    at schedule position 8000 gets ``mu_eff`` at full value from its own step 0 and the
    consistency-warmup gate opens immediately. It shifts NOTHING else -- there is no LR
    scheduler in this training path (verified: no scheduler is ever constructed or stepped,
    and ZOOBOT_COSINE_SCHEDULE is False), so the learning rate is unaffected by the offset.

    ORDERING, which matters: head loaded -> SAT rebuilt -> Adam warmed -> step-0 gate ->
    train. The rebuild and the warm-up both run with the encoder frozen (the warm-up
    freezes it explicitly), so this whole phase is identical in the frozen and unfrozen
    arms and the only bit that differs between them remains ``n_blocks``. The step-0 gate
    runs AFTER both, which makes it a check that neither touched the head or standardizer:
    buffers and optimizer moments do not enter a forward-pass evaluation, so the gate value
    must be unchanged.

    SCHEDULE vs BUDGET (``schedule_steps`` / ``max_steps``)
    ------------------------------------------------------
    ``schedule_steps`` sets the span of the mu cosine warmup and the consistency-warmup
    gate; ``max_steps`` sets where training actually stops. They are separate so a SHORT
    control run can be a true TRUNCATION of the long run's schedule rather than a
    compressed copy of it: with schedule_steps=8000, max_steps=2000 the control's step 250
    sits at exactly the mu_eff the 8000-step run had at step 250. Defaults (both None)
    reproduce the round-1 behaviour, where the schedule spans the whole planned run.
    """
    if head_type not in HEAD_TYPES:
        raise ValueError(f"head_type must be one of {HEAD_TYPES}, got {head_type!r}")
    if local_threshold_mode not in FREEMATCH_LOCAL_MODES:
        raise ValueError(f"local_threshold_mode must be one of {FREEMATCH_LOCAL_MODES}, "
                         f"got {local_threshold_mode!r}")
    set_seed(seed + fold)                            # decorrelate folds' inits/shuffles
    dev = torch.device(device)

    # ---- datasets ----
    # SOURCE: n_views=1 (WEAK only). Unlike AdaMatch, FreeMatch's supervised BCE uses the
    # source WEAK view (FixMatch convention) and never a source strong view, so building the
    # source with n_views=2 would compute (and collate) a 256-image strong augmentation every
    # step that is thrown away -- ~1/3 of the per-step transform work for nothing. n_views=1
    # drops it (measured: it is pure waste, the strong view is not read anywhere downstream).
    # TARGET: n_views=2 (weak drives SAT+pseudo-labels; strong drives consistency + SAF).
    src_ds = LensDataset(splits_path, data_root, role="source_train",
                         aug="weak", n_views=1)
    tgt_ds = LensDataset(splits_path, data_root, role="target", fold=fold,
                         aug="weak", n_views=2)
    assert np.all(tgt_ds.labels == UNLABELED), "target labels must all be UNLABELED(-1)"

    schedule = sorted(set(int(e) for e in epochs))
    max_epoch = schedule[-1]
    width = max(2, len(str(max_epoch)))
    steps = steps_per_epoch or (len(src_ds) // batch_size)
    # planned run -> mu schedule span. ``schedule_steps`` decouples it from where training
    # stops (see the docstring) so a truncated control keeps the full run's schedule.
    total_steps = int(schedule_steps) if schedule_steps else steps * max_epoch
    stop_at = int(max_steps) if max_steps else None

    # ---- balanced loaders: SOURCE on real labels, TARGET on latent identity ----
    # The per-step cost is dominated by (a) 3 frozen-encoder forwards on the GPU (~1 s at
    # batch 256, the hard floor) and (b) the CPU per-image augmentation transforms. With
    # num_workers>0 the transforms run in parallel across processes and the loader PREFETCHES
    # the next batch during the GPU forward, so the CPU cost is HIDDEN behind the GPU work
    # (measured: step time goes from load-sensitive ~1.4-6 s down to a flat ~1.1 s, GPU-bound).
    # pin_memory speeds the host->device copy; persistent_workers avoids re-pickling the
    # ~0.8 GB source dataset to the workers every epoch.
    src_loader = make_balanced_loader(src_ds, batch_size, use_latent=False,
                                      num_batches=steps, pos_fraction=pos_fraction,
                                      seed=seed + fold, num_workers=num_workers,
                                      pin_memory=True, persistent_workers=True)
    tgt_loader = make_balanced_loader(tgt_ds, batch_size, use_latent=True,
                                      num_batches=steps, pos_fraction=pos_fraction,
                                      seed=seed + fold + 1000, num_workers=num_workers,
                                      pin_memory=True, persistent_workers=True)
    paired = PairedDomainLoader(src_loader, tgt_loader)

    # ---- Trainer with the frozen encoder + FreeMatch loss ----
    cfg = TrainConfig(
        splits_path=splits_path, data_root=data_root, fold=fold, head_type=head_type,
        encoder_mode="frozen", lr=lr, weight_decay=weight_decay, batch_size=batch_size,
        pos_fraction=pos_fraction, use_freematch=True, freematch_mu_max=mu_max,
        freematch_saf_weight=saf_weight, freematch_sat_ema_momentum=sat_ema_momentum,
        freematch_consistency_warmup_frac=consistency_warmup_frac,
        freematch_local_threshold_mode=local_threshold_mode, freematch_pos_class=pos_class,
        freematch_fairness_target=fairness_target, total_steps=total_steps,
        epochs=max_epoch, seed=seed + fold, device=device, num_workers=num_workers,
        amp=fp16,                                    # fp16 -> autocast the forwards only
        # LEVER 1: n_blocks=0 leaves every one of these inert and the config identical to
        # round 1 (encoder_mode stays the plumbed switch name; n_blocks is the orthogonal
        # unfreeze knob -- see ZoobotEncoder.effective_mode).
        n_blocks=int(n_blocks), backbone_lr=backbone_lr, lr_decay=lr_decay,
        init_head_from=init_head_from, grad_checkpointing=bool(grad_checkpointing))
    trainer = Trainer(cfg, encoder=encoder)
    use_fp16 = bool(fp16) and trainer.device.type == "cuda"

    # ---- standardizer: fit from the CACHED source features (train/eval consistency) ----
    cache = ev.FeatureCache(cache_dir)
    if not cache.has("source_train"):
        raise FileNotFoundError(
            f"source_train.npz not found under {cache_dir}; run cache_features.py first "
            f"(FreeMatch fits the standardizer from the cached source features).")
    src_feats = torch.from_numpy(
        np.asarray(cache.load("source_train")["features"], dtype=np.float32)).to(dev)
    trainer.standardizer.fit(src_feats)
    del src_feats

    warmup_steps = int(math.ceil(consistency_warmup_frac * total_steps))
    patience = int(cb_patience) if cb_patience else steps
    detector = ConfirmationBiasDetector(patience=patience, warmup_steps=warmup_steps)

    if verbose:
        pk = adamatch_mu_at_step(total_steps // 2, total_steps, mu_max)
        print(f"  [fold {fold}] head={head_type}  {steps} steps/epoch  batch={batch_size}"
              f"  src={len(src_ds)}  tgt={len(tgt_ds)}  mu_max={mu_max}(mid={pk:.2f})"
              f"  saf_w={saf_weight}  sat_m={sat_ema_momentum}  mode={local_threshold_mode}"
              f"  warmup={warmup_steps} steps  cb_patience={patience}  "
              f"fp16={'on' if use_fp16 else 'off'}", flush=True)

    cfg_extra = {"encoder_mode": trainer.encoder.effective_mode,
                 "n_blocks": int(n_blocks), "lr": lr, "weight_decay": weight_decay,
                 "batch_size": batch_size, "pos_fraction": pos_fraction, "seed": seed,
                 "steps_per_epoch": steps, "method": "freematch", "mu_max": mu_max,
                 "saf_weight": saf_weight, "sat_ema_momentum": sat_ema_momentum,
                 "consistency_warmup_frac": consistency_warmup_frac,
                 "local_threshold_mode": local_threshold_mode, "pos_class": pos_class,
                 "fairness_target": (None if fairness_target is None
                                     else list(fairness_target))}

    # curve fold set for the optional per-step hook (matched bake-off passes [0..4]);
    # None -> the legacy fold-0-only default (existing run_freematch callers unchanged).
    cfolds = tuple(curve_hook.DEFAULT_FOLDS if curve_folds is None
                   else (int(k) for k in curve_folds))

    # ---- checkpoint writer -- the encoder delta MUST travel once the encoder trains ----
    # At n_blocks=0 this is exactly run_coral.save_checkpoint, so round-1 checkpoints keep
    # their byte format (and their 'epoch' cfg entry, which evaluate.py's epoch recovery
    # reads). At n_blocks>0 Trainer.save additionally stores encoder_trainable +
    # encoder_mode_effective, so no artifact can claim to be frozen when it is not.
    def save_ckpt(path, epoch_no, full_state=False, gstep=None):
        sched = {"mu_step_offset": int(mu_step_offset),
                 "schedule_total_steps": int(total_steps),
                 "global_step": (None if gstep is None else int(gstep))}
        if int(n_blocks) > 0:
            trainer.save(path, full_state=full_state, schedule=sched)
        else:
            rc.save_checkpoint(path, trainer.head, trainer.standardizer, head_type,
                               fold, epoch_no, cfg_extra,
                               da_buffers=trainer.da_buffer_state(),
                               optimizer_state=(trainer.opt.state_dict() if full_state
                                                else None),
                               schedule=sched)

    # ---- one curve point: checkpoint -> live/cached eval -> row -> instrument ----
    # ``ckpt_every`` separates WEIGHT retention from CURVE density (BRIEF §5.1: 16 weight
    # checkpoints at 500-step spacing, 64 curve points at 125). A curve point still needs a
    # file on disk to score, so a non-retained point writes a single reused scratch file
    # instead of a numbered one -- 59.4 MB per nb3 checkpoint means keeping all 64 would
    # cost ~3.8 GB for artifacts the derived-logit dump already replaces (PROJECT.md §7.7).
    ckpt_stride = int(ckpt_every) if ckpt_every else int(eval_every_steps or 0)
    scratch_ckpt = os.path.join(ckpt_dir, "_curve_scratch.pt")

    def curve_point(gstep, epoch_no, full_state=False):
        retain = bool(ckpt_stride) and (gstep % ckpt_stride == 0)
        spath = os.path.join(ckpt_dir, f"step_{gstep}.pt") if retain else scratch_ckpt
        save_ckpt(spath, epoch_no, full_state=full_state, gstep=gstep)
        loader_hook = None
        if instrument is not None:
            loader_hook = instrument.split_loader(trainer.encoder)
            if loader_hook is not None:
                loader_hook.reset()      # re-encode at THIS checkpoint, never a stale memo
        scores = {} if instrument is not None else None
        row = curve_hook.append_curve_row(
            curve_csv, gstep, spath, cache_dir, folds=cfolds,
            target_fpr=eval_target_fpr, val_select_metric=val_select_metric,
            device=eval_device, agreement_refs=agreement_refs,
            split_loader=loader_hook, collect_scores=scores)
        memo = instrument.on_curve_point(gstep, scores) if instrument is not None else None
        trainer.head.train()             # curve_hook loaded a separate head; be safe
        return row, memo, (spath if retain else None)

    def log_curve(gstep, row, memo):
        if not verbose:
            return
        per_fold = "  ".join(f"f{k}={_cf(row[f'fold{k}_tpr_1e-3'])}" for k in cfolds)
        agr = ("" if not agreement_refs else "  agr " + ",".join(
            f"f{k}={_cf(row.get(f'agreement_fold{k}'))}" for k in sorted(agreement_refs)))
        mem = ("" if not memo else
               f"  mem tgt-test={_cf(memo['target_minus_test_matched'])}")
        print(f"      [step {gstep:>6}] curve  "
              f"val_auroc={_cf(row['source_risk_auroc'])} | "
              f"tpr@1e-3 {per_fold}{agr}{mem}", flush=True)

    os.makedirs(ckpt_dir, exist_ok=True)

    # ---- STEP-0 EVALUATION (BRIEF §2's hard gate) ------------------------------------
    # With an LP->FT head loaded and the encoder still at its published weights, step 0 IS
    # the checkpoint the head came from, so its TPR@1e-3 must reproduce that run's number
    # exactly. It costs ~2 minutes and it is the only check that catches a head-norm
    # mismatch, an unrestored standardizer or a wrong local_threshold_mode BEFORE four
    # hours of GPU time. The caller compares the returned row against its expected value.
    #
    # WINDOWS CAVEAT (Linux/Kaggle unaffected): this builds the LiveEncoder -- and its
    # ~640 MB of eval datasets -- BEFORE the training DataLoader has spawned its workers,
    # which is the ordering PROJECT.md §7.7 records as intermittently fatal to the Windows
    # spawn pickle. The real runs are Linux (fork, no pickling). Use --num-workers 0 for a
    # local Windows smoke.
    # ---- CONTINUATION-STATE RECONSTRUCTION, before the step-0 gate (see docstring) ----
    sat_report = adam_report = None
    if rebuild_sat or adam_warmup_steps:
        import continuation_state as cs
        print("  [continuation] reconstructing the training state a weights-only "
              "checkpoint did not save", flush=True)
    if rebuild_sat:
        sat_report = cs.rebuild_sat_buffers(
            trainer, tgt_loader, n_passes=int(sat_passes), pool_size=len(tgt_ds),
            batch_size=batch_size, ema_iters=int(sat_ema_iters), verbose=verbose)
        cs.sat_check_or_raise(sat_report)          # ABORTS on a rule mismatch
    if adam_warmup_steps:
        adam_report = cs.warm_adam_moments(
            trainer, paired, n_steps=int(adam_warmup_steps), total_steps=total_steps,
            mu_step_offset=int(mu_step_offset), probe_steps=tuple(adam_probe_steps),
            verbose=verbose)
        if sat_report is not None:
            # the warm-up runs 200 more real forward passes, so it also drives the SAT EMA.
            # If the rebuild landed on the fixed point this barely moves -- reported as a
            # self-consistency check rather than assumed.
            sat_report["tau_after_adam_warmup"] = float(trainer.freematch.tau_global)
            sat_report["pbar_after_adam_warmup"] = [float(x) for x in trainer.freematch.pbar]
            sat_report["tau_drift_during_warmup"] = abs(
                sat_report["tau_after_adam_warmup"] - sat_report["tau_direct"])
            if verbose:
                print(f"  [sat] tau after the Adam warm-up: "
                      f"{sat_report['tau_after_adam_warmup']:.6f} "
                      f"(drift {sat_report['tau_drift_during_warmup']:.2e} from the "
                      f"rebuilt value -- it should barely move if the rebuild was the "
                      f"fixed point)", flush=True)

    step0_row = None
    if eval_at_step0 and eval_every_steps and curve_csv:
        if verbose:
            print("  [step0] evaluating the loaded model BEFORE training (hard gate) ...",
                  flush=True)
        step0_row, step0_memo, _ = curve_point(0, 0)
        log_curve(0, step0_row, step0_memo)
        # The gate runs HERE, before a single optimizer step, so a plumbing failure costs
        # two minutes instead of four hours. The callable is expected to raise on failure.
        if step0_check is not None:
            step0_check(step0_row)

    checkpoints: list[str] = []
    epoch_logs: list[dict] = []
    global_step = 0
    for epoch in range(1, max_epoch + 1):
        src_loader.batch_sampler.set_epoch(epoch)
        tgt_loader.batch_sampler.set_epoch(epoch)
        trainer.head.train()
        if trainer.device.type == "cuda":
            torch.cuda.synchronize()               # fair wall-clock (async kernels)
        t_epoch = time.time()
        acc_list, posfrac_list, saf_list = [], [], []
        last = {"sup_loss": float("nan"), "consistency": float("nan"),
                "saf": float("nan"), "mu_eff": 0.0, "acceptance": float("nan"),
                "accepted_pos_fraction": float("nan"), "tau_global": float("nan"),
                "local_neg": float("nan"), "local_pos": float("nan")}
        epoch_fired = False
        for sbatch, tbatch in paired:
            # mu_step_offset shifts ONLY the schedule position the loss reads (mu + the
            # consistency-warmup gate). Nothing else is a function of the step -- there is
            # no LR scheduler in this path -- so the LR is untouched.
            sched_step = global_step + int(mu_step_offset)
            stats = trainer.train_step(sbatch, tbatch, step=sched_step,
                                       total_steps=total_steps)
            # the confirmation-bias detector must follow the SAME gate: with the offset,
            # consistency is live from step 0, so the detector has to be too.
            active = sched_step >= warmup_steps
            if detector.update(stats["acceptance"], stats["accepted_pos_fraction"],
                               active=active):
                epoch_fired = True
            acc_list.append(stats["acceptance"])
            if math.isfinite(stats["accepted_pos_fraction"]):
                posfrac_list.append(stats["accepted_pos_fraction"])
            saf_list.append(stats["saf"])
            last = stats
            if instrument is not None:
                instrument.observe_step(stats)     # §5.3 window accumulation (read-only)
            global_step += 1
            # curve hook (behind --eval-every-steps; default runs never enter here): save the
            # trainable head+standardizer at this step + append one curve row. Legacy callers
            # (curve_folds=None) get the fold-0-only row unchanged; the matched bake-off passes
            # curve_folds=[0..4] + agreement_refs so all five fold columns PLUS the read-only
            # agreement column are logged (eval is feature-space, same as CORAL/AdaMatch).
            if eval_every_steps and curve_csv and (global_step % int(eval_every_steps) == 0):
                # the FINAL checkpoint carries the optimizer state too, so THIS run can be
                # continued from without repeating the reconstruction. Intermediate
                # curve-point checkpoints stay lean (at nb3 the moments would add ~118 MB
                # each).
                is_last = bool(stop_at and global_step >= stop_at)
                row, memo, _kept = curve_point(global_step, epoch, full_state=is_last)
                log_curve(global_step, row, memo)

            # hard stop, INSIDE the epoch: `max_steps` is where training ends while
            # `total_steps` stays the schedule span (docstring). Round-1 callers pass
            # neither and never reach this branch.
            if stop_at and global_step >= stop_at:
                break

        if trainer.device.type == "cuda":
            torch.cuda.synchronize()               # flush the epoch's kernels before timing
        epoch_secs = time.time() - t_epoch

        saved = epoch in schedule
        if saved:
            path = os.path.join(ckpt_dir, ckpt_name(head_type, fold, epoch, width))
            save_ckpt(path, epoch)
            checkpoints.append(path)

        mean_acc = float(np.mean(acc_list)) if acc_list else float("nan")
        mean_pf = float(np.mean(posfrac_list)) if posfrac_list else float("nan")
        mean_saf = float(np.mean(saf_list)) if saf_list else float("nan")
        epoch_logs.append({
            "epoch": epoch, "sup_loss": last["sup_loss"],
            "consistency": last["consistency"], "saf": last["saf"],
            "mean_saf": mean_saf, "mu_eff": last["mu_eff"],
            "mean_acceptance": mean_acc, "mean_accepted_pos_fraction": mean_pf,
            "tau_global": last["tau_global"], "local_neg": last["local_neg"],
            "local_pos": last["local_pos"], "cb_fired": bool(epoch_fired),
            "secs": float(epoch_secs), "saved": bool(saved)})

        if verbose:
            print(f"    [ep {epoch:>{width}}] {epoch_secs:6.1f}s "
                  f"bce={last['sup_loss']:.4f} cons={last['consistency']:.4f} "
                  f"saf={mean_saf:.4f} mu={last['mu_eff']:.3f} "
                  f"accept={mean_acc:.3f} acc_pos_frac={mean_pf:.3f} "
                  f"tau={last['tau_global']:.3f} "
                  f"loc[neg={last['local_neg']:.3f},pos={last['local_pos']:.3f}] "
                  f"cb={'FIRED' if epoch_fired else 'ok'}"
                  + (f"  saved {os.path.basename(path)}" if saved else ""), flush=True)

        if stop_at and global_step >= stop_at:
            break

    # the scratch checkpoint is a scoring buffer, not a result -- remove it so a packaging
    # step cannot mistake it for a retained artifact.
    if os.path.exists(scratch_ckpt):
        try:
            os.remove(scratch_ckpt)
        except OSError:
            pass

    stability = {
        "fold": int(fold), "mu_max": float(mu_max), "saf_weight": float(saf_weight),
        "sat_ema_momentum": float(sat_ema_momentum),
        "local_threshold_mode": local_threshold_mode,
        "consistency_warmup_frac": float(consistency_warmup_frac),
        "warmup_steps": int(warmup_steps), "cb_patience": int(patience),
        "cb_detected": bool(detector.detected), "cb_reason": detector.reason,
        "fairness_target": (None if fairness_target is None else list(fairness_target)),
        "n_saved_checkpoints": len(checkpoints), "epochs": epoch_logs,
        # unfreeze/instrumentation provenance (inert at the round-1 defaults)
        "n_blocks": int(n_blocks), "schedule_total_steps": int(total_steps),
        "stopped_at_step": int(global_step),
        "max_steps": (int(stop_at) if stop_at else None),
        "ckpt_every": (int(ckpt_stride) if ckpt_stride else None),
        "step0_row": step0_row,
        # continuation-state reconstruction (all None on a fresh run)
        "mu_step_offset": int(mu_step_offset),
        "mu_eff_at_step0": adamatch_mu_at_step(int(mu_step_offset), total_steps, mu_max)
        if int(mu_step_offset) >= warmup_steps else 0.0,
        "lr_at_step0": {g.get("name", f"group{i}"): g["lr"]
                        for i, g in enumerate(trainer.opt.param_groups)},
        "sat_reconstruction": sat_report,
        "adam_warmup": adam_report,
        "instrument": (None if instrument is None else instrument.summary())}
    return checkpoints, stability


# ---------------------------------------------------------------------------
# metric suite on the dumped SOURCE-RISK per-fold scores (compare_methods family)
# ---------------------------------------------------------------------------

_SUITE_KEYS = ("tpr", "purity_corr", "purity_raw", "auprc", "comp_at_purity", "auroc")


def _suite_from_scores(out_dir, folds, *, target_fpr=DEFAULT_TARGET_FPR,
                       target_completeness=DEFAULT_TARGET_COMPLETENESS,
                       nn_np_ratio=NN_NP_Q1):
    """Compute the compare_methods metric suite on the SOURCE-RISK scores the evaluator
    dumped (scores/freematch_fold{k}.npz), per fold + across-fold mean."""
    sdir = os.path.join(out_dir, "scores")
    per_fold, agg = {}, {k: [] for k in _SUITE_KEYS}
    for k in folds:
        p = os.path.join(sdir, f"freematch_fold{k}.npz")
        if not os.path.exists(p):
            continue
        with np.load(p, allow_pickle=False) as z:
            pos, neg = z["pos_scores"], z["neg_scores"]
        y = np.concatenate([np.ones(pos.size, np.int64), np.zeros(neg.size, np.int64)])
        s = np.concatenate([pos, neg]).astype(float)
        su = metric_suite(y, s, target_fpr=target_fpr,
                          target_completeness=target_completeness, nn_np_ratio=nn_np_ratio)
        per_fold[int(k)] = {kk: _num(su[kk]) for kk in _SUITE_KEYS}
        for kk in _SUITE_KEYS:
            agg[kk].append(su[kk])
    mean = {kk: _num(float(np.nanmean(agg[kk]))) if agg[kk] else None for kk in _SUITE_KEYS}
    # std only meaningful with >=2 folds (avoids a nanstd DoF<=0 warning on a single fold)
    std = {kk: (_num(float(np.nanstd(agg[kk]))) if len(agg[kk]) > 1 else 0.0) if agg[kk]
           else None for kk in _SUITE_KEYS}
    return {"per_fold": per_fold, "mean": mean, "std": std,
            "target_completeness": target_completeness, "nn_np_ratio": float(nn_np_ratio)}


def _num(x):
    return None if (x is None or not np.isfinite(x)) else float(x)


def _fmt(x, nd=4):
    return "nan" if (x is None or not np.isfinite(x)) else f"{x:.{nd}f}"


def _report_suite(suite, folds):
    print("\n" + "=" * 80)
    print("FREEMATCH METRIC SUITE  (SOURCE-RISK per-fold scores; compare_methods family)")
    print(f"  purity corrected to N_N/N_P = 1:{int(round(suite['nn_np_ratio']))} "
          f"(Lines Q1 rarity);  purity/comp @ {suite['target_completeness']*100:g}% completeness")
    print("=" * 80)
    m, s = suite["mean"], suite["std"]
    labels = [("tpr", "TPR@1e-3           (HEADLINE)"),
              ("purity_corr", "purity@50% (corr 1:2200) HEAD"),
              ("auprc", "AUPRC              (HEADLINE)"),
              ("purity_raw", "purity@50% (raw pool)"),
              ("comp_at_purity", "completeness@24% purity"),
              ("auroc", "AUROC              (secondary)")]
    for key, lab in labels:
        print(f"    {lab:<30} across-fold mean = {_fmt(m[key])} +- {_fmt(s[key])}")


def _report_stability(stability, folds):
    print("\n" + "=" * 80)
    print("FREEMATCH CONFIRMATION-BIAS LOG  (SAT thresholds + accepted class balance + SAF)")
    print("=" * 80)
    for k in sorted(folds):
        st = stability.get(int(k)) or stability.get(str(k))
        if st is None:
            continue
        print(f"\n  [fold {k}]  mode={st['local_threshold_mode']}  mu_max={st['mu_max']:g}  "
              f"saf_w={st['saf_weight']:g}  warmup={st['warmup_steps']} steps  "
              f"collapse={'DETECTED (' + str(st['cb_reason']) + ')' if st['cb_detected'] else 'none'}")
        print("    epoch |   bce   |  cons  |  saf   |  mu   | accept | acc_pos_frac | "
              "tau_t | loc_neg | loc_pos | cb    | saved")
        print("    " + "-" * 108)
        for e in st["epochs"]:
            print(f"    {e['epoch']:>5} | {_fmt(e['sup_loss'],4):>7} | "
                  f"{_fmt(e['consistency'],4):>6} | {_fmt(e['mean_saf'],4):>6} | "
                  f"{_fmt(e['mu_eff'],3):>5} | {_fmt(e['mean_acceptance'],3):>6} | "
                  f"{_fmt(e['mean_accepted_pos_fraction'],3):>12} | "
                  f"{_fmt(e['tau_global'],3):>5} | {_fmt(e['local_neg'],3):>7} | "
                  f"{_fmt(e['local_pos'],3):>7} | "
                  f"{'FIRED' if e['cb_fired'] else ' ok':>5} | "
                  f"{'yes' if e['saved'] else ' no'}")
    n_bad = sum(1 for k in folds
                if (stability.get(int(k)) or stability.get(str(k), {})).get("cb_detected"))
    print("\n" + "=" * 80)
    print(f"CONFIRMATION-BIAS SUMMARY: {n_bad}/{len(list(folds))} folds tripped the collapse "
          f"diagnostic (logged, NOT a selector).")
    print("=" * 80)


# ---------------------------------------------------------------------------
# full FreeMatch run: train per fold (image-space) -> per-fold eval -> aggregate
# ---------------------------------------------------------------------------

def run_freematch(splits_path, data_root, cache_dir, out_dir, *, folds=(0, 1, 2, 3, 4),
                  head_type=LINEAR, epochs=(1, 2, 5, 10, 20), batch_size=256,
                  pos_fraction=0.5, lr=1e-3, weight_decay=1e-4, steps_per_epoch=None,
                  mu_max=1.0, saf_weight=0.01, sat_ema_momentum=0.999,
                  consistency_warmup_frac=0.1, local_threshold_mode="standard",
                  pos_class=1, fairness_target=None, cb_patience=None, seed=42,
                  train_device="cpu", eval_device="cpu", num_workers=0, fp16=False,
                  target_fpr=DEFAULT_TARGET_FPR, bootstrap=1000, eval_seed=0,
                  val_select_metric="auroc", eval_every_steps=None, encoder=None,
                  verbose=True) -> dict:
    from zoobot_encoder import ZoobotEncoder
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    # fold-0 learning-curve hook (behind --eval-every-steps): one curve.csv per run dir.
    # Intended for single-fold diagnostic runs (global_step resets per fold, so a
    # multi-fold run would interleave rows / collide step_*.pt names).
    curve_csv = os.path.join(out_dir, "curve.csv") if eval_every_steps else None
    if verbose:
        print(f"FreeMatch  splits={splits_path}  cache={cache_dir}  out={out_dir}  "
              f"folds={list(folds)}  mode={local_threshold_mode}  mu_max={mu_max} "
              f"saf_w={saf_weight} sat_m={sat_ema_momentum} "
              f"warmup_frac={consistency_warmup_frac} "
              f"fairness={'[0.5,0.5]' if fairness_target is None else list(fairness_target)}")

    # frozen encoder built ONCE (image-space training runs it in the loop)
    if encoder is None:
        if verbose:
            print("[load] building the frozen Zoobot encoder (once) ...", flush=True)
        encoder = ZoobotEncoder(encoder_mode="frozen", device=train_device, verbose=False)

    fold_ckpts, fold_stability = {}, {}
    for k in folds:
        if verbose:
            print(f"\n[train fold {k}] (image-space FreeMatch; encoder in the loop) ...")
        fold_ckpts[k], fold_stability[k] = train_freematch_fold(
            splits_path, data_root, cache_dir, k, ckpt_dir, encoder=encoder,
            head_type=head_type, epochs=epochs, batch_size=batch_size,
            pos_fraction=pos_fraction, lr=lr, weight_decay=weight_decay,
            steps_per_epoch=steps_per_epoch, mu_max=mu_max, saf_weight=saf_weight,
            sat_ema_momentum=sat_ema_momentum,
            consistency_warmup_frac=consistency_warmup_frac,
            local_threshold_mode=local_threshold_mode, pos_class=pos_class,
            fairness_target=fairness_target, cb_patience=cb_patience, seed=seed,
            device=train_device, num_workers=num_workers, fp16=fp16,
            eval_every_steps=eval_every_steps, curve_csv=curve_csv,
            eval_target_fpr=target_fpr, val_select_metric=val_select_metric,
            eval_device=eval_device, verbose=verbose)

    empty = [k for k in folds if not fold_ckpts[k]]
    if empty:
        raise RuntimeError(f"folds {empty} produced NO checkpoints (schedule empty?).")

    if verbose:
        print(f"\n[eval] per-fold models -> per-fold TPR@{target_fpr:g} + across-fold aggregate")
    results = rc.evaluate_per_fold(
        fold_ckpts, cache_dir, device=eval_device, target_fpr=target_fpr, B=bootstrap,
        seed=eval_seed, val_select_metric=val_select_metric, out_dir=out_dir,
        verbose=verbose, method="freematch", score_prefix="freematch")

    # the compare_methods metric suite (AUPRC + purity) on the SOURCE-RISK scores
    suite = _suite_from_scores(out_dir, folds, target_fpr=target_fpr)
    results["metric_suite"] = suite
    results["freematch_stability"] = {int(k): fold_stability[k] for k in fold_stability}
    results["local_threshold_mode"] = local_threshold_mode
    if verbose:
        _report_suite(suite, folds)
        _report_stability(results["freematch_stability"], folds)

    if out_dir:
        with open(os.path.join(out_dir, "results.json"), "w") as fh:
            json.dump(results, fh, indent=2)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    default_root = os.path.normpath(os.path.join(here, "..", "data"))
    p = argparse.ArgumentParser(description="FreeMatch -- per-fold DA runner (image-space).")
    p.add_argument("--data-root", default=os.environ.get("LENS_DATA_ROOT", default_root))
    p.add_argument("--splits", default=None,
                   help="splits.json path (default: <data-root>/splits/splits.json)")
    p.add_argument("--cache-dir", default=os.path.join(default_root, "feature_cache"),
                   help="feature cache (used for the standardizer fit + evaluation)")
    p.add_argument("--out-dir", default=os.path.normpath(os.path.join(here, "..", "runs", "freematch")))
    p.add_argument("--folds", default="0,1,2,3,4", help="comma-separated fold ids")
    p.add_argument("--head-type", default=LINEAR, choices=HEAD_TYPES)
    p.add_argument("--epochs", default="1,2,5,10,20,50",
                   help="comma-separated epoch schedule = SOURCE-RISK selection candidates")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--pos-fraction", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--steps-per-epoch", type=int, default=None)
    p.add_argument("--mu-max", type=float, default=1.0,
                   help="peak consistency weight -- FIXED, not tuned on target")
    p.add_argument("--saf-weight", type=float, default=0.01,
                   help="self-adaptive-fairness weight -- FIXED (USB default 0.01)")
    p.add_argument("--sat-ema-momentum", type=float, default=0.999,
                   help="SAT EMA momentum (FreeMatch default 0.999)")
    p.add_argument("--consistency-warmup-frac", type=float, default=0.1,
                   help="fraction of steps to withhold the consistency loss entirely")
    p.add_argument("--local-threshold-mode", default="standard", choices=FREEMATCH_LOCAL_MODES,
                   help="'standard' (paper) or 'floor' (never lower the lens bar below tau_t) "
                        "-- the make-or-break rare-positive knob")
    p.add_argument("--pos-class", type=int, default=1, help="lens class index (floor protects it)")
    p.add_argument("--fairness-target", default=None,
                   help="'p_neg,p_pos' SAF target (default: uniform [0.5,0.5], balanced pool)")
    p.add_argument("--cb-patience", type=int, default=None,
                   help="confirmation-bias detector patience in steps (default: one epoch)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-device", default=None)
    p.add_argument("--eval-device", default="cpu")
    p.add_argument("--num-workers", type=int, default=4,
                   help="DataLoader workers -- parallelise the CPU augmentation transforms + "
                        "prefetch so they hide behind the GPU forward (0 = single-threaded, "
                        "load-sensitive; 4 keeps the step GPU-bound ~1.1 s). Set 0 to debug.")
    p.add_argument("--fp16", action="store_true",
                   help="autocast the encoder+head FORWARDS to fp16 on CUDA (loss/SAT/SAF "
                        "stay fp32); default off")
    p.add_argument("--target-fpr", type=float, default=DEFAULT_TARGET_FPR)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--eval-seed", type=int, default=0)
    p.add_argument("--val-select-metric", default="auroc", choices=("auroc", "tpr"))
    p.add_argument("--eval-every-steps", type=int, default=None,
                   help="if set, every N optimizer steps save checkpoints/step_N.pt and "
                        "append a fold-0 row to <out-dir>/curve.csv (learning-curve "
                        "diagnostic; default off -> normal run unchanged). Single-fold only.")
    args = p.parse_args()

    splits_path = args.splits or os.path.join(args.data_root, "splits", "splits.json")
    if not os.path.exists(splits_path):
        print(f"SKIP: real splits not found at {splits_path}")
        return 3
    if not os.path.isdir(args.cache_dir):
        print(f"SKIP: feature cache dir not found: {args.cache_dir} (run cache_features.py first)")
        return 3
    cache = ev.FeatureCache(args.cache_dir)
    folds = [int(x) for x in args.folds.split(",") if x.strip() != ""]
    for need in ["source_train", "test_negatives_shared"] + [f"target_fold{k}" for k in folds] \
            + [f"test_pos_fold{k}" for k in folds]:
        if not cache.has(need):
            print(f"SKIP: cache missing {need}.npz (run cache_features.py first)")
            return 3

    fairness_target = None
    if args.fairness_target:
        fairness_target = [float(x) for x in args.fairness_target.split(",")]
        if len(fairness_target) != 2:
            print("--fairness-target must be 'p_neg,p_pos'")
            return 2

    train_device = args.train_device or ("cuda" if torch.cuda.is_available() else "cpu")
    epochs = parse_epochs(args.epochs)
    run_freematch(
        splits_path, args.data_root, args.cache_dir, args.out_dir, folds=folds,
        head_type=args.head_type, epochs=epochs, batch_size=args.batch_size,
        pos_fraction=args.pos_fraction, lr=args.lr, weight_decay=args.weight_decay,
        steps_per_epoch=args.steps_per_epoch, mu_max=args.mu_max, saf_weight=args.saf_weight,
        sat_ema_momentum=args.sat_ema_momentum,
        consistency_warmup_frac=args.consistency_warmup_frac,
        local_threshold_mode=args.local_threshold_mode, pos_class=args.pos_class,
        fairness_target=fairness_target, cb_patience=args.cb_patience, seed=args.seed,
        train_device=train_device, eval_device=args.eval_device,
        num_workers=args.num_workers, fp16=args.fp16, target_fpr=args.target_fpr,
        bootstrap=args.bootstrap, eval_seed=args.eval_seed,
        val_select_metric=args.val_select_metric, eval_every_steps=args.eval_every_steps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
