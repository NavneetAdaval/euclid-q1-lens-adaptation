"""
unfreeze_instrument.py
======================
The INSTRUMENTATION for the FreeMatch x n_blocks=3 fold-0 diagnostic run
(BRIEF_freematch_nb3_fold0.md §5 -- "the main deliverable of this run").

This module owns *only* recording. It computes no training signal, feeds nothing back
into any loss, and never selects anything. The training loop calls it through one small
object (:class:`CurveInstrument`) so ``run_freematch.train_freematch_fold`` keeps a single
optional argument rather than eight, and so a legacy caller that passes nothing gets the
round-1 code path byte-for-byte.

WHAT IT RECORDS (one row/file per CURVE POINT)
---------------------------------------------
§5.2  DERIVED LOGITS -> ``logits/step_{step}.npz``
      Per-split head logits for target_fold{k}, val, test_pos_fold{k} and
      test_negatives_shared. This is the fix PROJECT.md §7.7 proposed against the 38 GB
      weight-dump scaling flag: at 64 curve points these total ~10 MB against ~950 MB of
      weights, and they are what ``validators.py`` (IM / SND / ATC) actually needs -- the
      artifacts whose loss made those validators uncomputable on nb1/nb3 (§7.6). They cost
      NOTHING extra to produce: ``curve_hook`` already scores exactly these four splits to
      build its row, so the dump is a by-product, not a second pass.

§5.3  FREEMATCH INTERNALS -> ``diagnostics.csv``
      Self-adaptive threshold tau (global AND per-class), mask rate (global and per class),
      the pseudo-label class distribution, and the unlabeled-vs-supervised loss magnitudes.
      These are per-STEP quantities, so each row is the MEAN over the window of steps since
      the previous curve point (plus the last step's value, since tau is an EMA and its
      instantaneous value is the meaningful one). Windowed means are used rather than a
      single sampled step because acceptance is noisy per batch and the question ("is SAF
      keeping predictions diverse?") is about the trend, not one draw.

§5.4  THE MEMORIZATION PROBE -> ``memorization.csv``
      Mean predicted lens-class confidence on the TARGET pool (trained on, ~782 images seen
      ~1000x each at 8000 steps x batch 256) against the TEST pool (never trained on).
      Rising together => the model is learning something real; target pulling away => the
      §4 target-pool-memorization signature (Arazo et al. arXiv:1908.02983 confirmation
      bias, now with 14.8M parameters instead of 1,921 to store 782 images with).

      COMPOSITION MATTERS AND IS HANDLED. The target pool is ~1:1 lens:random-galaxy
      (PROJECT.md §4.3) while the test pool is ~1:269, so a raw mean-vs-mean comparison
      would move with composition alone and could masquerade as memorization. We log the
      raw means AND a composition-matched test statistic,
      ``0.5*mean(sigma(test_pos)) + 0.5*mean(sigma(test_neg))``, which reweights test to the
      target's 1:1 composition -- the same class mix, so the only remaining difference is
      trained-on vs not. The matched column is the one to read; the raw ones are kept
      because suppressing an input to a diagnostic is how confounds hide.

WHAT IT DOES NOT DO
-------------------
No pseudo-label precision. The ~391 target lenses have known A/B grades, and comparing
pseudo-labels against them is legitimate POST-HOC (§5.6) but must never touch the training
loop -- the moment it could, it is oracle selection and §7.6's argument collapses. It is
therefore implemented in a separate script (``pseudolabel_precision.py``) that reads the
dumped logits, and there is deliberately no hook for it here.
"""

from __future__ import annotations

import csv
import math
import os

import numpy as np

# ---------------------------------------------------------------------------
# per-step accumulation window
# ---------------------------------------------------------------------------

# The per-step FreeMatch statistics we average across a curve window. Every one of these
# is produced by FreeMatchLoss.forward()'s info dict and forwarded by
# Trainer._freematch_train_step; nothing is recomputed here.
STEP_KEYS = (
    "sup_loss",                 # supervised BCE on the SOURCE weak view
    "consistency",              # the UNLABELED loss term (pre-mu)
    "saf",                      # self-adaptive fairness penalty
    "mu_eff",                   # effective consistency weight at this step
    "acceptance",               # mask rate over the whole unlabeled batch
    "mask_rate_pos",            # mask rate WITHIN the lens-argmax subset
    "mask_rate_neg",            # mask rate WITHIN the non-lens-argmax subset
    "accepted_pos_fraction",    # class balance of the ACCEPTED pseudo-labels
    "pseudo_pos_fraction_all",  # class balance of ALL pseudo-labels (pre-mask)
    "tau_global",               # self-adaptive threshold, global
    "local_neg",                # local threshold, non-lens class
    "local_pos",                # local threshold, lens class
    "pbar_neg", "pbar_pos",     # the per-class confidence EMAs tau is built from
    "mean_confidence_target",   # mean max-prob on the unlabeled batch
    "n_accepted",
    # fp16 health. ``step_skipped_mean`` is the fraction of steps in the window whose
    # optimizer update the GradScaler DISCARDED on a non-finite gradient. It is 0 for a
    # frozen run and was measured non-zero on an unfrozen smoke, where two consecutive
    # checkpoints came out bit-identical. A run that loses many updates this way is
    # indistinguishable, from the metrics alone, from a run where unfreezing did nothing.
    "step_skipped",
    "amp_scale",
)

# columns of diagnostics.csv, in order
DIAG_COLUMNS = (["step", "n_steps_in_window"]
                + [f"{k}_mean" for k in STEP_KEYS]
                # ``amp_scale`` is here as a _last as well as a _mean for a reason: the
                # GradScaler scale is a STEPWISE POWER-OF-TWO ladder, and a window MEAN of
                # it is a number the ladder never takes. Reading a mean as if it were the
                # scale produces sequences like 65536 -> 49152 -> 12288 that match no
                # backoff factor and invite a hunt for a non-default scaler. The _last
                # column is the instantaneous value; read that one.
                + [f"{k}_last" for k in ("tau_global", "local_neg", "local_pos",
                                         "pbar_neg", "pbar_pos", "amp_scale")]
                + ["unlabeled_over_supervised", "unlabeled_loss_weighted"])

MEMO_COLUMNS = ["step", "conf_target_mean", "conf_test_matched",
                "conf_test_pos_mean", "conf_test_neg_mean", "conf_test_raw_mean",
                "target_minus_test_matched", "conf_val_mean"]


class StepWindow:
    """Accumulates per-step FreeMatch stats between two curve points.

    ``nan`` values are skipped per key (``accepted_pos_fraction`` is legitimately undefined
    on a step where nothing was accepted, and ``mask_rate_pos`` when no sample's argmax was
    the lens class). A key whose every value in the window was undefined stays undefined --
    it is never silently reported as 0.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._sum = {k: 0.0 for k in STEP_KEYS}
        self._n = {k: 0 for k in STEP_KEYS}
        self._last = {}
        self.n_steps = 0

    def update(self, stats: dict):
        self.n_steps += 1
        for k in STEP_KEYS:
            v = stats.get(k)
            if v is None:
                continue
            v = float(v)
            self._last[k] = v
            if math.isfinite(v):
                self._sum[k] += v
                self._n[k] += 1

    def mean(self, key):
        n = self._n.get(key, 0)
        return (self._sum[key] / n) if n else float("nan")

    def last(self, key):
        return self._last.get(key, float("nan"))

    def row(self, step: int) -> dict:
        row = {"step": int(step), "n_steps_in_window": int(self.n_steps)}
        for k in STEP_KEYS:
            row[f"{k}_mean"] = _num(self.mean(k))
        for k in ("tau_global", "local_neg", "local_pos", "pbar_neg", "pbar_pos",
                  "amp_scale"):
            row[f"{k}_last"] = _num(self.last(k))
        sup, cons, mu = self.mean("sup_loss"), self.mean("consistency"), self.mean("mu_eff")
        # the brief asks for "unlabeled loss magnitude vs supervised loss magnitude". Both
        # the raw ratio and the AS-APPLIED (mu-weighted) unlabeled term are logged: during
        # the consistency warmup mu_eff is exactly 0, so the raw ratio can be large while
        # the unlabeled term contributes literally nothing to the gradient.
        row["unlabeled_over_supervised"] = _num(cons / sup if (sup and math.isfinite(sup)
                                                              and sup != 0) else float("nan"))
        row["unlabeled_loss_weighted"] = _num(
            mu * cons if (math.isfinite(mu) and math.isfinite(cons)) else float("nan"))
        return row


# ---------------------------------------------------------------------------
# the memorization probe (§5.4)
# ---------------------------------------------------------------------------

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def memorization_row(step: int, scores: dict, fold: int) -> dict:
    """Mean predicted LENS-class confidence, target pool vs test pool.

    ``scores`` is the ``collect_scores`` dict curve_hook filled: raw head LOGITS per split.
    The head is binary, so the lens-class probability is ``sigmoid(logit)`` -- the same
    monotone transform the metrics use, applied here only to make the means comparable
    across splits (a logit mean is not a confidence).
    """
    tgt = scores.get(f"target_fold{int(fold)}")
    pos = scores.get(f"test_pos_fold{int(fold)}")
    neg = scores.get("test_negatives_shared")
    val = scores.get("val")
    if tgt is None or pos is None or neg is None:
        raise KeyError(
            f"memorization probe needs target_fold{fold} / test_pos_fold{fold} / "
            f"test_negatives_shared in the collected scores; got {sorted(scores)}")

    c_t = float(np.mean(sigmoid(tgt)))
    c_p = float(np.mean(sigmoid(pos)))
    c_n = float(np.mean(sigmoid(neg)))
    n_p, n_n = pos.size, neg.size
    c_raw = float((c_p * n_p + c_n * n_n) / (n_p + n_n))
    # composition-matched: reweight test to the target pool's ~1:1 lens:non-lens mix so the
    # comparison isolates trained-on vs never-trained-on rather than class composition.
    c_matched = 0.5 * c_p + 0.5 * c_n
    return {"step": int(step), "conf_target_mean": c_t, "conf_test_matched": c_matched,
            "conf_test_pos_mean": c_p, "conf_test_neg_mean": c_n,
            "conf_test_raw_mean": c_raw,
            "target_minus_test_matched": c_t - c_matched,
            "conf_val_mean": (None if val is None
                              else float(np.mean(sigmoid(val))))}


# ---------------------------------------------------------------------------
# tidy CSV appender
# ---------------------------------------------------------------------------

def append_row(path: str, columns, row: dict):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(columns))
        if new:
            w.writeheader()
        w.writerow({c: row.get(c) for c in columns})


# ---------------------------------------------------------------------------
# derived-logit dump (§5.2)
# ---------------------------------------------------------------------------

LOGIT_SPLITS = ("val", "test_negatives_shared", "test_pos_fold{k}", "target_fold{k}")


def dump_logits(logit_dir: str, step: int, scores: dict, fold: int,
                write_ids: bool = False) -> str:
    """Write one ``step_{step}.npz`` of per-split head logits; return its path.

    float32 is deliberate and sufficient: these are a RANKING signal (every metric here is
    rank-based or threshold-on-rank) and float32 carries ~7 decimal digits, far finer than
    the 1/98 = 0.0102 one-positive resolution of the headline metric. ids are written ONCE
    to ``ids.npz`` rather than into all 64 files -- they are identical at every curve point
    (same rows, same order) and are what lets a later paired analysis VERIFY row alignment
    instead of assuming it.
    """
    os.makedirs(logit_dir, exist_ok=True)
    names = [n.format(k=int(fold)) for n in LOGIT_SPLITS]
    payload = {}
    for n in names:
        if n not in scores:
            raise KeyError(f"logit dump is missing split {n!r} (have: "
                           f"{sorted(k for k in scores if not k.endswith('__ids'))})")
        payload[n] = np.asarray(scores[n], dtype=np.float32)
    path = os.path.join(logit_dir, f"step_{int(step)}.npz")
    np.savez_compressed(path, step=np.asarray(int(step)), **payload)

    ids_path = os.path.join(logit_dir, "ids.npz")
    if write_ids and not os.path.exists(ids_path):
        idp = {n: np.asarray(scores[f"{n}__ids"]) for n in names
               if f"{n}__ids" in scores}
        if idp:
            np.savez_compressed(ids_path, **idp)
    return path


# ---------------------------------------------------------------------------
# the object the training loop talks to
# ---------------------------------------------------------------------------

class CurveInstrument:
    """One object holding every §5 recorder, so the training loop gains one argument.

    Parameters
    ----------
    out_dir : str
        Run directory. ``diagnostics.csv``, ``memorization.csv`` and ``logits/`` land here.
    fold : int
        Which fold's target/test splits the probes read.
    live_encoder_factory : callable | None
        ``() -> LiveEncoder`` (or None at n_blocks=0). Called LAZILY at the first curve
        point, never at construction: PROJECT.md §7.7 records that building the eval
        datasets before the training DataLoader spawns its workers intermittently kills the
        Windows spawn pickle (~800 MB source dataset + ~640 MB eval datasets resident at
        once). Deferring it means the workers are already up.
    fp16_eval : bool
        Whether the LIVE eval encode runs under autocast. Default False (fp32), which is
        what makes the step-0 gate an EXACT comparison against the frozen feature cache --
        live_encode reproduces the cache byte-for-byte at n_blocks=0 in fp32.
    """

    def __init__(self, out_dir, fold, *, live_encoder_factory=None, fp16_eval=False,
                 dump_logits_every=True, verbose=True):
        self.out_dir = out_dir
        self.fold = int(fold)
        self._factory = live_encoder_factory
        self.fp16_eval = bool(fp16_eval)
        self.dump_logits = bool(dump_logits_every)
        self.verbose = verbose
        self.window = StepWindow()
        self._live = None
        self._live_built = False
        self._wrote_ids = False
        self.diag_csv = os.path.join(out_dir, "diagnostics.csv")
        self.memo_csv = os.path.join(out_dir, "memorization.csv")
        self.logit_dir = os.path.join(out_dir, "logits")
        self.n_curve_points = 0
        self.n_logit_dumps = 0
        self.total_steps_seen = 0
        self.total_steps_skipped = 0.0
        self.max_window_skip_rate = 0.0

    # -- live encoding -----------------------------------------------------

    def live(self):
        if not self._live_built:
            self._live = self._factory() if self._factory is not None else None
            self._live_built = True
        return self._live

    def split_loader(self, encoder):
        """The ``split_loader`` curve_hook consumes: None at n_blocks=0 (the frozen cache is
        valid and byte-identical), else a memoised live re-encode with the CURRENT encoder.

        The returned callable carries ``.reset()``, which the caller MUST invoke at each
        curve point -- without it the second curve point would silently re-serve the first
        point's features and every later row would describe a stale model.
        """
        live = self.live()
        if live is None:
            return None
        memo = {}

        def loader(name):
            if name not in memo:
                memo[name] = live.encode(encoder, name, fp16=self.fp16_eval)
            return memo[name]

        loader.reset = memo.clear
        return loader

    # -- per-step ----------------------------------------------------------

    # fraction of DISCARDED optimizer steps in one window above which the operator is
    # warned. 5% is a judgement call, not a measurement [UNVERIFIED]: it is low enough to
    # surface a developing problem early and high enough not to fire on the occasional
    # overflow that GradScaler's scale search produces by design.
    SKIP_WARN_RATE = 0.05

    def observe_step(self, stats: dict):
        self.window.update(stats)
        self.total_steps_seen += 1
        sk = stats.get("step_skipped")
        if sk is not None and math.isfinite(float(sk)):
            self.total_steps_skipped += float(sk)

    # -- per-curve-point ---------------------------------------------------

    def on_curve_point(self, step: int, scores: dict) -> dict:
        """Write the diagnostics row, the memorization row and the logit dump.

        Returns the memorization row (the training loop prints one field of it, so the
        operator can see the §4 signature developing without waiting for the analysis)."""
        diag = self.window.row(step)
        append_row(self.diag_csv, DIAG_COLUMNS, diag)
        skip = diag.get("step_skipped_mean")
        if skip is not None and skip > 0:
            self.max_window_skip_rate = max(self.max_window_skip_rate, float(skip))
            if skip >= self.SKIP_WARN_RATE and self.verbose:
                print(f"      [WARN step {step}] fp16 GradScaler DISCARDED "
                      f"{skip*100:.1f}% of this window's optimizer steps (non-finite "
                      f"gradients). Those steps changed nothing. Read the final TPR with "
                      f"this in mind -- lost updates and 'unfreezing does not help' look "
                      f"identical in the metric.", flush=True)
        self.window.reset()

        memo = memorization_row(step, scores, self.fold)
        append_row(self.memo_csv, MEMO_COLUMNS, memo)

        if self.dump_logits:
            dump_logits(self.logit_dir, step, scores, self.fold,
                        write_ids=not self._wrote_ids)
            self._wrote_ids = True
            self.n_logit_dumps += 1
        self.n_curve_points += 1
        return memo

    def summary(self) -> dict:
        n = max(self.total_steps_seen, 1)
        return {"n_curve_points": self.n_curve_points,
                "n_logit_dumps": self.n_logit_dumps,
                "diagnostics_csv": self.diag_csv, "memorization_csv": self.memo_csv,
                "logit_dir": self.logit_dir,
                "eval_precision": ("fp16" if self.fp16_eval else "fp32"),
                "steps_observed": int(self.total_steps_seen),
                "steps_skipped_fp16": int(round(self.total_steps_skipped)),
                "skip_rate_fp16": float(self.total_steps_skipped) / n,
                "max_window_skip_rate": float(self.max_window_skip_rate)}


def _num(x):
    """None (-> blank CSV cell) for anything non-finite, matching curve_hook._num."""
    if x is None:
        return None
    x = float(x)
    return None if not math.isfinite(x) else x
