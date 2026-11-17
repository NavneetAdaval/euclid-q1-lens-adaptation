"""
l1_target_labels.py -- review round 1b, L1: does any arm read target labels in training?
POST HOC AND EXPLORATORY (code reading; nothing is run).

    python -B audit/review_r1b/scripts/l1_target_labels.py

Every answer is a file:line citation located by exact text search (``cite`` from the
round-1 helpers), so a line number cannot drift from the text it quotes. Also records that
the Kaggle-bundle copies of the shared modules are byte-identical to ``scripts/``.

Writes audit/review_r1b/out/L1_target_labels.json.
"""
from __future__ import annotations

import glob
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "review_r1", "scripts"))
import rr_common as RC                                   # noqa: E402  (round-1 helpers)
from f_facts import cite                                 # noqa: E402

SCRIPT = os.path.abspath(__file__)
OUT = os.path.join(os.path.dirname(HERE), "out")


def S(name):
    return RC.R("scripts", name)


def c(name, needle):
    return cite(S(name), needle)[0]


def main():
    I = RC.Inputs()
    files = ["lens_dataset.py", "samplers.py", "run_coral_r4_confirm.py", "run_coral_round4.py",
             "run_unfreeze_freematch.py", "run_freematch.py", "continuation_state.py",
             "run_dann_r5.py", "dann_round5.py", "run_bnm_r6.py", "bnm_round6.py",
             "run_bakeoff_coral.py", "run_coral.py", "run_bakeoff_adamatch.py",
             "run_adamatch.py", "run_bakeoff_freematch.py", "run_dann.py",
             "unfreeze_instrument.py", "train_skeleton.py", "cache_features.py",
             "live_encode.py", "study_validators.py", "validators.py",
             "pseudolabel_precision.py", "da_losses.py"]
    I.add(*[S(f) for f in files])

    shared = {
        "target_items_served_unlabeled": [
            c("lens_dataset.py", 'self._add("real_lens", UNLABELED, TARGET, 1,'),
            c("lens_dataset.py", 'self._add("random", UNLABELED, TARGET, 0,'),
            c("lens_dataset.py", "true identities live in `latent_labels` for DIAGNOSTICS ONLY"),
            c("lens_dataset.py", "self.latent_labels = np.array("),
            c("lens_dataset.py", "self.indices_by_latent = self._group_indices(self.latent_labels)")],
        "balanced_sampler_reads_latent": [
            c("samplers.py", "we balance on ``indices_by_latent`` (latent identity {0,1})"),
            c("samplers.py", "index_map = dataset.indices_by_latent if use_latent else"),
            c("samplers.py", "def make_balanced_loader(")],
        "latent_stratified_target_holdout": [
            c("dann_round5.py", "def build_target_holdout(target_ds"),
            c("dann_round5.py", "3. **Stratified by LATENT class**"),
            c("dann_round5.py", "by_latent = target_ds.indices_by_latent"),
            c("dann_round5.py", "train[cls] = np.sort(np.setdiff1d(pool, pick, assume_unique=True))")],
    }

    arms = {
        "CORAL x nb3": {
            "producer": "scripts/run_coral_r4_confirm.py -> run_coral_round4.build_arm",
            "sampling_mode": "b",
            "sampling": [c("run_coral_r4_confirm.py", "A = R4.build_arm(ARM_KEY, sa, root, for_calibration=True)"),
                         c("run_coral_round4.py", "tgt_loader = make_balanced_loader(tgt_ds, batch_size, use_latent=True,")],
            "other_training_reads": [],
            "diagnostic_only_reads": [],
            "checked_and_label_free": [
                c("run_coral_round4.py", "def gate_or_abort_coral(step0_row"),
                c("unfreeze_instrument.py", "def memorization_row(step: int, scores: dict, fold: int)")],
        },
        "FreeMatch x nb3": {
            "producer": "scripts/run_unfreeze_freematch.py -> run_freematch.train_freematch_fold",
            "sampling_mode": "b",
            "sampling": [c("run_unfreeze_freematch.py", "from run_freematch import train_freematch_fold"),
                         c("run_freematch.py", "tgt_loader = make_balanced_loader(tgt_ds, batch_size, use_latent=True,")],
            "other_training_reads": [
                {"what": "SAT buffer rebuild (FreeMatch's thresholds) averages over batches from "
                         "the same latent-balanced target loader",
                 "lines": [c("continuation_state.py", "def collect_target_batch_stats(trainer, target_loader"),
                           c("continuation_state.py", "Batches come from the SAME loader the training loop uses -- class-balanced on latent")]},
                {"what": "SAF class marginal: uniform [0.5, 0.5] (a fixed constant; reads no "
                         "label, matches the 1:1 target pool)",
                 "lines": [c("da_losses.py", "ft = torch.full((self.num_classes,), init)")]}],
            "diagnostic_only_reads": [],
            "checked_and_label_free": [
                c("run_unfreeze_freematch.py", "def gate_or_abort(step0_row, expected, tol, run_key"),
                c("unfreeze_instrument.py", "def memorization_row(step: int, scores: dict, fold: int)")],
        },
        "DANN x nb3": {
            "producer": "scripts/run_dann_r5.py (arm dann_nb3)",
            "sampling_mode": "b",
            "sampling": [c("run_dann_r5.py", "holdout = build_target_holdout(tgt_ds, frac=float(args.dann_d_holdout_frac),"),
                         c("run_dann_r5.py", 'tgt_sampler = ClassBalancedBatchSampler(holdout["train_pos"], holdout["train_neg"],')],
            "other_training_reads": [
                {"what": "the ~10% held-out target slice (78 images) is chosen stratified by "
                         "latent class and REMOVED from the sampler pool; abort condition 1 "
                         "(armed on DANN's folds) reads D's domain accuracy on that slice "
                         "(domain labels, not lens labels)",
                 "lines": [c("dann_round5.py", "by_latent = target_ds.indices_by_latent"),
                           c("dann_round5.py", 'return {"d_acc_holdout": float(pred_target.float().mean()),'),
                           c("run_dann_r5.py", 'holdout_images = materialise_holdout_images(tgt_ds, holdout["holdout"])')]}],
            "diagnostic_only_reads": [],
            "checked_and_label_free": [
                c("dann_round5.py", 'COND3_COLUMN = "morph_proxy_auroc"'),
                c("dann_round5.py", "# ---- morphology, TWO INSTRUMENTS on one ML-FP vs P1-reserve pool ----")],
        },
        "lambda=0": {
            "producer": "scripts/run_dann_r5.py (arm dann_nb3_lambda0; same code path as DANN)",
            "sampling_mode": "b",
            "sampling": [c("run_dann_r5.py", 'tgt_sampler = ClassBalancedBatchSampler(holdout["train_pos"], holdout["train_neg"],')],
            "other_training_reads": [
                {"what": "same latent-stratified holdout removed from the sampler pool; aborts "
                         "are log_only on every fold of this arm",
                 "lines": [c("dann_round5.py", "by_latent = target_ds.indices_by_latent")]}],
            "diagnostic_only_reads": [],
            "checked_and_label_free": [],
        },
        "BNM x nb3": {
            "producer": "scripts/run_bnm_r6.py",
            "sampling_mode": "b",
            "sampling": [c("run_bnm_r6.py", "holdout = build_target_holdout(tgt_ds, frac=D_HOLDOUT_FRAC, seed=SEED + fold)"),
                         c("run_bnm_r6.py", 'tgt_sampler = ClassBalancedBatchSampler(holdout["train_pos"], holdout["train_neg"],')],
            "other_training_reads": [
                {"what": "same latent-stratified 78-image holdout removed from the sampler pool",
                 "lines": [c("dann_round5.py", "by_latent = target_ds.indices_by_latent")]}],
            "diagnostic_only_reads": [
                {"what": "size-matched, latent-stratified 'trained-on' probe; its normalised "
                         "nuclear norm / entropy / predicted-positive fraction are logged "
                         "(the §8 memorisation diagnostic), no gradient",
                 "lines": [c("bnm_round6.py", "def build_trainedon_probe(target_ds, holdout: dict, *, seed: int)"),
                           c("bnm_round6.py", "by_latent = target_ds.indices_by_latent"),
                           c("run_bnm_r6.py", "probe = build_trainedon_probe(tgt_ds, holdout, seed=SEED + fold)")]}],
            "checked_and_label_free": [],
        },
        "round-1 CORAL (frozen)": {
            "producer": "scripts/run_bakeoff_coral.py -> run_coral.train_coral_fold (cached features)",
            "sampling_mode": "b",
            "sampling": [c("run_bakeoff_coral.py", "from run_coral import train_coral_fold"),
                         c("run_coral.py", 'tgt_latent = np.asarray(tgt["latent_labels"], dtype=np.int64)   # 0/1 identity (balancing only)'),
                         c("run_coral.py", "tpos_idx = np.where(tgt_latent == POS_LABEL)[0]")],
            "other_training_reads": [],
            "diagnostic_only_reads": [],
            "checked_and_label_free": [c("run_coral.py", "stop_on_blowup=True,")],
        },
        "round-1 AdaMatch (frozen)": {
            "producer": "scripts/run_bakeoff_adamatch.py -> run_adamatch.train_adamatch_fold",
            "sampling_mode": "b",
            "sampling": [c("run_bakeoff_adamatch.py", "from run_adamatch import train_adamatch_fold"),
                         c("run_adamatch.py", "tgt_loader = make_balanced_loader(tgt_ds, batch_size, use_latent=True,")],
            "other_training_reads": [],
            "diagnostic_only_reads": [],
            "checked_and_label_free": [],
        },
        "round-1 FreeMatch (frozen)": {
            "producer": "scripts/run_bakeoff_freematch.py -> run_freematch.train_freematch_fold",
            "sampling_mode": "b",
            "sampling": [c("run_bakeoff_freematch.py", "from run_freematch import train_freematch_fold"),
                         c("run_freematch.py", "tgt_loader = make_balanced_loader(tgt_ds, batch_size, use_latent=True,")],
            "other_training_reads": [
                {"what": "SAF class marginal uniform [0.5, 0.5] (constant)",
                 "lines": [c("da_losses.py", "ft = torch.full((self.num_classes,), init)")]}],
            "diagnostic_only_reads": [],
            "checked_and_label_free": [],
        },
        "round-1 DANN probe (fold 0)": {
            "producer": "scripts/run_dann.py (runs/curve_dann_fold0)",
            "sampling_mode": "b",
            "sampling": [c("run_dann.py", 'tgt_latent = np.asarray(tgt["latent_labels"], dtype=np.int64)   # 0/1 identity (balancing)'),
                         c("run_dann.py", "tpos_idx = np.where(tgt_latent == POS_LABEL)[0]")],
            "other_training_reads": [],
            "diagnostic_only_reads": [],
            "checked_and_label_free": [],
        },
    }
    post_hoc_only = {
        "what": "target-identity reads in scripts that run after training on stored outputs "
                "(no effect on gradients or on which steps run)",
        "lines": [c("cache_features.py", "latent_labels   (N,)     int64     true identity 0/1 (diagnostics only)"),
                  c("live_encode.py", '"latent_labels": np.asarray(ds.latent_labels, dtype=np.int64)[idx],'),
                  c("study_validators.py", 'lat = np.asarray(self.tgt["latent_labels"]).reshape(-1)'),
                  c("validators.py", "``latent_labels`` is used ONLY to assert the pool is balanced"),
                  c("pseudolabel_precision.py", 'lens_dataset.py builds role="target" as ``target_lens_rows``')],
    }
    # Kaggle-bundle copies of the shared modules
    bundles = {}
    for name in ("samplers.py", "lens_dataset.py", "dann_round5.py", "run_freematch.py",
                 "run_coral_round4.py", "bnm_round6.py", "run_bnm_r6.py", "run_dann_r5.py"):
        ref = RC.sha256(S(name))
        copies = [p for p in glob.glob(RC.R("kaggle*", "**", name), recursive=True)]
        bundles[name] = {"scripts_sha256": ref,
                         "copies": {RC.rel(p): RC.sha256(p) == ref for p in sorted(copies)}}
    res = {"shared_mechanism": shared, "arms": arms, "post_hoc_only_reads": post_hoc_only,
           "kaggle_bundle_copies_identical_to_scripts": bundles,
           "question_1_modes": {"a": "uniform from the target pool, ignoring labels",
                                "b": "class-balanced / stratified on lens/non-lens identity",
                                "c": "something else"},
           "scope_note": "evaluation on the held-out test fold (step-0 gates, curve scoring, "
                         "collapse gate) is excluded as the brief directs"}
    RC.OUT = OUT
    RC.write_json("L1_target_labels.json", res, script=SCRIPT, inputs=I, pool="n/a",
                  folds="n/a")


if __name__ == "__main__":
    main()
