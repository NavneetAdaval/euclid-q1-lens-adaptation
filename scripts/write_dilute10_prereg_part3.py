"""
write_dilute10_prereg_part3.py
==============================
Writes ``runs/dilute10_prereg_part3.json`` -- PART 3 (final) of the 1:10 diluted-target
pre-registration (brief_dilute10_part3; PROJECTv7.md 7.48). Binding on CORAL x nb3 and
FreeMatch x nb3 only; it changes nothing about how any arm is trained.

Mirrors the part-2 writer: canonical body, sha256 under
``sha256_of_body_without_this_field``, re-read from disk, RAISE on a digest mismatch.

Every value is READ at write time:
  * parts 1 and 2: their self-certifying digests, re-derived and checked against the pinned
    values (dilute10_pool.PART1_SHA256, dilute10_common.PART2_SHA256);
  * the evidence: sha256 of every file in audit/dilute10_coral_guard/ (7.46) and
    audit/dilute10_coral_300/ (7.47); the per-fold 300-step summary from
    audit/dilute10_coral_300/summary.json; the known-null re-derived from kn_f1.json against
    round 4's own steps.csv;
  * the three decisions (CORAL nb3, the no-verdict rule, the FreeMatch nb3 rule): extracted
    from the brief's blockquotes, in order, never retyped; the brief's sha256 is recorded.
    The CORAL decision's quoted figures ("10-11", "290-291") are checked against the summary;
  * the rerun record: the ABORTED folder's file and its sha256, which must be enrolled.

    python -u scripts/write_dilute10_prereg_part3.py --out runs/dilute10_prereg_part3.json \
        --brief C:/Users/krris/Downloads/brief_dilute10_part3.md
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import glob
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import dilute10_pool as DP                                   # noqa: E402

FOLDS = (1, 2, 3, 4)
NB3_ARMS = ("coral_nb3", "freematch_nb3")
ABORTED = "runs/dilute10/coral_nb3/fold1_ABORTED_20261017/cell4_output.txt"
R4_STEPS = "kaggle/coral_r4/runs/main/extracted/coral_r4_confirm/fold{k}/steps.csv"


def R(*p):
    return os.path.join(REPO, *p)


def jload(p):
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def _canonical(body):
    return json.dumps(body, sort_keys=True, separators=(",", ":"))


def self_cert(path):
    d = jload(path)
    claimed = d.get("sha256_of_body_without_this_field")
    body = {k: v for k, v in d.items() if k != "sha256_of_body_without_this_field"}
    got = hashlib.sha256(_canonical(body).encode()).hexdigest()
    if got != claimed:
        raise SystemExit(f"[part3] ABORT: {path} fails its self-certifying digest")
    return got, d


def blockquotes(brief_text):
    """The brief's blockquotes, in order, each joined into one line (the '> ' prefix and the
    line wrapping removed)."""
    out, cur = [], []
    for line in brief_text.splitlines():
        m = re.match(r"^\s*> ?(.*)$", line)
        if m:
            cur.append(m.group(1).strip())
        elif cur:
            out.append(" ".join(cur))
            cur = []
    if cur:
        out.append(" ".join(cur))
    return out


def evidence_hashes(d):
    files = sorted(glob.glob(R("audit", d, "*")))
    if not files:
        raise SystemExit(f"[part3] ABORT: no evidence files in audit/{d}")
    return {f"audit/{d}/{os.path.basename(p)}": DP.sha256_file(p) for p in files}


def summary_300():
    s = jload(R("audit", "dilute10_coral_300", "summary.json"))
    per = {}
    for k in FOLDS:
        f = s["folds"][str(k)]
        if f["any_nonfinite_loss"] or f["any_nonfinite_weight"]:
            raise SystemExit(f"[part3] ABORT: fold {k} summary records a non-finite value")
        per[str(k)] = {
            "recovery_step": f["recovery_step"],
            "applied_updates_of_300": f["applied_updates"],
            "steps_run": f["steps_run"],
            "skipped_steps": f["skipped_steps"],
            "final_scale": f["final_scale"],
            "raw_coral_at": f["raw_coral_at"],
            "feat_l2_src_tgt_at_300": f["src_tgt_f_at"]["300"],
            "feat_l2_below_110": f["f_below_110"],
            "every_loss_finite": not f["any_nonfinite_loss"],
            "every_weight_finite": not f["any_nonfinite_weight"],
            "nonfinite_grad_steps": f["nonfinite_grad_steps"],
            "init_head_l2": f["init_head_l2"],
            "config_diff_mismatches": f["config_diff_mismatches"]}
    return s["definition_recovery_step"], per


def known_null():
    """Re-derived: kn_f1.json (original mode, fold 1, 30 steps) against round 4's own
    steps.csv for fold 1."""
    h = jload(R("audit", "dilute10_coral_300", "kn_f1.json"))
    ref = list(csv.DictReader(open(R(R4_STEPS.format(k=1)), newline="")))[:len(h["rows"])]
    hs = [x["step"] for x in h["rows"] if x["skipped"]]
    rs = [int(r["step"]) for r in ref if float(r["step_skipped"]) > 0]
    hsc = [x["scale"] for x in h["rows"]]
    rsc = [float(r["amp_scale"]) for r in ref]

    def rng(v):
        return [min(v), max(v)]
    res = {"mode": h["mode"], "fold": h["fold"], "steps": h["steps_run"],
           "init_head": os.path.relpath(h["init_head"], REPO).replace("\\", "/"),
           "reference": R4_STEPS.format(k=1),
           "skipped_steps_harness": hs, "skipped_steps_round4": rs,
           "scale_identical_every_step": hsc == rsc,
           "raw_coral_range_harness_1_10": rng([x["coral_raw"] for x in h["rows"][:10]]),
           "raw_coral_range_round4_1_10": rng([float(r["coral_raw"]) for r in ref[:10]]),
           "src_f_step1_harness_round4": [h["rows"][0]["src_feat_l2"],
                                          float(ref[0]["src_feat_l2"])],
           "tgt_f_step1_harness_round4": [h["rows"][0]["tgt_feat_l2"],
                                          float(ref[0]["tgt_feat_l2"])],
           "config_diff_mismatches": h["config_diff_mismatches"]}
    res["passed"] = bool(hs == rs and res["scale_identical_every_step"]
                         and not h["config_diff_mismatches"])
    if not res["passed"]:
        raise SystemExit(f"[part3] ABORT: the known-null does not re-derive as passed: {res}")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--brief", required=True)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    import integrity
    integrity.verify(where="(write_dilute10_prereg_part3 start)")
    out = os.path.abspath(a.out)
    if os.path.exists(out) and not a.force:
        raise SystemExit(f"[part3] refusing to overwrite {out}")

    import dilute10_common as C
    p1h, p1 = self_cert(R("runs", "dilute10_prereg_part1.json"))
    p2h, p2 = self_cert(R("runs", "dilute10_prereg_part2.json"))
    if p1h != DP.PART1_SHA256 or p2h != C.PART2_SHA256:
        raise SystemExit("[part3] ABORT: part 1 or part 2 differs from the pinned digest")

    brief_text = open(a.brief, encoding="utf-8").read()
    bq = blockquotes(brief_text)
    if len(bq) != 3:
        raise SystemExit(f"[part3] ABORT: expected 3 blockquotes in the brief, found {len(bq)}")
    coral_decision, no_verdict, fm_rule = bq
    for want, txt in (("CORAL × nb3 runs exactly as pre-registered", coral_decision),
                      ("fewer than 50% of its budget", no_verdict),
                      ("Before bundle v3 is built, FreeMatch × nb3", fm_rule)):
        if txt is not None and want not in txt:
            raise SystemExit(f"[part3] ABORT: blockquote order/content unexpected ({want!r})")

    definition, per = summary_300()
    rec = sorted({v["recovery_step"] for v in per.values()})
    upd = sorted({v["applied_updates_of_300"] for v in per.values()})
    quoted = (f"recovery by step {rec[0]}–{rec[-1]}", f"{upd[0]}–{upd[-1]} of 300 updates")
    for q in quoted:
        if q not in coral_decision:
            raise SystemExit(f"[part3] ABORT: the decision's figures do not match the "
                             f"summary: {q!r} not in the decision text")
    kn = known_null()

    if not os.path.isfile(R(ABORTED)):
        raise SystemExit(f"[part3] ABORT: {ABORTED} missing")
    inv = jload(R("runs", "protected_artifacts.json"))
    ab_sha = DP.sha256_file(R(ABORTED))
    if not any(x["path"] == ABORTED and x["sha256"] == ab_sha for x in inv["artifacts"]):
        raise SystemExit(f"[part3] ABORT: {ABORTED} is not enrolled at its current hash")

    body = {
        "experiment": "dilute10 -- the 1:10 diluted-target rerun",
        "part": 3,
        "status": ("part 3 of 3 -- FINAL; binding on CORAL x nb3 and FreeMatch x nb3 only; it "
                   "changes nothing about how any arm is trained; it must be enrolled in "
                   "runs/protected_artifacts.json before either CORAL x nb3 or FreeMatch x nb3 "
                   "runs"),
        "binding_on": list(NB3_ARMS),
        "part1": {"path": "runs/dilute10_prereg_part1.json", "sha256": p1h,
                  "verified": True, "recorded_utc": p1["recorded_utc"]},
        "part2": {"path": "runs/dilute10_prereg_part2.json", "sha256": p2h,
                  "verified": True, "recorded_utc": p2["recorded_utc"]},
        "evidence": {
            "section_7_46_10_step": evidence_hashes("dilute10_coral_guard"),
            "section_7_47_300_step": evidence_hashes("dilute10_coral_300"),
            "coral_nb3_300_step_summary": {
                "source": "audit/dilute10_coral_300/summary.json",
                "configuration": ("pre-registered CORAL x nb3: fp16, gamma 16, real new 1:10 "
                                  "frozen heads, dilute10 + uniform; feature-norm guard "
                                  "log-only; label-free (no step-0 gate, no scorer)"),
                "definition_recovery_step": definition,
                "per_fold": per},
            "known_null": kn},
        "decision_coral_nb3": coral_decision,
        "no_verdict_rule": no_verdict,
        "freematch_nb3_rule": fm_rule,
        "coral_nb3_fold1_rerun": {
            "permitted": True,
            "under": "part 1 stop condition 6",
            "logged_reason": (
                "the 2026-10-17 run stopped at step 1 on a guard that part 2 declares advisory "
                "(an implementation nonconformance, since fixed, PROJECTv7 7.46-7.47). It "
                "produced no confirmatory data."),
            "record": "runs/dilute10/coral_nb3/fold1_ABORTED_20261017/",
            "record_file_sha256": ab_sha,
            "rerun_dir": "runs/dilute10/coral_nb3/fold1/"},
        "brief": {"path": os.path.abspath(a.brief).replace("\\", "/"),
                  "sha256": DP.sha256_file(a.brief),
                  "decisions_extracted_from": "the brief's three blockquotes, in order"},
        "writer_sha256": DP.sha256_file(os.path.abspath(__file__)),
        "recorded_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    body = json.loads(json.dumps(body, default=str))
    h = hashlib.sha256(_canonical(body).encode()).hexdigest()
    body["sha256_of_body_without_this_field"] = h
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(body, fh, indent=2, sort_keys=True, ensure_ascii=False)
    chk = jload(out)
    got = chk.pop("sha256_of_body_without_this_field")
    if got != h or hashlib.sha256(_canonical(chk).encode()).hexdigest() != h:
        raise SystemExit("[part3] DIGEST MISMATCH on re-read")
    integrity.verify(where="(write_dilute10_prereg_part3 end)")
    print(f"[part3] {out}\n  sha256       {h}\n  recorded_utc {body['recorded_utc']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
