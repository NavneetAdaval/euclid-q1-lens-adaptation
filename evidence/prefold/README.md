# Pre-fold evidence: the DANN pre-registration as of 2026-09-02

## `dann_r5_prereg_kaggle_v2_20260902.json`

A byte-for-byte copy of `runs/dann_r5_prereg.json` as it stood before DANN's confirmatory
folds.

| Field | Value |
|---|---|
| **Source** | Kaggle dataset `navneet78/lens-paper-coral-r4`, **version 2**. Version 2 is the only version created on 2 September 2026, **as reported by the author from the dataset page**. The dataset is private and also contains Euclid imaging, which is not redistributed. Only this file is archived here |
| **Saved** | by the author on 2026-09-27, with the browser's "Save page as". Copied here byte-for-byte (`cmp` identical) from `kaggle_prefold/dann_r5_prereg.json` on 2026-09-28 |
| **Whole-file sha256** | `6c49184fb5398701f5e3980b636d409653538b80c0513d2845bdd69a63f0bcbc` |
| **Self-certifying digest** (recomputed with `scripts/integrity.self_certifying_sha`) | `51351308196ad33204a3841e86d32cff9b0410d918938aae24fe57363f65a1bb`. This equals the value embedded in the file's `sha256_of_body_without_this_field` |
| **Match** | entry **10** of the hash history in `runs/dann_main/results_pack.md` §8, `51351308196ad332`, "the step-2000 sensitivity readout pre-declared — before any confirmatory fold". The pack prints only 16 hex characters, so the match is on that 64-bit prefix |
| **Embedded `recorded_utc`** | `2026-09-02T16:26:53Z`. The first DANN confirmatory fold started at `2026-09-03T13:54:49Z` (`runs/dann_main/fold1/extracted/fold1/run.json`) |
| **Byte format** | CRLF, no trailing newline, `indent=1`, sorted keys, ASCII-escaped. This is exactly the output of the project writer (`scripts/write_dann_r5_prereg.py`) on Windows, so the browser save did not alter the bytes |

**Relation to the current file.** The current `runs/dann_r5_prereg.json` (entry 11, regenerated
2026-09-05) has 41 keys. All 35 non-self-describing keys of this copy are present in it with
identical values. Only `recorded_utc` and the self-hash differ. The current file adds four
post-fold keys: `ROUND_5_RESULT`, `PAIRED_cross_method_analysis`,
`prediction_3_spread_not_mean_RESOLVED` and `SENSITIVITY_CHECK_RESOLVED`.
`paper/scripts/make_a2_hashes.py` asserts this at every build.

**The budget file** saved from the same dataset version was **byte-identical** to the
protected `runs/dann_r5_budget.json`: whole-file sha256
`8ed4c72eeea075244712b133f15df80afd9f943a31154d071bcc740325f0f824`, self-certifying
`abf0dae0…`. That is the digest every DANN fold recorded when it started. It is not copied
here.

**Protection.** The file is set read-only (mode 444). It is **not** in
`runs/protected_artifacts.json`: `scripts/integrity.py` enrols only files under `runs/` whose
digest the project record carries, and this file's digest is not in the record. See
`paper/DANN_PREREG_RESTORED_REPORT.md`.

**What it does not establish.** It establishes the file as of 2026-09-02 (entry 10). It does
not recover entries 1–9, including the 2026-08-26 original. By 2 September the file already
contained the arc-suppression prediction's resolved outcome (resolved 2026-09-01), so this
copy postdates that prediction's own fold-0 test. The prediction's precedence over that test
(entry 5, `e28adf000ee127ab`, "`prediction_arc_suppression` hashed — before any TPR curve
existed") rests on the project log.
