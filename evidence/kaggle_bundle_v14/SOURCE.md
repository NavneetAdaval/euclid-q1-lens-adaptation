# Files recovered from the Kaggle training bundle v14

These two files were overwritten in the project after the λ = 0 confirmatory folds ran
(PROJECTv7 §7.38, §7.57). They were recovered from the archived Kaggle dataset version that the
folds trained from, and are kept here byte for byte. Both are protected artifacts
(`runs/protected_artifacts.json`) and read-only.

| | |
|---|---|
| Kaggle dataset | `navneet78/lens-paper-coral-r4` (the dataset the λ = 0 run files name; it holds the `dann_r5` bundle) |
| Dataset version | Kaggle version **3** (its contents are bundle `dann_r5 v14`; Kaggle numbers its own versions) |
| Version date | **2026-09-06** (the v14 rebuild, PROJECTv7 §7.21; confirmed by the author) |
| Downloaded | 2026-10-24 (19:12 and 19:14 IST) |

| File | Bundle path | sha256 | Matches |
|---|---|---|---|
| `da_losses.py` | `scripts/da_losses.py` | `db8ba9ecb2a1364f408a32e6fdacc9c8cb9c4adaf7d0f858aca6f3093f7540d0` | `script_sha256_16` of all four λ = 0 confirmatory folds (`runs/baseline_unfrozen/fold*/extracted/lambda0_r6_fold*/run.json`) |
| `lambda0_r6_audit.json` | `runs/lambda0_r6_audit.json` | `7bc26896614ea685225c0f8c1bd4553dde40746bcccb6ee5b3eaa457720c77fa` | `input_artifacts` of the hashed λ = 0 pre-registration (`runs/lambda0_r6_prereg.json`) |
| `lambda0_r6_prereg.json` | `runs/lambda0_r6_prereg.json` | `883cca8dd3da1a5e96290bdca2fe2ddea971696a69488e53612cc8cd35a08170` (self-certifying body digest `c1694a479eafb040cf7ad108e554047a95811cde8ce646444a448e8b1d717a25`) | byte-identical to `runs/lambda0_r6_prereg.json`; its body digest is the one the inventory records for that file |

The third file was downloaded on 2026-10-27 (saved 20:22 IST) from the same Kaggle version and
added on 2026-10-27 (PROJECTv7 §7.60). It was not overwritten in the project; it is kept here as
the third-party copy of the λ = 0 pre-registration, which holds the four pre-registered contrasts
and, in `input_artifacts`, the audit's digest `7bc26896…`.

**What they show.**

- `da_losses.py` (1,616 lines) equals `scripts/da_losses.py` through line 1440. Those lines hash
  to `6b05cd66…`, the version the DANN folds ran. It differs from today's file only inside
  `BNMLoss`, at line 1606: today's file computes the entropy monitor in float32.
- `lambda0_r6_audit.json` is the pre-run audit. The copy at `runs/lambda0_r6_audit.json` is a
  later re-run (2026-09-08) that adds three `"detail": {"scanned": …}` entries and a `"runner"`
  field to `part_c_schedule_audit` and is otherwise identical.
- **Does the Kaggle copy predate the λ = 0 confirmatory folds?** Fold 1 started
  2026-09-07T08:38:34Z (fold 1 `run_config.json` `started_utc`; folds 2–4: 2026-09-07T14:52:11Z,
  2026-09-07T17:37:25Z, 2026-09-08T11:10:05Z). Version 3 is dated 2026-09-06. Its creation time
  could not be read: the Kaggle CLI (2.2.4, account `navneet78`) answered 403 Forbidden on
  2026-10-27. Whether the date is read in IST or UTC, the version predates fold 1 by at least 8.6 h.
  The inputs agree: every fold recorded `da_losses.py` prefix `db8ba9ecb2a1364f`, as in this
  version, and names `navneet78/lens-paper-coral-r4` as its input. The run files record no version
  number, so another version holding the same bytes cannot be excluded on the inputs alone.
  **Verdict: predates (on the version date).**
