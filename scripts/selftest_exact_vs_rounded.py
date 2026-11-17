"""Self-test: a planted overlap is caught by the exact comparison and missed by the old one.

Temp files only (tempfile.mkdtemp, outside the project and deleted afterwards). Mirrors the
two real failure modes: (1) rounded deflector IDs compared against exact negative IDs
(build_splits purge / verify_leakage (3)); (2) two different roundings of the same ID
(filter_deflectors_leakfree: float64 vs a 16-significant-digit CSV float).
"""
import csv
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exact_ids as X  # noqa: E402

tmp = tempfile.mkdtemp(prefix="decontam_selftest_")
ok = True
try:
    # a planted galaxy that is BOTH a simulation deflector and a test negative
    ra, dec = 272.7137773, 63.9721420          # arbitrary position
    exact = X.rebuild(ra, dec)
    rounded = int(np.float64(exact))            # what the deflector file stored
    assert rounded != exact, "pick a position whose ID is not float-representable"
    np.savez(os.path.join(tmp, "defl.npz"), object_id=np.array([rounded], "int64"),
             ra=np.array([ra]), dec=np.array([dec]))
    np.savez(os.path.join(tmp, "testneg.npz"), object_ids=np.array([exact, exact + 7], "int64"))

    d = np.load(os.path.join(tmp, "defl.npz"))
    t = set(int(o) for o in np.load(os.path.join(tmp, "testneg.npz"))["object_ids"])
    old = X.rounded_compare({int(o) for o in d["object_id"]}, t)
    new = X.exact_compare({X.rebuild(float(a), float(b)) for a, b in zip(d["ra"], d["dec"])}, t)
    print(f"  (1) rounded-vs-exact comparison finds {len(old)} overlap(s)  [expected 0: the bug]")
    print(f"      exact-vs-exact comparison finds {len(new)} overlap(s)    [expected 1]")
    ok &= (len(old) == 0 and new == {exact})

    # (2) two different roundings of the same true ID
    with open(os.path.join(tmp, "manifest.csv"), "w", newline="") as fh:
        csv.writer(fh).writerow(["object_id"]); csv.writer(fh).writerow([f"{float(exact):.15e}"])
    man = int(float(next(csv.DictReader(open(os.path.join(tmp, "manifest.csv"))))["object_id"]))
    old2 = {rounded} & {man}
    print(f"  (2) float64 rounding {rounded} vs CSV-float rounding {man}: "
          f"old comparison finds {len(old2)} [expected 0]; exact finds "
          f"{len({exact} & {X.rebuild(ra, dec)})} [expected 1]")
    ok &= (len(old2) == 0 and man != rounded)
finally:
    shutil.rmtree(tmp, ignore_errors=True)
print("\n  SELF-TEST", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
