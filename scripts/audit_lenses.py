"""
audit_lenses.py  --  Euclid Q1 lens positive-class audit.

Goal: from q1_discovery_engine_lens_catalog.csv, isolate the positive class the
field uses (Lines et al. 2025: the 497 discovery_engine grade-A/B lenses), then
check how many you can actually build an image for, given the Zenodo archives you
have downloaded. You skipped unsuccess.zip, so this tells you exactly how many
grade-A/B lenses are stranded there and what you still need.

It does NOT cut any stamps. It only reads the CSV and the zip *directory listings*
(no extraction, no decompression -- fast even for the 3.1 GB lens.zip). It writes
one manifest CSV you'll use for the cutout step next.

Run from your repo root (C:\\Users\\krris\\ML\\lens_paper):
    python audit_lenses.py
"""

from pathlib import Path
import zipfile
import sys
import pandas as pd

# ---------------- config ----------------
ROOT = Path("data/q1_lenses")                       # where the CSV + zips live
CATALOG_NAME = "q1_discovery_engine_lens_catalog.csv"
ARCHIVES = ["lens.zip", "group.zip", "recenter.zip", "unsuccess.zip"]  # any that exist
KEEP_SUBSET = "discovery_engine"
KEEP_GRADES = {"A", "B"}
OUT_MANIFEST = ROOT / "lens_manifest_positive.csv"
# ----------------------------------------


def find_catalog(root: Path) -> Path:
    direct = root / CATALOG_NAME
    if direct.exists():
        return direct
    # fall back to a recursive search under root, then under cwd
    for base in (root, Path(".")):
        hits = list(base.rglob(CATALOG_NAME))
        if hits:
            return hits[0]
    sys.exit(f"ERROR: could not find {CATALOG_NAME} under {root} or {Path('.').resolve()}")


def id_from_path(path: str, id_set: set) -> str | None:
    """Return the id_str found in a zip member path, or None.
    Archive layout is <maybe_topfolder>/<id_str>/<id_str>.ext, so the id_str is
    one of the path components (possibly with a .fits/.png suffix on the file)."""
    for part in Path(path).parts:
        if part in id_set:
            return part
        stem = part.rsplit(".", 1)[0]
        if stem in id_set:
            return stem
    return None


def scan_zip(zip_path: Path, id_set: set) -> set:
    found = set()
    try:
        with zipfile.ZipFile(zip_path) as z:
            for name in z.namelist():
                hit = id_from_path(name, id_set)
                if hit:
                    found.add(hit)
    except zipfile.BadZipFile:
        print(f"  WARNING: {zip_path.name} is not a readable zip -- skipped")
    return found


def scan_extracted(root: Path, id_set: set) -> set:
    """In case you already unzipped: match any directory or .fits file whose name
    is an id_str."""
    found = set()
    for p in root.rglob("*"):
        name = p.name.rsplit(".", 1)[0] if p.is_file() else p.name
        if name in id_set:
            found.add(name)
    return found


def main():
    root = ROOT.resolve()
    print(f"ROOT: {root}\n")

    cat_path = find_catalog(ROOT)
    df = pd.read_csv(cat_path, dtype={"id_str": str})
    print(f"Loaded {cat_path.name}: {len(df)} rows, columns = {list(df.columns)}\n")

    for col in ("subset", "id_str", "grade"):
        if col not in df.columns:
            sys.exit(f"ERROR: expected column '{col}' not in catalogue. "
                     f"Got: {list(df.columns)}")

    # full picture so you can verify the headline numbers yourself
    print("Catalogue breakdown (subset x grade):")
    print(pd.crosstab(df["subset"], df["grade"], margins=True).to_string())
    print()

    target = df[(df["subset"] == KEEP_SUBSET) & (df["grade"].isin(KEEP_GRADES))].copy()
    n_target = len(target)
    by_grade = target["grade"].value_counts().to_dict()
    print(f"POSITIVE CLASS (subset=={KEEP_SUBSET}, grade in {sorted(KEEP_GRADES)}): "
          f"{n_target}  (grade A={by_grade.get('A', 0)}, B={by_grade.get('B', 0)})")
    print("  -> Lines et al. report 497 (250 A + 247 B). Confirm this matches.\n")

    id_set = set(target["id_str"])

    # which archives are actually present
    present_archives = [a for a in ARCHIVES if (ROOT / a).exists()]
    missing_archives = [a for a in ARCHIVES if not (ROOT / a).exists()]
    print(f"Archives present: {present_archives}")
    print(f"Archives NOT downloaded: {missing_archives}\n")

    # build availability, tracking which archive each id came from
    source = {}
    for a in present_archives:
        for idn in scan_zip(ROOT / a, id_set):
            source.setdefault(idn, a)
    for idn in scan_extracted(ROOT, id_set):     # any already-unzipped folders
        source.setdefault(idn, "extracted")

    target["available"] = target["id_str"].isin(source)
    target["source"] = target["id_str"].map(source).fillna("MISSING")

    n_avail = int(target["available"].sum())
    n_missing = n_target - n_avail
    print("=" * 60)
    print(f"AVAILABLE now : {n_avail} / {n_target}")
    print(f"MISSING       : {n_missing}  (image only in an archive you don't have)")
    print("=" * 60)

    print("\nAvailable, by source archive:")
    print(target.loc[target["available"], "source"].value_counts().to_string())

    if n_missing:
        miss = target.loc[~target["available"]]
        print(f"\nMissing, by grade: {miss['grade'].value_counts().to_dict()}")
        print("These are almost certainly the grade-A/B lenses whose stamps live in")
        print("unsuccess.zip. To reach the full positive class you either (a) download")
        print("unsuccess.zip and pull ONLY these id_str (never the whole archive), or")
        print("(b) cut them fresh from the Q1 tiles via object_id/tile_index.")
        print("First few missing id_str:", list(miss["id_str"].head(10)))

    # manifest for the cutout step
    keep_cols = [c for c in ["id_str", "grade", "expert_score", "available", "source",
                             "right_ascension", "declination", "tile_index", "object_id"]
                 if c in target.columns]
    target[keep_cols].sort_values(["available", "grade", "id_str"]).to_csv(OUT_MANIFEST, index=False)
    print(f"\nWrote manifest: {OUT_MANIFEST.resolve()}")
    print("Next step uses this file to drive cutout extraction.")


if __name__ == "__main__":
    main()
