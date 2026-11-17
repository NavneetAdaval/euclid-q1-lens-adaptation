"""Sanity check ESA Euclid TAP service and inspect MER catalog schema."""
from astroquery.esa.euclid import Euclid

print("=== ESA Euclid TAP status ===")
try:
    msgs = Euclid.get_status_messages()
    print(msgs if msgs else "(no status messages — service likely healthy)")
except Exception as e:
    print(f"Status check error: {e}")

# print("\n=== Listing tables (filtering for MER) ===")
# tables = Euclid.load_tables(only_names=True)
# for t in tables:
#     name = t.name if hasattr(t, "name") else str(t)
#     if "mer" in name.lower():
#         print(f"  {name}")
#
# print("\n=== Columns of q1.mer_final_catalogue ===")
# table = Euclid.load_table("q1.mer_final_catalogue")
# for col in table.columns:
#     desc = (col.description or "")[:80]
#     print(f"  {col.name:<35} {desc}")

print("\n=== Listing all q1.* and sedm.* tables ===")
tables = Euclid.load_tables(only_names=True)
for t in tables:
    name = t.name if hasattr(t, "name") else str(t)
    if name.startswith("q1.") or name.startswith("sedm."):
        print(f"  {name}")

print("\n=== Searching for mosaic/image product tables ===")
for t in tables:
    name = t.name if hasattr(t, "name") else str(t)
    if any(k in name.lower() for k in ["mosaic", "image", "product", "vis"]):
        print(f"  {name}")