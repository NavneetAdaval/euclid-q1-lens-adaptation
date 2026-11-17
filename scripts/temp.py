import os, zipfile

SRC = r"C:\Users\krris\ML\lens_paper\data"     # the folder to archive
OUT = r"C:\Users\krris\ML\lens_paper\data.zip" # output

with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as z:
    for root, dirs, files in os.walk(SRC):
        for f in files:
            full = os.path.join(root, f)
            rel  = os.path.relpath(full, os.path.dirname(SRC))  # keeps a top-level data/
            arc  = rel.replace(os.sep, "/")                     # <-- the fix
            z.write(full, arc)
print("done:", OUT)