"""
split_output_parts.py
=====================
Package a Kaggle output directory as a tar SPLIT INTO DOWNLOADABLE PARTS, with a per-part
sha256 manifest.

WHY THIS EXISTS. The fold-0 development run produced a 1.95 GB checkpoint tarball that
Kaggle's browser download would not deliver. Splitting it into 900 MiB parts worked -- the
same size that worked for the sweep bundle before it (`part_aa` / `part_ab`) -- but the
parts had to be reassembled and there was no way to tell a truncated download from a good
one. Now the checksums ship with the parts, so a bad download fails loudly at `sha256sum -c`
instead of yielding a corrupt archive.

The archive is written STREAMING, straight into the parts: no intermediate full-size tar,
so packaging needs no extra disk and works even when /kaggle/working is already most of the
way full. The sha256 of every part AND of the whole reassembled stream are computed in the
same pass.

Reassemble and verify, locally:

    cat <name>.tar.part_a* > <name>.tar
    sha256sum -c PARTS.sha256
    tar xf <name>.tar -C runs/

Run:
    python -u scripts/split_output_parts.py --src /kaggle/working/runs/coral_r4_confirm \
        --arcname coral_r4_confirm --dest /kaggle/working/parts
    python -u scripts/split_output_parts.py --self-test
"""
from __future__ import annotations

import argparse
import hashlib
import io
import os
import shutil
import sys
import tarfile
import tempfile

DEFAULT_CHUNK = 900 * 1024 * 1024        # 900 MiB -- the size measured to download cleanly


class ChunkedWriter(io.RawIOBase):
    """A write-only file object that rolls over to a new file every ``chunk`` bytes.

    Used as tarfile's fileobj in stream mode ('w|'), which never seeks, so the archive is
    produced directly as parts.
    """

    def __init__(self, prefix, chunk, verbose=True):
        self.prefix, self.chunk, self.verbose = prefix, int(chunk), verbose
        self.idx = self.written = 0
        self.fh = None
        self.parts = []                  # (name, size, sha256)
        self.whole = hashlib.sha256()

    def writable(self):
        return True

    def _open(self):
        if self.idx >= 26:
            raise RuntimeError("more than 26 parts; raise --chunk")
        self.name_ = f"{os.path.basename(self.prefix)}.part_a{chr(ord('a') + self.idx)}"
        self.path = os.path.join(os.path.dirname(self.prefix), self.name_)
        self.fh = open(self.path, "wb")
        self._h = hashlib.sha256()
        self.written = 0

    def _close_part(self):
        if self.fh is None:
            return
        self.fh.close()
        size = os.path.getsize(self.path)
        self.parts.append((self.name_, size, self._h.hexdigest()))
        if self.verbose:
            print(f"  {self.name_}  {size / 2 ** 20:.0f} MiB", flush=True)
        self.fh = None
        self.idx += 1

    def write(self, b):
        b = bytes(b)
        self.whole.update(b)
        pos = 0
        while pos < len(b):
            if self.fh is None:
                self._open()
            take = b[pos:pos + (self.chunk - self.written)]
            self.fh.write(take)
            self._h.update(take)
            self.written += len(take)
            pos += len(take)
            if self.written >= self.chunk:
                self._close_part()
        return len(b)

    def close(self):
        self._close_part()
        super().close()


def split_tar(src_dir, arcname, out_dir, chunk=DEFAULT_CHUNK, base=None, exclude_suffix=None,
              verbose=True):
    """Tar ``src_dir`` into ``out_dir`` as parts. Returns (parts, whole_sha, total, manifest).

    ``arcname`` is the path INSIDE the tar; ``base`` names the archive file. They are
    separate on purpose -- an arcname containing a separator would otherwise send the part
    files to a directory that does not exist.
    """
    os.makedirs(out_dir, exist_ok=True)
    base = base or os.path.basename(str(arcname).rstrip("/\\"))
    prefix = os.path.join(out_dir, f"{base}.tar")
    w = ChunkedWriter(prefix, chunk, verbose=verbose)
    filt = None
    if exclude_suffix:
        def filt(ti):
            return None if ti.name.endswith(tuple(exclude_suffix)) else ti
    with tarfile.open(fileobj=w, mode="w|") as t:
        t.add(src_dir, arcname=arcname, filter=filt)
    w.close()

    total = sum(s for _n, s, _h in w.parts)
    manifest = os.path.join(out_dir, "PARTS.sha256")
    with open(manifest, "w", newline="\n") as fh:
        fh.write(f"# cat {base}.tar.part_a* > {base}.tar\n")
        fh.write(f"# sha256sum -c PARTS.sha256\n")
        fh.write(f"# tar xf {base}.tar -C runs/\n")
        fh.write(f"# {len(w.parts)} part(s), {total} bytes total\n")
        for n, _s, h in w.parts:
            fh.write(f"{h}  {n}\n")
        fh.write(f"{w.whole.hexdigest()}  {base}.tar\n")
    if verbose:
        print(f"\n{len(w.parts)} part(s) + PARTS.sha256 in {out_dir}  "
              f"({total / 2 ** 30:.2f} GiB)", flush=True)
    return w.parts, w.whole.hexdigest(), total, manifest


def self_test():
    """Prove the round-trip: parts -> cat -> identical tar -> extracts byte-identically."""
    import numpy as np
    ok = True
    td = tempfile.mkdtemp(prefix="split_selftest_")
    try:
        run = "coral_r4_confirm"
        src = os.path.join(td, run, "fold1", "checkpoints")
        os.makedirs(src)
        rng = np.random.default_rng(0)
        for step in range(250, 2001, 250):
            with open(os.path.join(src, f"step_{step}.pt"), "wb") as fh:
                fh.write(rng.integers(0, 256, 300_000, dtype=np.uint8).tobytes())
        out = os.path.join(td, "parts")
        parts, whole, total, man = split_tar(os.path.join(td, run), run, out,
                                             chunk=500_000, base=run)
        tar = os.path.join(out, f"{run}.tar")
        with open(tar, "wb") as o:
            for n, _s, _h in parts:
                with open(os.path.join(out, n), "rb") as i:
                    shutil.copyfileobj(i, o)
        h = hashlib.sha256()
        with open(tar, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 20), b""):
                h.update(b)
        checks = [("more than one part was produced", len(parts) > 1),
                  ("reassembled sha == whole-stream sha", h.hexdigest() == whole),
                  ("reassembled size == total", os.path.getsize(tar) == total)]
        ex = os.path.join(td, "ex")
        os.makedirs(ex)
        with tarfile.open(tar) as t:
            t.extractall(ex)
        got = os.path.join(ex, run, "fold1", "checkpoints")
        files = sorted(os.listdir(got))
        same = all(open(os.path.join(src, f), "rb").read()
                   == open(os.path.join(got, f), "rb").read() for f in files)
        checks += [(f"tar extracts {len(files)} files byte-identically",
                    same and len(files) == 8),
                   ("member paths drop into runs/ unrenamed",
                    tarfile.open(tar).getnames()[0] == run)]
        # a corrupted part must fail the manifest
        p0 = os.path.join(out, parts[0][0])
        data = bytearray(open(p0, "rb").read())
        data[123] ^= 0xFF
        open(p0, "wb").write(bytes(data))
        bad = hashlib.sha256(open(p0, "rb").read()).hexdigest()
        checks.append(("a single flipped byte changes the part's sha256",
                       bad != parts[0][2]))
        for msg, cond in checks:
            print(f"  {'PASS' if cond else 'FAIL'}  {msg}")
            ok &= bool(cond)
    finally:
        shutil.rmtree(td, ignore_errors=True)
    print(f"\nself-test: {'PASSED' if ok else 'FAILED'}")
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", help="directory to package")
    ap.add_argument("--arcname", default=None, help="path inside the tar (default: basename)")
    ap.add_argument("--dest", default=None, help="where the parts go")
    ap.add_argument("--chunk", type=int, default=DEFAULT_CHUNK)
    ap.add_argument("--exclude-suffix", nargs="*", default=None,
                    help="e.g. .pt to leave checkpoints out of a light archive")
    ap.add_argument("--self-test", action="store_true", default=False)
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if not a.src:
        ap.error("--src is required (or pass --self-test)")
    arc = a.arcname or os.path.basename(os.path.normpath(a.src))
    dest = a.dest or os.path.join(os.path.dirname(os.path.normpath(a.src)), "parts")
    # `arc` is the path INSIDE the tar and may contain separators (coral_r4_confirm/
    # fold1); the part FILES must land flat in `dest`. Passing base=arc made the
    # writer try to open dest/coral_r4_confirm/fold1.tar.part_aa, in a directory that
    # does not exist -- FileNotFoundError mid-stream, after the tar had started.
    base = os.path.basename(arc.rstrip('/')) or 'archive'
    split_tar(a.src, arc, dest, chunk=a.chunk, base=base,
              exclude_suffix=a.exclude_suffix)
    return 0


if __name__ == "__main__":
    sys.exit(main())
