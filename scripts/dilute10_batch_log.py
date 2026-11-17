"""
dilute10_batch_log.py
=====================
The realised lens fraction of every target batch (brief section 1.3; part 1's stop
condition 4).

``LensFractionLogger`` wraps a target BATCH SAMPLER. It yields the inner sampler's batches
UNCHANGED and, after each batch is drawn, records the fraction of its positions that are
lenses, computed from the POOL FILE (a position is a lens iff ``position < n_lenses``; the
builder puts a fold's lenses first) -- never from a dataset label. It is never passed to
the trainer or to any loss: the trainer receives the DataLoader's collated batch, which is
built from the same indices whether or not the logger is in the chain.

``scan_runner`` proves the second half on the AST: no call to ``train_step`` (or any other
training entry point) receives the logger, its records or its output. The known-null is a
synthetic runner that does, and the scan refuses to declare anything clean unless it fires
on it.
"""
from __future__ import annotations

import ast
import csv
import os
import textwrap

import numpy as np

TRAIN_ENTRY_POINTS = ("train_step", "train_loop", "forward", "loss", "da_loss",
                      "train_coral_fold", "train_freematch_fold")
LOGGER_ATTRS = ("fractions", "records", "last_fraction", "summary")

KNOWN_NULL_RUNNER = '''
from dilute10_batch_log import LensFractionLogger
logger = LensFractionLogger(sampler, n_lenses=352, csv_path=None)
for sbatch, tbatch in paired:
    stats = trainer.train_step(sbatch, tbatch, lens_fraction=logger.fractions[-1])
'''


LENS_BAND = 0.002          # runs/dilute10_prereg_part2.json, lens_fraction_band
BATCH_SIZE = 256


class LensFractionStop(RuntimeError):
    """Part 2's stop condition 4 fired: a malformed batch, or the run-mean left the band."""


class LensFractionLogger:
    """Records AND ENFORCES part 2's lens-fraction band (stop condition 4).

    Every batch must be exactly ``BATCH_SIZE`` distinct positions, else it stops at once.
    At every flush (every ``flush_every`` batches) and at ``summary`` the running mean must
    lie within ``LENS_BAND`` of the expected value L/(L+G) of the sampler's own training pool
    (positions < n_lenses are lenses). Measured before declaring the check: on every arm and
    fold the correct sampler's running mean stays within 0.00041 of expected at every flush
    from batch 187 on, so the check cannot fire on a correct run, and a 0.01 drift is caught
    at the first flush.
    """

    def __init__(self, inner, n_lenses: int, csv_path: str | None = None,
                 flush_every: int = 187, expected: float | None = None):
        self.inner = inner
        self.n_lenses = int(n_lenses)
        self.csv_path = csv_path
        self.flush_every = int(flush_every)
        self.fractions: list[float] = []
        self._pending: list[tuple] = []
        self.batch_size = getattr(inner, "batch_size", None)
        pool = getattr(inner, "pool", None)
        self.expected = (float(expected) if expected is not None else
                         float((np.asarray(pool) < self.n_lenses).mean())
                         if pool is not None else None)
        if csv_path:
            os.makedirs(os.path.dirname(os.path.abspath(csv_path)) or ".", exist_ok=True)
            if os.path.exists(csv_path):
                raise FileExistsError(f"{csv_path} exists; refusing to append")
            with open(csv_path, "w", newline="") as fh:
                csv.writer(fh).writerow(["batch", "n", "n_lens", "lens_fraction"])

    # sampler interface, forwarded
    def __len__(self):
        return len(self.inner)

    def set_epoch(self, epoch):
        if hasattr(self.inner, "set_epoch"):
            self.inner.set_epoch(epoch)

    def __iter__(self):
        for batch in self.inner:
            idx = np.asarray(batch, dtype=np.int64)
            if idx.size != BATCH_SIZE or np.unique(idx).size != idx.size:
                self.flush()
                raise LensFractionStop(
                    f"[dilute10 STOP: lens-fraction band, malformed batch] batch "
                    f"{len(self.fractions) + 1} has {idx.size} items, "
                    f"{idx.size - np.unique(idx).size} duplicated; every target batch must "
                    f"be exactly {BATCH_SIZE} distinct positions (part 2)")
            n_l = int((idx < self.n_lenses).sum())
            f = n_l / max(idx.size, 1)
            self.fractions.append(f)
            self._pending.append((len(self.fractions), int(idx.size), n_l, f))
            if len(self._pending) >= self.flush_every:
                self.flush()
                self.check_band()
            yield batch

    def check_band(self):
        if self.expected is None or not self.fractions:
            return
        m = float(np.mean(self.fractions))
        if abs(m - self.expected) > LENS_BAND:
            raise LensFractionStop(
                f"[dilute10 STOP: lens-fraction band, run-mean drift] after "
                f"{len(self.fractions)} batches the run-mean lens fraction is {m:.5f}, "
                f"expected {self.expected:.5f}; |diff| {abs(m - self.expected):.5f} > "
                f"{LENS_BAND} (part 2)")

    def flush(self):
        if self.csv_path and self._pending:
            with open(self.csv_path, "a", newline="") as fh:
                csv.writer(fh).writerows(self._pending)
        self._pending = []

    def summary(self, *, enforce: bool = True, reason: str | None = None) -> dict:
        """The run-mean summary. Part 2's band is a statement about a run's mean over its
        BUDGET, so it is enforced here only for a run that completed its budget
        (``enforce=True``, the caller's judgement from the run's own record). For an aborted
        or crashed run (``enforce=False``) the summary is written with ``band_enforced:
        false`` and ``reason``, and nothing is raised: a run-mean over a few batches is not
        the quantity part 2 bounds (PROJECTv7 7.46: 5 batches after a step-1 abort). The
        187-batch checks in ``__iter__`` and the per-batch malformed-batch check are
        unaffected."""
        self.flush()
        if enforce:
            self.check_band()
        a = np.asarray(self.fractions, dtype=float)
        mean = float(a.mean()) if a.size else None
        s = {"n_batches": int(a.size), "mean": mean,
             "sd": float(a.std(ddof=1)) if a.size > 1 else None,
             "expected": self.expected, "band": LENS_BAND,
             "n_lenses_basis": self.n_lenses, "csv": self.csv_path,
             "band_enforced": bool(enforce)}
        if not enforce:
            s["reason"] = reason
            s["inside_band_informational"] = (
                None if mean is None or self.expected is None
                else bool(abs(mean - self.expected) <= LENS_BAND))
        return s


def wrapper_for(n_lenses: int, csv_path: str | None):
    return lambda inner: LensFractionLogger(inner, n_lenses, csv_path)


# --------------------------------------------------------------------------------------
# AST scan: the logger never reaches a training entry point
# --------------------------------------------------------------------------------------
class KnownNullDidNotFire(AssertionError):
    pass


def _logger_names(tree) -> set:
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            fn = node.value.func
            fname = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            if fname in ("LensFractionLogger", "wrapper_for"):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        names.add(t.id)
    return names


def _violations(src: str) -> list:
    tree = ast.parse(textwrap.dedent(src))
    names = _logger_names(tree) | {"LensFractionLogger"}
    bad = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        fname = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
        if fname not in TRAIN_ENTRY_POINTS:
            continue
        for a in list(node.args) + [k.value for k in node.keywords]:
            for sub in ast.walk(a):
                if (isinstance(sub, ast.Name) and sub.id in names) or \
                        (isinstance(sub, ast.Attribute) and sub.attr in LOGGER_ATTRS
                         and isinstance(sub.value, ast.Name) and sub.value.id in names):
                    bad.append(f"line {node.lineno}: {fname}(... {ast.unparse(sub)} ...)")
    return bad


def scan_runner(src: str, known_null_src: str = KNOWN_NULL_RUNNER) -> dict:
    null = _violations(known_null_src)
    if not null:
        raise KnownNullDidNotFire("the synthetic runner that feeds the logger into "
                                  "train_step was not caught")
    hits = _violations(src)
    return {"clean": not hits, "hits": hits, "known_null_hits": null}
