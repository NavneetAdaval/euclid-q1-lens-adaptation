"""
uniform_target_sampler.py
=========================
The label-free TARGET batch sampler of the 1:10 diluted-target experiment (brief
section 1.2).

``UniformTargetBatchSampler`` yields batches of exactly ``batch_size`` dataset positions,
each drawn UNIFORMLY WITHOUT REPLACEMENT from a training pool of positions. Draws walk a
permutation of the pool; when a pass is exhausted the pool is reshuffled and drawing
continues (the ``_CyclingPool`` shape), so the batch count is fixed and never ragged.
Across one pass every pool member is drawn exactly once. At a pass boundary a batch takes
the rest of the old pass plus the head of the new one; any member that would appear twice
in that batch is swapped with the next eligible member of the new permutation, so no batch
ever holds a position twice and each pass is still a permutation.

It takes NO labels, NO latent map and NO ``pos_fraction``: its only inputs are positions,
a batch size, a batch count and a seed. ``label_scan`` proves it on the AST, with
``samplers.ClassBalancedBatchSampler`` (which reads ``indices_by_latent``) as the known-null
that must fire.

RNG. A dedicated ``numpy.random.Generator`` from
``SeedSequence([seed, NAMESPACE_INT])``: it shares no stream with the source sampler (its
own ``default_rng(seed + epoch)``), with the augmentation (torch's RNG) or with torch's
global RNG. The cycling state persists across epochs; ``set_epoch`` is accepted for
interface compatibility and changes nothing.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import textwrap

import numpy as np
from torch.utils.data import DataLoader, Sampler

NAMESPACE = "dilute10/uniform_target_sampler/v1"
NAMESPACE_INT = int.from_bytes(hashlib.sha256(NAMESPACE.encode()).digest()[:8], "little")

# identifiers whose presence in the sampler's code would mean it can see a label
LABEL_IDENTIFIERS = ("indices_by_latent", "indices_by_label", "latent_labels", "latent",
                     "labels", "label", "pos_fraction", "pos_indices", "neg_indices",
                     "n_pos", "n_neg", "use_latent")


class UniformTargetBatchSampler(Sampler):
    def __init__(self, pool_positions, batch_size: int, num_batches: int, seed: int):
        super().__init__()
        self.pool = np.asarray(pool_positions, dtype=np.int64).ravel()
        if self.pool.size == 0:
            raise ValueError("empty training pool")
        if len(set(self.pool.tolist())) != self.pool.size:
            raise ValueError("training pool holds a position twice")
        self.batch_size = int(batch_size)
        if not (0 < self.batch_size <= self.pool.size):
            raise ValueError(f"batch_size {batch_size} must be in (0, {self.pool.size}]")
        self.num_batches = int(num_batches)
        self.seed = int(seed)
        self.rng = np.random.Generator(np.random.PCG64(
            np.random.SeedSequence([self.seed, NAMESPACE_INT])))
        self._perm = self.rng.permutation(self.pool)
        self._cur = 0
        self.n_passes_started = 1
        self.n_boundary_swaps = 0

    def set_epoch(self, epoch: int):
        """Accepted for interface compatibility; the cycling state is NOT reset."""
        return None

    def __len__(self) -> int:
        return self.num_batches

    def _next_batch(self) -> list[int]:
        k = self.batch_size
        rest = self._perm.size - self._cur
        if rest >= k:
            out = self._perm[self._cur:self._cur + k]
            self._cur += k
            return out.tolist()
        head = self._perm[self._cur:]                       # end of this pass
        perm = self.rng.permutation(self.pool)              # the next pass
        need = k - head.size
        taken = set(head.tolist())
        # move members already in `head` out of the first `need` slots of the new pass
        j = need
        for i in range(need):
            if int(perm[i]) in taken:
                while int(perm[j]) in taken:
                    j += 1
                perm[i], perm[j] = perm[j], perm[i]
                j += 1
                self.n_boundary_swaps += 1
        self._perm, self._cur = perm, need
        self.n_passes_started += 1
        return head.tolist() + perm[:need].tolist()

    def __iter__(self):
        for _ in range(self.num_batches):
            yield self._next_batch()


def make_uniform_target_loader(dataset, pool_positions, batch_size, num_batches, seed, *,
                               num_workers=0, pin_memory=False, persistent_workers=False,
                               wrap=None):
    """A DataLoader over ``dataset`` whose batches come from the uniform sampler.

    ``wrap`` (optional) wraps the batch sampler -- the lens-fraction logger uses it to
    observe drawn indices; it must yield the inner sampler's batches unchanged.
    """
    bs = UniformTargetBatchSampler(pool_positions, batch_size, num_batches, seed)
    if wrap is not None:
        bs = wrap(bs)
    return DataLoader(dataset, batch_sampler=bs, num_workers=int(num_workers),
                      pin_memory=bool(pin_memory),
                      persistent_workers=bool(persistent_workers and int(num_workers) > 0))


# --------------------------------------------------------------------------------------
# the label scan (AST), with a mandatory known-null
# --------------------------------------------------------------------------------------
class KnownNullDidNotFire(AssertionError):
    pass


def _identifiers(src: str) -> set:
    tree = ast.parse(textwrap.dedent(src))
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            out.add(node.id)
        elif isinstance(node, ast.Attribute):
            out.add(node.attr)
        elif isinstance(node, ast.arg):
            out.add(node.arg)
        elif isinstance(node, ast.keyword) and node.arg:
            out.add(node.arg)
    return {s.lower() for s in out}


def label_scan(target_src: str | None = None, known_null_src: str | None = None) -> dict:
    """AST identifiers of the sampler vs LABEL_IDENTIFIERS, with the balanced sampler as the
    known-null. Raises if the known-null matches nothing."""
    import samplers
    if target_src is None:
        target_src = inspect.getsource(UniformTargetBatchSampler)
    if known_null_src is None:
        known_null_src = inspect.getsource(samplers.ClassBalancedBatchSampler)
    toks = {t.lower() for t in LABEL_IDENTIFIERS}
    null_hits = sorted(_identifiers(known_null_src) & toks)
    if not null_hits:
        raise KnownNullDidNotFire("the balanced-sampler known-null matched no label "
                                  "identifier, so the scan proves nothing")
    hits = sorted(_identifiers(target_src) & toks)
    return {"clean": not hits, "hits": hits, "known_null_hits": null_hits,
            "method": "ast identifiers (Name, Attribute, arg, keyword)"}
