"""
samplers.py
===========
Class-balanced batch sampling + the paired source/target iterator for the
label-free sim-to-real strong-lens project (Euclid Q1).

This is PLUMBING only: it decides WHICH indices go into each batch. No DA loss,
no encoder, no learning lives here.

Two pieces, both grounded in PROJECT.md (tags: RB=research-backed, EC=engineering):

  1. ``ClassBalancedBatchSampler`` (RB)
     A batch sampler that builds every batch with a fixed positive fraction
     (default 0.5). Our SOURCE is 1:1 (24k:24k) and the theory that forces the
     balanced target (Generalized Label Shift, Zhao/Tachet arXiv:2003.04475;
     PROJECT.md 4.3) only holds if the batches the aligner sees are actually
     balanced -- so we enforce it at the sampler, not by luck of the draw. When a
     class is scarce (the per-fold target has only ~391 of each latent class) we
     draw WITH REPLACEMENT by reshuffling-and-cycling that class's pool, so a batch
     can always be filled to its target composition.

     For SOURCE we balance on ``indices_by_label`` (real labels {0,1}); for TARGET
     we balance on ``indices_by_latent`` (latent identity {0,1}) because the served
     labels are all UNLABELED(-1) and must never be looked at during training.

  2. ``PairedDomainLoader`` (RB)
     The standard two-loader UDA pattern: iterate the LARGER source loader once per
     epoch (source defines the epoch length) and CYCLE the smaller target loader,
     reshuffling it whenever it is exhausted (~782/fold target vs ~48k source). Each
     step yields ``(source_batch, target_batch)``.

  ``assert_target_unlabeled`` is a cheap guard the training loop calls every step to
  prove the target loader never leaks a real label into a batch.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from lens_dataset import UNLABELED

POS_LABEL, NEG_LABEL = 1, 0


# ---------------------------------------------------------------------------
# per-class cycling pool (the "with replacement when scarce" mechanism)
# ---------------------------------------------------------------------------

class _CyclingPool:
    """Draw ``k`` items from a fixed index pool.

    Items are handed out from a shuffled permutation; when the permutation is
    exhausted it is reshuffled and drawing continues. Across an epoch this is
    exactly sampling-with-replacement for a scarce class (it will reuse indices),
    while a plentiful class is effectively without-replacement within one pass.
    If a single request exceeds the whole pool, we fall back to a direct
    with-replacement draw.
    """

    def __init__(self, items: np.ndarray, rng: np.random.Generator):
        self.items = np.asarray(items, dtype=np.int64)
        if self.items.size == 0:
            raise ValueError("cannot sample from an empty class pool")
        self.rng = rng
        self._perm = self.rng.permutation(self.items)
        self._cur = 0

    def take(self, k: int) -> list[int]:
        if k <= 0:
            return []
        if k > self.items.size:                       # very scarce: direct WITH replacement
            idx = self.rng.integers(0, self.items.size, size=k)
            return self.items[idx].tolist()
        out: list[int] = []
        while k > 0:
            if self._cur >= self._perm.size:          # exhausted -> reshuffle (cycle)
                self._perm = self.rng.permutation(self.items)
                self._cur = 0
            take = min(k, self._perm.size - self._cur)
            out.extend(self._perm[self._cur:self._cur + take].tolist())
            self._cur += take
            k -= take
        return out


# ---------------------------------------------------------------------------
# class-balanced batch sampler
# ---------------------------------------------------------------------------

class ClassBalancedBatchSampler(Sampler):
    """Yield batches (lists of dataset indices) with a fixed positive fraction.

    Parameters
    ----------
    pos_indices, neg_indices : array-like[int]
        Dataset positions of the positive / negative class (from a LensDataset's
        ``indices_by_label`` for source, or ``indices_by_latent`` for target).
    batch_size : int
    num_batches : int | None
        Batches per epoch. Default ``ceil((|pos|+|neg|)/batch_size)`` -- one nominal
        pass over the combined pool. The paired loader overrides this so SOURCE
        defines the epoch length.
    pos_fraction : float
        Fraction of each batch that is positive (default 0.5). Rounded to a whole
        count, clamped to keep at least one of each class in a batch of >= 2.
    seed : int
        Base RNG seed; the effective seed is ``seed + epoch`` so successive epochs
        reshuffle deterministically. Use ``set_epoch`` to control it explicitly.
    """

    def __init__(self, pos_indices, neg_indices, batch_size: int,
                 num_batches: int | None = None, pos_fraction: float = 0.5,
                 seed: int = 0):
        # No argument to super(): the modern torch.utils.data.Sampler takes none. Older
        # versions accepted (and deprecated) a `data_source`; newer ones REMOVE it, so
        # super().__init__(None) would hit object.__init__() and raise TypeError. Passing
        # nothing is safe across both.
        super().__init__()
        self.pos = np.asarray(pos_indices, dtype=np.int64).ravel()
        self.neg = np.asarray(neg_indices, dtype=np.int64).ravel()
        if self.pos.size == 0 or self.neg.size == 0:
            raise ValueError(
                f"need >=1 of each class, got pos={self.pos.size} neg={self.neg.size}")
        if batch_size < 2:
            raise ValueError("batch_size must be >= 2 to hold both classes")
        if not (0.0 < pos_fraction < 1.0):
            raise ValueError("pos_fraction must be in (0,1)")

        self.batch_size = int(batch_size)
        self.pos_fraction = float(pos_fraction)
        # at least one of each class so every batch is genuinely two-class
        self.n_pos = int(min(batch_size - 1, max(1, round(pos_fraction * batch_size))))
        self.n_neg = self.batch_size - self.n_pos

        if num_batches is None:
            num_batches = max(1, -(-(self.pos.size + self.neg.size) // self.batch_size))
        self.num_batches = int(num_batches)

        self.seed = int(seed)
        self.epoch = 0

    # let a training loop pin the shuffle per epoch (reproducibility)
    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    @property
    def actual_pos_fraction(self) -> float:
        return self.n_pos / self.batch_size

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        pos_pool = _CyclingPool(self.pos, rng)
        neg_pool = _CyclingPool(self.neg, rng)
        for _ in range(self.num_batches):
            batch = np.array(pos_pool.take(self.n_pos) + neg_pool.take(self.n_neg),
                             dtype=np.int64)
            rng.shuffle(batch)                        # interleave pos/neg within the batch
            yield batch.tolist()
        self.epoch += 1

    # -- convenience constructors ------------------------------------------

    @classmethod
    def from_dataset(cls, dataset, batch_size: int, use_latent: bool = False,
                     num_batches: int | None = None, pos_fraction: float = 0.5,
                     seed: int = 0) -> "ClassBalancedBatchSampler":
        """Build directly from a LensDataset.

        ``use_latent=False`` balances on real labels (SOURCE roles); ``True`` balances
        on latent identity (TARGET role, whose served labels are all UNLABELED).
        """
        index_map = dataset.indices_by_latent if use_latent else dataset.indices_by_label
        if POS_LABEL not in index_map or NEG_LABEL not in index_map:
            raise ValueError(
                f"dataset index map missing a class (have {sorted(index_map)}); "
                f"use_latent={use_latent}")
        return cls(index_map[POS_LABEL], index_map[NEG_LABEL], batch_size,
                   num_batches=num_batches, pos_fraction=pos_fraction, seed=seed)


def make_balanced_loader(dataset, batch_size: int, use_latent: bool = False,
                         num_batches: int | None = None, pos_fraction: float = 0.5,
                         seed: int = 0, num_workers: int = 0,
                         pin_memory: bool = False,
                         persistent_workers: bool = False) -> DataLoader:
    """A DataLoader whose batches are class-balanced by ``ClassBalancedBatchSampler``.

    Uses ``batch_sampler`` (so ``batch_size``/``shuffle``/``sampler`` are handled by us).
    ``num_workers>0`` parallelises the (CPU-bound) per-image transforms across processes
    and lets the loader PREFETCH the next batch while the GPU is busy -- the image-space
    DA methods are otherwise stalled single-threaded on augmentation. ``persistent_workers``
    keeps those workers (and their in-memory copy of the dataset) alive ACROSS epochs, so
    the ~0.8 GB source dataset is not re-pickled to the workers every epoch (only useful
    with ``num_workers>0``; ignored otherwise).
    """
    sampler = ClassBalancedBatchSampler.from_dataset(
        dataset, batch_size, use_latent=use_latent, num_batches=num_batches,
        pos_fraction=pos_fraction, seed=seed)
    return DataLoader(dataset, batch_sampler=sampler, num_workers=num_workers,
                      pin_memory=pin_memory,
                      persistent_workers=bool(persistent_workers and num_workers > 0))


# ---------------------------------------------------------------------------
# paired source/target iterator
# ---------------------------------------------------------------------------

class PairedDomainLoader:
    """Yield ``(source_batch, target_batch)`` each step.

    Iterates ``source_loader`` once per epoch (source defines the epoch length) and
    cycles ``target_loader``, rebuilding its iterator -- which reshuffles it -- every
    time it runs dry. This is the standard UDA two-loader pattern for a large source
    and a small target.
    """

    def __init__(self, source_loader, target_loader):
        self.source_loader = source_loader
        self.target_loader = target_loader

    def __len__(self) -> int:
        return len(self.source_loader)

    def __iter__(self):
        target_iter = iter(self.target_loader)
        for source_batch in self.source_loader:
            try:
                target_batch = next(target_iter)
            except StopIteration:
                target_iter = iter(self.target_loader)   # exhausted -> reshuffle target
                target_batch = next(target_iter)
            yield source_batch, target_batch


# ---------------------------------------------------------------------------
# label-free guard
# ---------------------------------------------------------------------------

def assert_target_unlabeled(target_batch: dict, unlabeled: int = UNLABELED):
    """Raise if any label in a target batch is not the UNLABELED sentinel.

    The training loop calls this every step: it proves LensDataset's -1 survives
    default collation and that no real target label can ever reach the DA loss.
    """
    labels = target_batch["label"]
    if not torch.is_tensor(labels):
        labels = torch.as_tensor(labels)
    if not torch.all(labels == unlabeled):
        bad = labels[labels != unlabeled].tolist()
        raise AssertionError(
            f"target batch leaked non-UNLABELED labels {bad[:8]}"
            f"{' ...' if len(bad) > 8 else ''} (expected all == {unlabeled})")
