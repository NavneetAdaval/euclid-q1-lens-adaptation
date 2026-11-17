"""
zoobot_encoder.py
=================
FROZEN feature extractor for the label-free sim-to-real strong-lens project (Euclid Q1).

Wraps the Zoobot greyscale ConvNeXT-Nano as a fixed 640-dim encoder. Everything here is
grounded in PROJECT.md (tags: RB=research-backed, EC=engineering-choice, FLAG=deviation
from the literal spec that was forced by a measurement on the real checkpoint).

  * timm-ONLY LOAD (RB) -- Zoobot encoders are published as plain timm models, so we load
      with `timm.create_model(SLUG, pretrained=True, num_classes=0)` and never import the
      `zoobot` package. That avoids its Lightning/finetuning machinery and its documented
      timm-version checkpoint fragility.

  * GREYSCALE SLUG (RB) -- `...zoobot-encoder-greyscale-convnext_nano` is the single-channel
      model for Euclid VIS. The plain `convnext_nano` in most docs is 3-band and would be
      the wrong encoder.

  * in_chans=1 (FLAG, forced) -- MEASURED: the greyscale checkpoint's stem is (80,1,4,4),
      but `timm.create_model(...)` alone builds a 3-channel stem (80,3,4,4) and the state_dict
      load RAISES a size-mismatch. `in_chans=1` is therefore REQUIRED to load the real
      weights; it is not optional. (This is the one addition to the literal load call in the
      spec, and it is mandatory, not a preference.)

  * NO INPUT STRETCH, NO NORMALIZATION (RB) -- input is (B,1,224,224) ALREADY stretched to
      [0,1] by LensTransform/LensDataset. Re-stretching or applying mean/std would corrupt
      the per-image percentiles the stretch established (and would fold in the augmented/
      padded borders). We assert the input is rank-4 single-channel and pass it straight in.

  * IGNORE pretrained_cfg TRANSFORM (RB) -- MEASURED: this checkpoint's pretrained_cfg
      reports ImageNet mean/std = [0.485,0.456,0.406]/[0.229,0.224,0.225] and input_size
      [3,224,224]. That is inherited timm metadata, NOT what Zoobot trained on. We print it
      once for inspection and deliberately do not apply it.

  * EXPLICIT POOL + SHAPE GUARD (RB) -- we do not assume `num_classes=0` pools. We run
      `forward_features` -> `forward_head(pre_logits=True)` (global-pool + ConvNeXt head-norm,
      classifier skipped) and assert the result is exactly (B, 640). Any other shape fails
      loudly (guards against a silent config/checkpoint drift).

  * FROZEN, EXTENSIBLE (EC) -- `.eval()`, every param `requires_grad=False`, forward under
      `torch.no_grad()`. `encoder_mode` is plumbed through so 'lp_then_lastblock' /
      'lp_then_lora' can be switched on later WITHOUT a rewrite, but only 'frozen' is
      implemented now; the others raise NotImplementedError.

  * PARTIAL UNFREEZE via `n_blocks` (RB, Lever 1) -- `n_blocks=0` (DEFAULT) is the frozen
      encoder above, bit-for-bit. `n_blocks>0` makes the top `n_blocks` entries of Zoobot's
      OWN `tuneable_blocks` list trainable, replicating
      `zoobot/pytorch/training/finetune.py` (v2.0.1,
      `FinetuneableZoobotAbstract.configure_optimizers`) EXACTLY:

          tuneable_blocks = [encoder.stem] + [stage for stage in encoder.stages]  # 5 entries
          tuneable_blocks.reverse()                 # highest layer first
          blocks_to_tune = tuneable_blocks[:n_blocks]

      So `n_blocks=3` == `stages[3], stages[2], stages[1]` -- timm STAGES, not ConvNeXt
      residual blocks. MEASURED trainable encoder params (timm 1.0.26, convnext_nano,
      in_chans=1, total 14,950,000 -- matches the 14.95M above):

          n_blocks : 0 -> 0 | 1 -> 7,448,320 (49.8%) | 2 -> 14,355,840 (96.0%)
                     3 -> 14,835,520 (99.2%) | 4 -> 14,947,200 | 5 -> 14,948,720

      TWO CONSEQUENCES worth internalising: (a) `n_blocks=3` (Lines et al.'s setting) is
      EFFECTIVELY FULL FINE-TUNING, not a partial unfreeze; (b) `head.norm` (the final
      LayerNorm2d, 1,280 params) sits OUTSIDE Zoobot's `tuneable_blocks` and is therefore
      NEVER tuned at any `n_blocks` -- we replicate that rather than "fixing" it, because
      making it trainable would be a silent deviation from the benchmark protocol.

  * requires_grad=False, NOT optimizer-omission (EC, deliberate) -- Zoobot itself never sets
      `requires_grad=False`; it simply omits frozen params from the optimizer. The two give
      the IDENTICAL weight trajectory (no grad => no update, and no weight decay applied to
      those params either), but `requires_grad=False` additionally stops autograd at the
      first trainable module, so it actually saves backward compute. We use it.

  * NO BatchNorm ANYWHERE (RB, measured) -- ConvNeXt is LayerNorm throughout; MEASURED on
      this build: 0 BatchNorm modules, 0 active Dropout, and every `drop_path` is an
      Identity (stochastic depth off). So train() vs eval() is a NO-OP for this encoder and
      the usual train/eval BN-running-statistics domain-adaptation footgun does not apply.
      We therefore keep `.eval()` even when partially unfrozen -- deterministic, and no
      BN-freezing logic is needed (or added).
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    import timm
except ImportError as e:  # pragma: no cover
    raise ImportError("zoobot_encoder requires `timm` (pip install timm)") from e

# ---------------------------------------------------------------------------
# Constants (all measured against the real checkpoint)
# ---------------------------------------------------------------------------
MODEL_SLUG = "hf_hub:mwalmsley/zoobot-encoder-greyscale-convnext_nano"
IN_CHANS = 1            # greyscale VIS; REQUIRED to load the checkpoint (see module docstring)
INPUT_SIZE = 224        # encoder input side (Zoobot ConvNeXT)
EMBED_DIM = 640         # timm convnext_nano final feature dim

FROZEN = "frozen"
_FUTURE_MODES = ("lp_then_lastblock", "lp_then_lora")   # plumbed, not implemented yet
SUPPORTED_MODES = (FROZEN, *_FUTURE_MODES)

# MEASURED trainable-encoder param counts per n_blocks (timm 1.0.26, convnext_nano,
# in_chans=1). This is the regression anchor test_encoder_unfreeze.py pins.
TRAINABLE_PARAMS_BY_N_BLOCKS = {
    0: 0, 1: 7_448_320, 2: 14_355_840, 3: 14_835_520, 4: 14_947_200, 5: 14_948_720}


class ZoobotEncoder(nn.Module):
    """640-dim Zoobot greyscale ConvNeXT-Nano feature extractor (frozen by default).

    Parameters
    ----------
    encoder_mode : str
        'frozen' (default, fully implemented). 'lp_then_lastblock' / 'lp_then_lora' are
        accepted names reserved for future partial-unfreezing but raise NotImplementedError.
    n_blocks : int
        How many of Zoobot's OWN ``tuneable_blocks`` (top-first: stages[3], stages[2],
        stages[1], stages[0], stem) to make trainable. ``0`` (default) == the fully frozen
        encoder, bit-for-bit unchanged. See the module docstring for the exact convention
        and the measured param counts.
    device : str | torch.device | None
        Optional device to move the model to at construction.
    verbose : bool
        Print the pretrained_cfg (input_size/mean/std) once at load for inspection.
    self_check : bool
        Run a one-time dummy forward at construction to assert the (B,640) contract early.
    """

    def __init__(self, encoder_mode: str = FROZEN, n_blocks: int = 0, device=None,
                 verbose: bool = True, self_check: bool = True,
                 grad_checkpointing: bool = False):
        super().__init__()
        if encoder_mode not in SUPPORTED_MODES:
            raise ValueError(f"encoder_mode must be one of {SUPPORTED_MODES}, got {encoder_mode!r}")
        self.encoder_mode = encoder_mode

        # ---- load the real timm model (weights download on first run) ----
        self.model = timm.create_model(
            MODEL_SLUG, pretrained=True, num_classes=0, in_chans=IN_CHANS)
        self.pretrained_cfg = dict(getattr(self.model, "pretrained_cfg", {}) or {})
        self.cfg_mean = self.pretrained_cfg.get("mean")
        self.cfg_std = self.pretrained_cfg.get("std")
        self.cfg_input_size = self.pretrained_cfg.get("input_size")
        self.embed_dim = EMBED_DIM

        # ---- Zoobot's tuneable_blocks, built EXACTLY as finetune.py does ----
        # [stem] + [stage for stage in stages], then reversed => highest layer first.
        # Exposed as ``tuneable_blocks_in_lr_order`` so the optimizer builder in
        # train_skeleton.py consumes THIS ordering instead of re-deriving it (one source
        # of truth: the LR-decay index and the unfreeze index cannot drift apart).
        blocks = ([("stem", self.model.stem)]
                  + [(f"stages.{i}", s) for i, s in enumerate(self.model.stages)])
        blocks.reverse()
        self.tuneable_blocks_in_lr_order: list[tuple[str, nn.Module]] = blocks

        n_blocks = int(n_blocks)
        assert 0 <= n_blocks <= len(self.tuneable_blocks_in_lr_order), (
            f"n_blocks={n_blocks} is out of range: this encoder exposes exactly "
            f"{len(self.tuneable_blocks_in_lr_order)} tuneable blocks "
            f"({[n for n, _ in self.tuneable_blocks_in_lr_order]}), so n_blocks must be "
            f"in [0, {len(self.tuneable_blocks_in_lr_order)}].")
        self.n_blocks = n_blocks

        if verbose:
            self._print_cfg_once()

        # ---- apply the encoder mode (only 'frozen' now) ----
        if self.encoder_mode == FROZEN:
            self._apply_frozen()
        else:  # pragma: no cover - reserved future modes
            raise NotImplementedError(
                f"encoder_mode={self.encoder_mode!r} is reserved but not implemented; "
                f"only {FROZEN!r} is available now.")

        self.grad_checkpointing = False
        self.set_grad_checkpointing(bool(grad_checkpointing))

        if device is not None:
            self.to(device)

        if self_check:
            self._self_check(device)

    # -- gradient checkpointing (memory <-> recompute; NUMERICALLY EXACT) ----

    def set_grad_checkpointing(self, enable: bool = True):
        """Trade activation memory for recompute in the backward pass.

        WHY THIS AND NOT GRADIENT ACCUMULATION. At batch 256 the stored activations for a
        backward through stages[1..3] exceed a 6 GB card. Micro-batched accumulation would
        also fit, but it is EXACT ONLY for a per-sample-decomposable loss like Baseline A's
        BCE. Three of the four bake-off methods are BATCH-COUPLED by construction:
          * CoralLoss matches a 640x640 feature covariance WITHIN the batch -- at micro-batch
            32 that is a rank-32 estimate, eight times, not a noisier rank-256 estimate. A
            different objective, not a noisier one.
          * FreeMatchLoss's self-adaptive threshold is an EMA over per-batch confidence
            statistics -- micro-batching gives 8x more updates from 8x noisier estimates and
            changes the threshold trajectory.
          * AdaMatchLoss's distribution alignment uses per-batch source/target prediction
            statistics.
        PROJECT.md 7.4 pins batch size precisely because it is a known result-flipping
        confound, so accumulation is not available to us. Checkpointing recomputes
        activations in backward and touches NOTHING about batch composition, so the batch-256
        match is preserved exactly. Cost is ~30% step time (MEASURED below in the runner).

        NO-OP AT n_blocks=0 (hard requirement): with a fully frozen encoder the forward runs
        under torch.no_grad(), so there are no activations to store and no backward to
        recompute. The flag is refused there rather than silently accepted, so the frozen
        path cannot drift.
        """
        enable = bool(enable)
        if enable and self.n_blocks == 0:
            # not an error -- callers sweep the flag across rungs -- but it must not pretend
            # to have done something, and it must never perturb the frozen path.
            print("[ZoobotEncoder] --grad-checkpointing ignored at n_blocks=0 (frozen "
                  "encoder runs under no_grad; there is no backward to checkpoint).")
            enable = False
        self.model.set_grad_checkpointing(enable)
        self.grad_checkpointing = enable
        return self

    # -- mode application ---------------------------------------------------

    def _apply_frozen(self):
        """Freeze everything, then re-enable exactly the top ``n_blocks`` blocks.

        ``n_blocks == 0`` leaves this identical to the original frozen encoder: 0 trainable
        params, ``_frozen=True``, and the ``torch.no_grad()`` forward fast path. That
        identity is a hard requirement -- the frozen results already in ``runs/`` must stay
        reproducible byte-for-byte.
        """
        self.model.eval()          # no BN / no active dropout / drop_path=Identity => no-op,
                                   # kept for determinism (see module docstring)
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.trainable_block_names: list[str] = []
        for name, block in self.tuneable_blocks_in_lr_order[:self.n_blocks]:
            for p in block.parameters():
                p.requires_grad_(True)
            self.trainable_block_names.append(name)

        # _frozen gates the no_grad() forward fast path. It must stay True at n_blocks=0.
        self._frozen = (self.n_blocks == 0)

        if self.n_blocks > 0:
            n_train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            expected = TRAINABLE_PARAMS_BY_N_BLOCKS.get(self.n_blocks)
            print(f"[ZoobotEncoder] PARTIAL UNFREEZE n_blocks={self.n_blocks} -> trainable "
                  f"{self.trainable_block_names} ({n_train:,} params, "
                  f"{100.0 * n_train / sum(p.numel() for p in self.model.parameters()):.1f}% "
                  f"of the encoder)")
            if expected is not None and n_train != expected:
                raise RuntimeError(
                    f"trainable encoder param count is {n_train:,} but the MEASURED anchor "
                    f"for n_blocks={self.n_blocks} is {expected:,}. The block convention or "
                    f"the timm build has drifted -- refusing to train a model whose unfreeze "
                    f"scope does not match the verified table.")
            print("[ZoobotEncoder] NOTE: head.norm (final LayerNorm2d, 1,280 params) is "
                  "OUTSIDE Zoobot's tuneable_blocks and stays frozen at every n_blocks -- "
                  "this replicates the benchmark protocol, it is not an oversight.")

    # -- config reporting ---------------------------------------------------

    def _print_cfg_once(self):
        print("[ZoobotEncoder] loaded", MODEL_SLUG)
        print(f"[ZoobotEncoder] pretrained_cfg.input_size = {self.cfg_input_size}")
        print(f"[ZoobotEncoder] pretrained_cfg.mean       = {self.cfg_mean}")
        print(f"[ZoobotEncoder] pretrained_cfg.std        = {self.cfg_std}")
        is_imagenet = (self.cfg_mean == [0.485, 0.456, 0.406]
                       and self.cfg_std == [0.229, 0.224, 0.225])
        if is_imagenet:
            print("[ZoobotEncoder] NOTE: mean/std above are inherited ImageNet timm metadata, "
                  "NOT Zoobot's training stats -- NOT applied. Input must already be "
                  "stretched to [0,1] by LensTransform.")
        else:
            print("[ZoobotEncoder] NOTE: pretrained_cfg transform is NOT applied; input must "
                  "already be stretched to [0,1] by LensTransform.")

    # -- forward ------------------------------------------------------------

    def _check_input(self, x: torch.Tensor):
        if not torch.is_tensor(x):
            raise TypeError(f"expected a torch.Tensor, got {type(x)}")
        if x.dim() != 4:
            raise ValueError(f"expected rank-4 input (B,1,H,W), got shape {tuple(x.shape)}")
        if x.shape[1] != IN_CHANS:
            raise ValueError(
                f"expected single-channel input (C=={IN_CHANS}), got C={x.shape[1]} "
                f"(shape {tuple(x.shape)}). The encoder must NOT be fed a 3-channel image.")

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        """Explicit pool: forward_features -> head global-pool + norm (classifier skipped).
        Asserts the (B,640) contract and fails loudly on anything else."""
        feats = self.model.forward_features(x)                 # (B,640,7,7)
        pooled = self.model.forward_head(feats, pre_logits=True)  # (B,640)
        if pooled.dim() != 2 or pooled.shape[0] != x.shape[0] or pooled.shape[1] != EMBED_DIM:
            raise RuntimeError(
                f"encoder pooled output is {tuple(pooled.shape)}, expected "
                f"({x.shape[0]}, {EMBED_DIM}). The (B,640) contract is broken -- refusing "
                f"to return possibly-wrong features.")
        return pooled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input(x)
        if getattr(self, "_frozen", False):
            with torch.no_grad():
                return self._features(x)
        return self._features(x)   # reserved for future partially-trainable modes

    # convenience alias
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    # -- one-time construction self-check ----------------------------------

    def _self_check(self, device):
        dev = next(self.model.parameters()).device
        dummy = torch.zeros(2, IN_CHANS, INPUT_SIZE, INPUT_SIZE, device=dev)
        # no_grad here regardless of n_blocks: the construction-time contract check must not
        # build (and hold) an autograd graph for a throwaway batch.
        with torch.no_grad():
            out = self.forward(dummy)
        assert tuple(out.shape) == (2, EMBED_DIM), \
            f"self-check failed: expected (2,{EMBED_DIM}), got {tuple(out.shape)}"

    # -- introspection ------------------------------------------------------

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def effective_mode(self) -> str:
        """A HONEST label for checkpoints/meta: 'frozen' only when nothing is trainable.

        ``encoder_mode`` stays the plumbed switch name ('frozen'); ``n_blocks`` is the
        orthogonal unfreeze knob, so at n_blocks>0 the string 'frozen' would be a lie in
        every checkpoint cfg and cache meta. Runners record THIS instead.
        """
        return FROZEN if self.n_blocks == 0 else f"unfrozen_n_blocks_{self.n_blocks}"

    def extra_repr(self) -> str:
        return (f"mode={self.encoder_mode}, n_blocks={self.n_blocks}, "
                f"embed_dim={self.embed_dim}, in_chans={IN_CHANS}, "
                f"frozen={getattr(self, '_frozen', False)}, "
                f"grad_checkpointing={getattr(self, 'grad_checkpointing', False)}")


def build_encoder(encoder_mode: str = FROZEN, n_blocks: int = 0, device=None,
                  verbose: bool = True, grad_checkpointing: bool = False) -> ZoobotEncoder:
    """Factory mirroring the intended call site (keeps construction in one place)."""
    return ZoobotEncoder(encoder_mode=encoder_mode, n_blocks=n_blocks, device=device,
                         verbose=verbose, grad_checkpointing=grad_checkpointing)


if __name__ == "__main__":
    enc = build_encoder()
    print(enc)
    print("trainable params:", enc.num_trainable_params())
