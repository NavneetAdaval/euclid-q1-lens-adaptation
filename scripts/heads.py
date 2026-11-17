"""
heads.py
========
Classifier HEAD (a switch) + the frozen FEATURE STANDARDIZER for the label-free
sim-to-real strong-lens project (Euclid Q1). Everything here sits on TOP of the
frozen 640-dim Zoobot features; the encoder itself is never touched.

Tags: RB=research-backed, EC=engineering-choice.

HEAD SWITCH (``head_type``, default "linear")
---------------------------------------------
  * "linear" (DEFAULT, the headline baseline)
        LayerNorm(640) -> Linear(640 -> 1)
    This is the standard frozen-probe protocol -- the honest baseline for the
    paper's claim "recover X% of the supervised benchmark from FROZEN features."

  * "mlp" (diagnostic / ablation ONLY, NOT the default)
        LayerNorm(640) -> Linear(640 -> 128) -> GELU -> Dropout(0.2)
                       -> LayerNorm(128) -> Linear(128 -> 1)
    Hidden=128 is deliberate: the literature reports MLP gains over a linear probe
    on GOOD frozen features are small and inconsistent, and do NOT scale with width,
    while overfit risk on our ~391-per-fold target does. LayerNorm (not BatchNorm)
    everywhere -- our adapter-norm decision: it matches the frozen ConvNeXt, is
    batch-composition-invariant, and avoids the DSBN domain-mixing trap.

WHY BOTH ARE SELECTABLE (the diagnostic rationale)
--------------------------------------------------
linear-vs-mlp is run as a DIAGNOSTIC, not a horse race:
  - if "mlp" CLEARLY beats "linear", the lens signal is nonlinearly encoded in the
    frozen features and we report that finding;
  - if they TIE, the linear headline is justified by evidence (not by assumption).
Both build from the same call with no rewrite, so the DA bake-off can carry the
linear headline + the mlp ablation side by side.

FEATURE STANDARDIZER (RB/EC)
----------------------------
Fit mean/std of the 640-d frozen features on SOURCE_TRAIN once (a no-grad pass),
FREEZE those statistics, and apply ``(x - mean) / std`` to every role before the
head. Stored as buffers so it round-trips inside the checkpoint. It owns NO trainable
parameters today (the stats are frozen), but the training loop still includes any
future standardizer params in the optimizer set -- never the encoder.

The head exposes ``forward_with_features`` returning ``(logits, adapter_features)``:
the DA loss plugs onto ``adapter_features`` (the post-trunk representation), so it
operates on HEAD/adapter features and never backprops into the frozen encoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn

EMBED_DIM = 640
LINEAR, MLP = "linear", "mlp"
HEAD_TYPES = (LINEAR, MLP)


# ---------------------------------------------------------------------------
# classifier head
# ---------------------------------------------------------------------------

class ClassifierHead(nn.Module):
    """LayerNorm-based probe on the frozen 640-d features.

    Split into ``trunk`` (everything up to the final linear -- the "adapter") and
    ``classifier`` (the final ``Linear(feat_dim -> 1)``), so the adapter representation
    is available to the DA loss via ``forward_with_features``.
    """

    def __init__(self, head_type: str = LINEAR, in_dim: int = EMBED_DIM,
                 hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        if head_type not in HEAD_TYPES:
            raise ValueError(f"head_type must be one of {HEAD_TYPES}, got {head_type!r}")
        self.head_type = head_type
        self.in_dim = in_dim

        if head_type == LINEAR:
            self.trunk = nn.Sequential(nn.LayerNorm(in_dim))
            feat_dim = in_dim
        else:  # MLP
            self.trunk = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden),
            )
            feat_dim = hidden

        self.feat_dim = feat_dim
        self.classifier = nn.Linear(feat_dim, 1)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Post-trunk adapter representation (what the DA loss consumes)."""
        return self.trunk(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.trunk(x))          # (B, 1) logits

    def forward_with_features(self, x: torch.Tensor):
        f = self.trunk(x)
        return self.classifier(f), f                   # (logits, adapter_features)

    def extra_repr(self) -> str:
        return f"head_type={self.head_type}, in_dim={self.in_dim}, feat_dim={self.feat_dim}"


def build_head(head_type: str = LINEAR, in_dim: int = EMBED_DIM,
               hidden: int = 128, dropout: float = 0.2) -> ClassifierHead:
    """Factory: keeps head construction in one place for the bake-off."""
    return ClassifierHead(head_type=head_type, in_dim=in_dim, hidden=hidden, dropout=dropout)


# ---------------------------------------------------------------------------
# feature standardizer (frozen stats, saved with the checkpoint)
# ---------------------------------------------------------------------------

class FeatureStandardizer(nn.Module):
    """Freeze mean/std of the source-train frozen features; apply ``(x-mean)/std``.

    Stats live in buffers (mean, std, fitted) so they serialize with the checkpoint
    and move with ``.to(device)``. ``fit`` and ``set_stats`` populate them under
    no_grad; ``forward`` refuses to run before the stats are fitted (a guard against
    silently standardizing with the mean=0/std=1 identity).
    """

    def __init__(self, dim: int = EMBED_DIM, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))
        self.register_buffer("fitted", torch.zeros(1))     # 0/1 flag, serialized

    @property
    def is_fitted(self) -> bool:
        return float(self.fitted) >= 1.0

    @torch.no_grad()
    def fit(self, features: torch.Tensor) -> "FeatureStandardizer":
        """Fit from a stacked ``(N, dim)`` feature tensor (small enough to hold)."""
        if features.dim() != 2 or features.shape[1] != self.dim:
            raise ValueError(f"expected (N,{self.dim}) features, got {tuple(features.shape)}")
        self.set_stats(features.mean(0), features.std(0))
        return self

    @torch.no_grad()
    def set_stats(self, mean: torch.Tensor, std: torch.Tensor) -> "FeatureStandardizer":
        """Set stats from precomputed mean/std (e.g. a streaming pass over batches)."""
        self.mean.copy_(mean.to(self.mean))
        self.std.copy_(std.to(self.std).clamp_min(self.eps))
        self.fitted.fill_(1.0)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.is_fitted:
            raise RuntimeError(
                "FeatureStandardizer.forward() called before fit()/set_stats(); "
                "fit it on source_train once, under no_grad, then freeze it.")
        return (x - self.mean) / self.std

    def extra_repr(self) -> str:
        return f"dim={self.dim}, fitted={self.is_fitted}"
