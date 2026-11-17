"""
model.py — frozen Zoobot encoder + trainable adapter + linear head.

Architecture
------------
    image (B,1,64,64)
        -> Zoobot ConvNeXT-Nano  [FROZEN]   -> (B, 640)   features
        -> Adapter (640->512->256)           -> (B, 256)   DA-alignment space
        -> Head    (256->2)                  -> (B, 2)     logits

Design notes (see chat for the reasoning):
  * 2 logits + softmax, NOT 1 logit + BCE. AdaMatch's confidence threshold and
    distribution alignment are defined over a categorical distribution; 2 logits
    keep that machinery native. score(positive) = softmax(logits)[:, 1].
  * Adapter uses LayerNorm + GELU, NO BatchNorm. ConvNeXT itself is LN-based, and
    a BN-free path means AdaMatch's "random logit interpolation" (which exists
    purely to fix BN coupling source+target statistics) is unnecessary -> we drop
    that step entirely. One less moving part.
  * Encoder is held in eval() whenever frozen, even under model.train(), so its
    stochastic-depth / dropout stay off and frozen features are deterministic.
    (forgetting this is a classic frozen-backbone bug.)
  * The 256-d adapter output is the layer your WDGRL/DANN/CDAN critics will attach
    to later. AdaMatch doesn't need it, but the bake-off does, so we fix it now.
"""
from __future__ import annotations

import os
import random
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

ZOOBOT_HF = "hf_hub:mwalmsley/zoobot-encoder-greyscale-convnext_nano"


def _freeze(enc: nn.Module) -> nn.Module:
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc


def build_zoobot_encoder(pretrained: bool = True, freeze: bool = True) -> nn.Module:
    """ConvNeXT-Nano greyscale Zoobot encoder -> (B, 640) pooled features.

    num_classes=0 makes forward() return the global-avg-pooled vector directly
    (verified: (B,640) at 64x64). in_chans=1 for single-band VIS.
    NOTE: pretrained=True hits HuggingFace. On Colab prefer cached_zoobot_encoder.
    """
    enc = timm.create_model(ZOOBOT_HF, pretrained=pretrained, num_classes=0, in_chans=1)
    return _freeze(enc) if freeze else enc


def cached_zoobot_encoder(cache_pt: str, freeze: bool = True) -> nn.Module:
    """Load the Zoobot encoder, caching weights to a single .pt on Drive.

    First run : pull from HuggingFace (pretrained=True), copy the state_dict into
                a plain convnext_nano (identical architecture), save to cache_pt.
    Later runs: build plain convnext_nano (NO hf_hub prefix -> no HF call at all)
                and load cache_pt. Fully offline after the first pull, so a Colab
                reconnect never re-downloads.
    """
    enc = timm.create_model("convnext_nano", pretrained=False, num_classes=0, in_chans=1)
    if os.path.exists(cache_pt):
        enc.load_state_dict(torch.load(cache_pt, map_location="cpu"))
        print(f"[encoder] loaded cached Zoobot weights <- {cache_pt}")
    else:
        print("[encoder] no cache found; pulling Zoobot from HuggingFace (one-time) ...")
        src = timm.create_model(ZOOBOT_HF, pretrained=True, num_classes=0, in_chans=1)
        # both are convnext_nano(in_chans=1, num_classes=0) -> keys match exactly
        enc.load_state_dict(src.state_dict(), strict=True)
        os.makedirs(os.path.dirname(cache_pt) or ".", exist_ok=True)
        torch.save(enc.state_dict(), cache_pt)
        print(f"[encoder] cached Zoobot weights -> {cache_pt}")
    return _freeze(enc) if freeze else enc


class Adapter(nn.Module):
    """640 -> 512 -> 256 MLP. LayerNorm + GELU, no BatchNorm. This is the
    feature space DA operates in."""

    def __init__(self, in_dim: int = 640, hidden: int = 512, out_dim: int = 256,
                 p_drop: float = 0.1):
        super().__init__()
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LensClassifier(nn.Module):
    """Frozen encoder + adapter + 2-logit head.

    .features(x) -> (B,256)   the DA-alignment representation
    .forward(x)  -> (B,2)     classification logits
    """

    def __init__(self, pretrained: bool = True, freeze_encoder: bool = True,
                 adapter_hidden: int = 512, feat_dim: int = 256, p_drop: float = 0.1,
                 encoder: nn.Module | None = None):
        super().__init__()
        # inject a prebuilt (e.g. cached) encoder, else build from HF
        self.encoder = encoder if encoder is not None else \
            build_zoobot_encoder(pretrained=pretrained, freeze=freeze_encoder)
        if freeze_encoder:                      # ensure an injected encoder is frozen too
            _freeze(self.encoder)
        self.frozen = freeze_encoder
        self.adapter = Adapter(640, adapter_hidden, feat_dim, p_drop)
        self.head = nn.Linear(feat_dim, 2)
        self.feat_dim = feat_dim

    def train(self, mode: bool = True):
        """Keep a frozen encoder in eval() so its stochastic depth / norm stats
        never go into train mode when the rest of the model does."""
        super().train(mode)
        if self.frozen:
            self.encoder.eval()
        return self

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.frozen:
            with torch.no_grad():          # save memory; no grad needed for frozen trunk
                return self.encoder(x)
        return self.encoder(x)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.adapter(self._encode(x))      # (B, feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))         # (B, 2)

    @torch.no_grad()
    def predict_score(self, x: torch.Tensor) -> torch.Tensor:
        """P(lens) = softmax over the 2 logits, positive class. Use this for
        TPR@FPR scoring."""
        return F.softmax(self.forward(x), dim=1)[:, 1]

    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Weak augmentation for AdaMatch's weak view.
# strong view = SSLAugment from lens_dataset.py (crop + D4 + noise + blur).
# weak view = D4/flips only: label-preserving geometry, no intensity change.
# IMPORTANT: neither view rescales brightness/contrast -> we do NOT reintroduce
# the brightness shortcut the v_mid generator was tuned to kill.
# ---------------------------------------------------------------------------
import torchvision.transforms.functional as TF


class WeakAugment:
    """D4 (rot90 + flips) only. Single-band -> no color/brightness jitter."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.rot90(x, random.randint(0, 3), dims=[1, 2])
        if random.random() < 0.5:
            x = TF.hflip(x)
        if random.random() < 0.5:
            x = TF.vflip(x)
        return x


if __name__ == "__main__":
    # architecture-only smoke test (pretrained=False so it runs without HF egress)
    m = LensClassifier(pretrained=False)
    m.train()
    assert not m.encoder.training, "frozen encoder must stay in eval under .train()"
    x = torch.randn(8, 1, 64, 64)
    print("features:", tuple(m.features(x).shape))      # (8, 256)
    print("logits  :", tuple(m(x).shape))               # (8, 2)
    print("score   :", tuple(m.predict_score(x).shape)) # (8,)
    n_train = sum(p.numel() for p in m.trainable_parameters())
    n_frozen = sum(p.numel() for p in m.parameters() if not p.requires_grad)
    print(f"trainable={n_train:,}  frozen={n_frozen:,}")
