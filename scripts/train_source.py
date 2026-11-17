"""train_source.py - supervised training on the SOURCE pool (sim pos : neg = 1:1).

Produces the sim-only baseline checkpoint: NO domain adaptation, just the
adapter + head trained on simulated lenses vs negatives on top of the frozen
Zoobot encoder. This checkpoint is
    (a) the input to gate_eval.py, and
    (b) your sim-only reference itself (the X1-side baseline).

Model / method selection is on SIM-VAL only (TPR@FPR=1e-3). The 326 real lenses
are NOT touched here -- they are sealed for the single final measurement.

Run (Colab / Linux):
    python train_source.py --data-dir /content/lens_paper/data \
        --epochs 40 --out ckpts/source_only --num-workers 2 --amp --cosine

Run (Windows / sandbox):
    python train_source.py --data-dir C:\\Users\\krris\\ML\\lens_paper\\data \
        --epochs 40 --out ckpts/source_only --num-workers 0

LensDataset interface (verified against scripts/lens_dataset.py):
    LensDataset(role, data_dir, splits_path=None, fold=None, augment=None, ...)
      role='source_train' -> (image[1,H,W] float32, int label)  augment=SupervisedAugment by default
      role='val'          -> (image[1,H,W] float32, int label)  augment=None by default
    splits_path defaults to data_dir/splits/splits.json.
"""
from __future__ import annotations
import argparse
import json
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import build_lens_classifier, collect_logits, _split_batch
from metrics import tpr_at_fpr, auroc

try:
    from lens_dataset import LensDataset  # type: ignore
except Exception:
    LensDataset = None


def get_amp():
    """Return (autocast_ctx_factory, GradScaler) compatible across torch versions."""
    try:
        from torch.amp import autocast, GradScaler

        def ac(enabled):
            return autocast(device_type="cuda", enabled=enabled)

        return ac, GradScaler
    except Exception:
        from torch.cuda.amp import autocast, GradScaler  # older torch

        def ac(enabled):
            return autocast(enabled=enabled)

        return ac, GradScaler


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True,
                   help="data root (contains splits/, sim_lenses/, q1_unlabeled/, ...)")
    p.add_argument("--splits-path", default=None,
                   help="optional explicit splits.json; defaults to <data-dir>/splits/splits.json")
    p.add_argument("--out", default="ckpts/source_only", help="checkpoint dir prefix")
    p.add_argument("--arch", default="zoobot_greyscale_convnext_nano")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--repr-dim", type=int, default=256)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--cosine", action="store_true", help="cosine-decay the LR")
    p.add_argument("--amp", action="store_true", help="mixed precision (GPU only)")
    p.add_argument("--fpr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", default="", help="path to a checkpoint to resume from")
    return p.parse_args()


def main():
    args = parse()
    if LensDataset is None:
        raise RuntimeError(
            "Could not import LensDataset. Add scripts/ to sys.path or run from "
            "the scripts directory so `from lens_dataset import LensDataset` works."
        )
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.amp and device.type == "cuda"
    autocast_ctx, GradScaler = get_amp()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    model = build_lens_classifier(
        arch=args.arch, repr_dim=args.repr_dim, hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    # --- data ---------------------------------------------------------------
    # Per-role augment defaults in LensDataset: source_train -> SupervisedAugment
    # (ON, the regularizer we want), val -> None (OFF). Do not override.
    train_ds = LensDataset("source_train", args.data_dir, splits_path=args.splits_path)
    val_ds = LensDataset("val", args.data_dir, splits_path=args.splits_path)

    pin = device.type == "cuda"
    # drop_last=True avoids a final 1-sample batch breaking BatchNorm in the adapter.
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers, pin_memory=pin,
    )

    # --- optim --------------------------------------------------------------
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr,
                            weight_decay=args.wd)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
             if args.cosine else None)
    scaler = GradScaler(enabled=use_amp)
    # Source pool is exactly 1:1, so plain BCE (no pos_weight).
    criterion = nn.BCEWithLogitsLoss()

    start_epoch, best_tpr = 0, -1.0
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["state_dict"])
        if "optimizer" in ck:
            opt.load_state_dict(ck["optimizer"])
        start_epoch = ck.get("epoch", -1) + 1
        best_tpr = ck.get("best_tpr", -1.0)
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    # --- loop ---------------------------------------------------------------
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running, nb = 0.0, 0
        for batch in train_loader:
            x, y = _split_batch(batch)
            x = x.to(device, non_blocking=True)
            y = y.float().to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast_ctx(use_amp):
                logit = model(x)
                loss = criterion(logit, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item()
            nb += 1
        if sched is not None:
            sched.step()

        logits, labels, _ = collect_logits(
            model, val_ds, device, batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        m = tpr_at_fpr(logits, labels, args.fpr)
        a = auroc(logits, labels)
        print(
            f"epoch {epoch:3d} | train_loss {running / max(nb,1):.4f} | "
            f"val AUROC {a:.4f} | val TPR@FPR={args.fpr:g} {m['tpr']:.4f} "
            f"(achieved FPR {m['achieved_fpr']:.2e}, "
            f"resolution {m['fpr_resolution']:.2e}, n_neg {m['n_neg']})"
        )

        ckpt = dict(
            state_dict=model.state_dict(),
            optimizer=opt.state_dict(),
            config=model.config,
            epoch=epoch,
            val_tpr_at_fpr=m,
            val_auroc=a,
            args=vars(args),
        )
        torch.save(ckpt, args.out + "_last.pt")
        if m["tpr"] > best_tpr:
            best_tpr = m["tpr"]
            ckpt["best_tpr"] = best_tpr
            torch.save(ckpt, args.out + "_best.pt")
            with open(args.out + "_best.json", "w") as f:
                json.dump({"epoch": epoch, "val_auroc": a, "val_tpr_at_fpr": m}, f, indent=2)
            print(f"   -> new best (TPR {best_tpr:.4f}) saved to {args.out}_best.pt")

    print(f"done. best sim-val TPR@FPR={args.fpr:g} = {best_tpr:.4f}")


if __name__ == "__main__":
    main()
