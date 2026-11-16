"""Train a STANDARD CIFAR-10 ResNet-18 (random crop + flip only) to use as the source model.

Why: RobustBench has no plain ResNet-18 for CIFAR-10 corruptions (its ResNet-18 entry uses PRIME augmentation,
i.e. it is already corruption-robust). A standard-trained model is the usual TTA setting.
Usage:  python scripts/train_source.py [--epochs 30] [--seed 0] [--out checkpoints/resnet18_cifar10_seed0.pt]
Needs torchvision (only to fetch CIFAR-10). All data lives on the GPU; augmentation is done on the GPU too
(avoids multi-process data loaders, which are slow/fragile on Windows). Training is NOT bit-for-bit
reproducible (mixed precision, cuDNN); the checkpoint file itself (its sha256 goes into every run summary) is the artefact.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from src.models.resnet_cifar import ResNet18CIFAR


def random_crop_flip(x: torch.Tensor, pad: int = 4) -> torch.Tensor:
    """Per-sample random crop (zero padding `pad`) and horizontal flip. x: float (N, C, H, W)."""
    n, c, h, w = x.shape
    flip = torch.rand(n, device=x.device) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(3), x)
    xp = F.pad(x, (pad, pad, pad, pad))
    i = torch.randint(0, 2 * pad + 1, (n,), device=x.device)
    j = torch.randint(0, 2 * pad + 1, (n,), device=x.device)
    rows = (i[:, None] + torch.arange(h, device=x.device)[None])[:, None, :, None]
    cols = (j[:, None] + torch.arange(w, device=x.device)[None])[:, None, None, :]
    return xp[torch.arange(n, device=x.device)[:, None, None, None], torch.arange(c, device=x.device)[None, :, None, None],
              rows, cols]


@torch.no_grad()
def evaluate(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, bs: int = 1000) -> float:
    model.eval()
    correct = 0
    for i in range(0, len(x), bs):
        correct += int((model(x[i:i + bs].float() / 255.0).argmax(1) == y[i:i + bs]).sum())
    return correct / len(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-root", default="data_cache")
    ap.add_argument("--out", default="checkpoints/resnet18_cifar10_seed0.pt")
    args = ap.parse_args()

    from torchvision.datasets import CIFAR10  # lazy import: only this script needs torchvision

    torch.manual_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr, te = CIFAR10(args.data_root, train=True, download=True), CIFAR10(args.data_root, train=False, download=True)
    xtr = torch.from_numpy(tr.data).permute(0, 3, 1, 2).contiguous().to(dev)      # uint8 (50000, 3, 32, 32)
    ytr = torch.tensor(tr.targets, device=dev)
    xte = torch.from_numpy(te.data).permute(0, 3, 1, 2).contiguous().to(dev)
    yte = torch.tensor(te.targets, device=dev)

    model = ResNet18CIFAR().to(dev)
    spe = len(xtr) // args.batch_size
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * spe, pct_start=0.25)
    use_amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler(dev.type, enabled=use_amp)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    log_path = out.with_suffix(".train_log.csv")
    t0, acc = time.time(), float("nan")
    with open(log_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "clean_test_acc", "seconds"])
        for epoch in range(1, args.epochs + 1):
            model.train()
            perm = torch.randperm(len(xtr), device=dev)
            total = 0.0
            for b in range(spe):
                idx = perm[b * args.batch_size:(b + 1) * args.batch_size]
                x = random_crop_flip(xtr[idx].float() / 255.0)
                with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
                    loss = F.cross_entropy(model(x), ytr[idx])
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
                total += float(loss)
            acc = evaluate(model, xte, yte)
            w.writerow([epoch, total / spe, acc, time.time() - t0])
            f.flush()
            print(f"epoch {epoch}/{args.epochs} loss {total / spe:.4f} clean test acc {acc:.4f} ({time.time() - t0:.0f}s)")
    torch.save({"model": model.state_dict(), "epoch": args.epochs, "test_acc": acc, "seed": args.seed,
                "args": vars(args)}, out)
    print(f"saved {out} (clean test acc {acc:.4f}); log: {log_path}")


if __name__ == "__main__":
    main()
