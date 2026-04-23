"""
Training script for DualStreamCNN.

Quick-start (CPU, single-batch overfit sanity check):
    python train.py --overfit

Full 10-class training run:
    python train.py --epochs 20 --batch_size 64 --lr 1e-3

OSCAR cluster (full 100-class):
    python train.py --epochs 50 --batch_size 128 --num_classes 100 \\
                    --data_dir /path/to/data --save_dir ./outputs
"""

import argparse
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from models import DualStreamCNN
from preprocessing import create_dual_stream_loaders


# ---------------------------------------------------------------------------
# Argument parsing — all hyperparameters live here, nothing hardcoded below
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(description="Train DualStreamCNN on CIFAR-100")

    # Data
    p.add_argument("--data_dir",    type=str,   default="./data",
                   help="Root directory for CIFAR-100 download/cache")
    p.add_argument("--num_classes", type=int,   default=10,
                   help="10 for development subset, 100 for full CIFAR-100")
    p.add_argument("--img_size",    type=int,   default=128,
                   help="Spatial size fed to both backbones")

    # Model
    p.add_argument("--alpha",           type=float, default=0.5,
                   help="Initial M/P stream blend (0=P-only, 1=M-only)")
    p.add_argument("--learnable_alpha", action="store_true",
                   help="Make alpha a learnable parameter")

    # Optimisation
    p.add_argument("--epochs",     type=int,   default=20)
    p.add_argument("--batch_size", type=int,   default=64)
    p.add_argument("--lr",         type=float, default=1e-3)

    # Overfit mode: freeze one batch of 16, train for many epochs → loss must → 0
    p.add_argument("--overfit",        action="store_true",
                   help="Single-batch overfit sanity check")
    p.add_argument("--overfit_epochs", type=int, default=100,
                   help="Epochs to run in overfit mode (default 100)")

    # I/O
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--save_dir",    type=str, default="./outputs",
                   help="Directory for loss curve plot and checkpoints")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for m_batch, p_batch, labels in loader:
        m_batch = m_batch.to(device)
        p_batch = p_batch.to(device)
        labels  = labels.to(device)

        optimizer.zero_grad()
        logits = model(m_batch, p_batch)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct    += (logits.argmax(dim=1) == labels).sum().item()
        total      += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for m_batch, p_batch, labels in loader:
        m_batch = m_batch.to(device)
        p_batch = p_batch.to(device)
        labels  = labels.to(device)

        logits = model(m_batch, p_batch)
        loss   = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        correct    += (logits.argmax(dim=1) == labels).sum().item()
        total      += labels.size(0)

    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Overfit sanity check — trains exclusively on one fixed batch
# ---------------------------------------------------------------------------

def run_overfit(model, loader, criterion, optimizer, device, epochs, save_dir):
    """
    Isolate one batch and overfit on it for `epochs` steps.
    If the model is correct, training loss must approach zero.
    A flat or non-decreasing loss indicates a bug in the forward/backward pass.
    """
    # Grab exactly one batch and keep it on device for the whole run
    m_batch, p_batch, labels = next(iter(loader))
    m_batch = m_batch[:16].to(device)
    p_batch = p_batch[:16].to(device)
    labels  = labels[:16].to(device)

    losses = []
    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        logits = model(m_batch, p_batch)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if epoch % 10 == 0 or epoch == 1:
            print(f"  [overfit] epoch {epoch:>4}/{epochs}  loss={loss.item():.6f}")

    final = losses[-1]
    passed = final < 0.05
    print(f"\n  Final loss : {final:.6f}  →  {'PASSED ✓' if passed else 'FAILED ✗ (check backprop)'}")

    _save_loss_curve(losses, save_dir, filename="overfit_loss.png",
                     title="Single-Batch Overfit Loss (must → 0)")
    return losses


# ---------------------------------------------------------------------------
# Full training loop
# ---------------------------------------------------------------------------

def run_training(model, train_loader, val_loader, criterion, optimizer,
                 device, epochs, save_dir):
    train_losses, val_losses = [], []

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)

        train_losses.append(tr_loss)
        val_losses.append(va_loss)

        print(f"Epoch {epoch:>3}/{epochs}  "
              f"train loss={tr_loss:.4f} acc={tr_acc:.3f}  "
              f"val loss={va_loss:.4f} acc={va_acc:.3f}  "
              f"alpha={model.alpha.item():.3f}")

    _save_loss_curve(
        train_losses, save_dir,
        filename="train_loss.png",
        title="Training Loss",
        val_losses=val_losses,
    )
    _save_checkpoint(model, optimizer, epochs, save_dir)
    return train_losses, val_losses


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _save_loss_curve(train_losses, save_dir, filename, title, val_losses=None):
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_losses, label="train")
    if val_losses:
        ax.plot(val_losses, label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = os.path.join(save_dir, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Loss curve saved → {path}")


def _save_checkpoint(model, optimizer, epoch, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"checkpoint_epoch{epoch}.pt")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "alpha": model.alpha.item(),
    }, path)
    print(f"  Checkpoint saved → {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = get_args()

    # Device — CUDA first (OSCAR), then MPS (local Apple Silicon), then CPU
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Device      : {device}")
    print(f"Mode        : {'overfit sanity check' if args.overfit else 'full training'}")
    print(f"Num classes : {args.num_classes}")
    print(f"Alpha       : {args.alpha}  (learnable={args.learnable_alpha})\n")

    # Data
    use_subset = (args.num_classes == 10)
    train_loader, val_loader, num_classes = create_dual_stream_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        use_10_class_subset=use_subset,
    )

    # Model — move all parameters and buffers to device in one call
    model = DualStreamCNN(
        num_classes=num_classes,
        alpha=args.alpha,
        learnable_alpha=args.learnable_alpha,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.overfit:
        run_overfit(
            model, train_loader, criterion, optimizer,
            device, args.overfit_epochs, args.save_dir,
        )
    else:
        run_training(
            model, train_loader, val_loader, criterion, optimizer,
            device, args.epochs, args.save_dir,
        )


if __name__ == "__main__":
    main()
