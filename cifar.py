# MIT License
#
# Copyright (c) 2022 Chengfeng Zhou
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Train a ResNet-18 classifier on CIFAR-10 / CIFAR-100 with WIN normalization.

Supports evaluation on the corruption benchmarks CIFAR-10-C / CIFAR-100-C and
their extended variants CIFAR-10-C-Bar / CIFAR-100-C-Bar.

Example
-------
Train with WIN on CIFAR-10::

    python cifar.py --dataset cifar10 --data-path ./data --norm WIN

Train with WIN-WIN (self-distillation) on CIFAR-100::

    python cifar.py --dataset cifar100 --data-path ./data --norm WIN-WIN

Evaluate a saved checkpoint::

    python cifar.py --dataset cifar10 --data-path ./data \\
        --norm WIN --resume ./run/checkpoint.pth.tar --evaluate
"""

from __future__ import annotations

import argparse
import datetime
import os
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.models as models
from torch.amp import GradScaler, autocast
from torchvision import datasets, transforms

from WIN import WindowNorm2d

# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Train a CIFAR classifier with configurable normalization.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--dataset", type=str, default="cifar10",
    choices=["cifar10", "cifar100"],
    help="Dataset to use.",
)
parser.add_argument(
    "--data-path", type=str, required=True,
    help="Root directory containing CIFAR and CIFAR-C subdirectories.",
)
parser.add_argument(
    "--norm", type=str, default="BN",
    choices=["BN", "IN", "GN", "WIN", "WIN-WIN", "Identity"],
    help="Normalization layer to use throughout the network.",
)

# Optimisation
parser.add_argument("--epochs", "-e", type=int, default=180,
                    help="Total number of training epochs.")
parser.add_argument("--learning-rate", "-lr", type=float, default=0.3,
                    help="Peak learning rate (cosine schedule).")
parser.add_argument("--batch-size", "-b", type=int, default=64,
                    help="Training batch size.")
parser.add_argument("--eval-batch-size", type=int, default=64,
                    help="Evaluation batch size.")
parser.add_argument("--momentum", type=float, default=0.9,
                    help="SGD momentum.")
parser.add_argument("--decay", "-wd", type=float, default=5e-4,
                    help="Weight decay (L2 penalty).")

# Checkpointing
parser.add_argument("--save", "-s", type=str, default="./snapshots",
                    help="Directory to save checkpoints and logs.")
parser.add_argument("--resume", "-r", type=str, default="",
                    help="Path to a checkpoint to resume from.")
parser.add_argument("--evaluate", action="store_true",
                    help="Run evaluation only (no training).")
parser.add_argument("--print-freq", type=int, default=50,
                    help="Log training loss every N batches.")

# DataLoader
parser.add_argument("--num-workers", type=int, default=4,
                    help="Number of DataLoader worker processes.")

args = parser.parse_args()
print(args)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
    "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog",
    "brightness", "contrast", "elastic_transform", "pixelate",
    "jpeg_compression",
]

CBAR_CORRUPTIONS = [
    "blue_noise_sample", "brownish_noise", "checkerboard_cutout",
    "inverse_sparkles", "pinch_and_twirl", "ripple", "circular_motion_blur",
    "lines", "sparkles", "transverse_chromatic_abberation",
]

NUM_CLASSES = 100 if args.dataset == "cifar100" else 10

# ---------------------------------------------------------------------------
# Learning-rate schedule
# ---------------------------------------------------------------------------

def get_lr(step: int, total_steps: int, lr_max: float, lr_min: float) -> float:
    """Cosine annealing learning-rate schedule."""
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def train(net, train_loader, optimizer, scheduler, scaler):
    """Train for one epoch.

    Returns:
        tuple[float, float]: Exponential-moving-average loss and top-1
        accuracy over the epoch.
    """
    net.train()
    loss_ema = 0.0
    total_correct = 0
    total = 0

    for i, (images, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        images = images.cuda()
        targets = targets.cuda()

        with autocast(device_type="cuda"):
            if args.norm == "WIN-WIN":
                # WIN-WIN: consistency regularisation between eval and train modes
                net.eval()
                with torch.no_grad():
                    logits_eval = net(images)
                net.train()
                logits = net(images)

                p_loss = F.kl_div(
                    F.log_softmax(logits, dim=-1),
                    F.softmax(logits_eval, dim=-1),
                    reduction="none",
                )
                q_loss = F.kl_div(
                    F.log_softmax(logits_eval, dim=-1),
                    F.softmax(logits, dim=-1),
                    reduction="none",
                )
                jsd_loss = (p_loss.sum() + q_loss.sum()) / 2
                loss = (
                    0.5 * (F.cross_entropy(logits_eval, targets) +
                           F.cross_entropy(logits, targets))
                    + 0.3 * jsd_loss
                )
            else:
                logits = net(images)
                loss = F.cross_entropy(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        loss_ema = loss_ema * 0.9 + float(loss) * 0.1
        total_correct += logits.data.max(1)[1].eq(targets.data).sum().item()
        total += targets.size(0)

        if i % args.print_freq == 0:
            print(f"  [batch {i:4d}] loss_ema={loss_ema:.4f}")

    return loss_ema, total_correct / total


def test(net, test_loader, adv=None):
    """Evaluate network on a given dataset.

    Args:
        net: The model to evaluate.
        test_loader: DataLoader for the evaluation set.
        adv: Optional adversarial attack module.

    Returns:
        tuple[float, float]: Mean loss and top-1 accuracy.
    """
    net.eval()
    total_loss = 0.0
    total_correct = 0

    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.cuda(), targets.cuda()
            if adv:
                images = adv(net, images, targets)
            logits = net(images)
            total_loss += float(F.cross_entropy(logits, targets).data)
            total_correct += logits.data.max(1)[1].eq(targets.data).sum().item()

    return total_loss / len(test_loader), total_correct / len(test_loader.dataset)


def test_c(net, test_data, base_path):
    """Evaluate network on a corrupted dataset (CIFAR-C / CIFAR-C-Bar).

    Args:
        net: The model to evaluate.
        test_data: Base dataset object whose ``.data`` and ``.targets`` will
            be mutated per corruption type.
        base_path (str): Directory containing the corruption ``.npy`` files.

    Returns:
        float: Mean top-1 accuracy across all corruption types.
    """
    corruption_accs = []
    corrs = CBAR_CORRUPTIONS if "Bar" in base_path else CORRUPTIONS

    for corruption in corrs:
        test_data.data = np.load(os.path.join(base_path, corruption + ".npy"))
        test_data.targets = torch.LongTensor(
            np.load(os.path.join(base_path, "labels.npy"))
        )
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        test_loss, test_acc = test(net, test_loader)
        corruption_accs.append(test_acc)
        print(f"  {corruption:<35s} loss={test_loss:.3f}  err={100 - 100.*test_acc:.2f}%")

    return float(np.mean(corruption_accs))


# ---------------------------------------------------------------------------
# Adversarial attack (PGD)
# ---------------------------------------------------------------------------

def _normalize_l2(x):
    norm = torch.norm(x.view(x.size(0), -1), p=2, dim=1)
    norm = norm.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    return x / norm


class PGD(nn.Module):
    """Projected Gradient Descent adversarial attack.

    Args:
        epsilon (float): Perturbation budget (in [0, 1] pixel range).
        num_steps (int): Number of PGD iterations.
        step_size (float): Step size per iteration.
        grad_sign (bool): If ``True`` use FGSM-style sign gradient; otherwise
            use normalised gradient.
    """

    def __init__(self, epsilon, num_steps, step_size, grad_sign=True):
        super().__init__()
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.grad_sign = grad_sign

    def forward(self, model, bx, by):
        # Images are in [-1, 1]; convert to [0, 1] for perturbation budget
        bx = (bx + 1) / 2
        adv_bx = bx.detach() + torch.zeros_like(bx).uniform_(
            -self.epsilon, self.epsilon
        )

        for _ in range(self.num_steps):
            adv_bx.requires_grad_()
            with torch.enable_grad():
                logits = model(adv_bx * 2 - 1)
                loss = F.cross_entropy(logits, by, reduction="sum")
            grad = torch.autograd.grad(loss, adv_bx, only_inputs=True)[0]

            if self.grad_sign:
                adv_bx = adv_bx.detach() + self.step_size * torch.sign(grad.detach())
            else:
                adv_bx = adv_bx.detach() + self.step_size * _normalize_l2(grad.detach())

            adv_bx = torch.min(
                torch.max(adv_bx, bx - self.epsilon), bx + self.epsilon
            ).clamp(0, 1)

        return adv_bx * 2 - 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ── data ────────────────────────────────────────────────────────────────
    mean = std = [0.5] * 3
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    cifar_root = os.path.join(args.data_path, "cifar")
    if args.dataset == "cifar10":
        train_data = datasets.CIFAR10(cifar_root, train=True,
                                      transform=train_transform, download=True)
        test_data = datasets.CIFAR10(cifar_root, train=False,
                                     transform=test_transform, download=True)
        base_c_path = os.path.join(cifar_root, "CIFAR-10-C/")
        base_c_bar_path = os.path.join(cifar_root, "CIFAR-10-C-Bar/")
    else:
        train_data = datasets.CIFAR100(cifar_root, train=True,
                                       transform=train_transform, download=True)
        test_data = datasets.CIFAR100(cifar_root, train=False,
                                      transform=test_transform, download=True)
        base_c_path = os.path.join(cifar_root, "CIFAR-100-C/")
        base_c_bar_path = os.path.join(cifar_root, "CIFAR-100-C-Bar/")

    # Worker seed initialisation (fixes PyTorch issue #5059)
    def worker_init_fn(worker_id):
        seed = torch.initial_seed()
        np.random.seed(np.random.SeedSequence([seed]).generate_state(4))

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.eval_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # ── model ────────────────────────────────────────────────────────────────
    net = models.resnet18(weights=None, num_classes=NUM_CLASSES)

    norm_dispatch = {
        "GN":       lambda m: WindowNorm2d.convert_GN_model(m),
        "IN":       lambda m: WindowNorm2d.convert_IN_model(m),
        "Identity": lambda m: WindowNorm2d.convert_Identity_model(m),
        "WIN":      lambda m: WindowNorm2d.convert_WIN_model(m, cached=True),
        "WIN-WIN":  lambda m: WindowNorm2d.convert_WIN_model(m, cached=True),
    }
    if args.norm in norm_dispatch:
        net = norm_dispatch[args.norm](net)
    elif args.norm != "BN":
        raise NotImplementedError(f"Unsupported normalization: '{args.norm}'")

    print(f"Normalization layer : {args.norm}")
    print(net)

    # ── optimiser ────────────────────────────────────────────────────────────
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.decay,
        nesterov=True,
    )

    net = torch.nn.DataParallel(net).cuda()
    cudnn.benchmark = True
    scaler = GradScaler(device="cuda")
    adversary = PGD(epsilon=2.0 / 255, num_steps=1, step_size=2.0 / 255).cuda()

    # ── resume ───────────────────────────────────────────────────────────────
    start_epoch = 0
    best_acc = 0.0
    if args.resume and os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint["epoch"] + 1
        best_acc = checkpoint["best_acc"]
        net.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        print(f"Resumed from epoch {start_epoch}")

    # ── evaluate only ────────────────────────────────────────────────────────
    if args.evaluate:
        test_loss, test_acc = test(net, test_loader)
        print(f"Clean      loss={test_loss:.3f}  err={100 - 100.*test_acc:.2f}%")

        adv_loss, adv_acc = test(net, test_loader, adv=adversary)
        print(f"Adversarial loss={adv_loss:.3f}  err={100 - 100.*adv_acc:.2f}%")

        mce = test_c(net, test_data, base_c_path)
        print(f"Mean Corruption Error: {100 - 100.*mce:.3f}%")
        return

    # ── training loop ────────────────────────────────────────────────────────
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: get_lr(
            step,
            args.epochs * len(train_loader),
            lr_max=1,
            lr_min=1e-6 / args.learning_rate,
        ),
    )

    run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    save_dir = os.path.join(args.save, f"resnet18_{args.norm}_{run_id}")
    os.makedirs(save_dir, exist_ok=True)

    log_path = os.path.join(save_dir, f"{args.dataset}_training_log.csv")
    with open(log_path, "w") as f:
        f.write("epoch,time_s,train_loss,test_loss,train_acc,test_acc\n")

    print(f"Saving to: {save_dir}")
    print(f"Starting training from epoch {start_epoch + 1}")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss, train_acc = train(net, train_loader, optimizer, scheduler, scaler)
        elapsed = time.time() - t0
        test_loss, test_acc = test(net, test_loader)

        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)

        checkpoint = {
            "epoch": epoch,
            "dataset": args.dataset,
            "model": "resnet18",
            "state_dict": net.state_dict(),
            "best_acc": best_acc,
            "optimizer": optimizer.state_dict(),
        }
        ckpt_path = os.path.join(save_dir, "checkpoint.pth.tar")
        torch.save(checkpoint, ckpt_path)
        if is_best:
            shutil.copyfile(ckpt_path, os.path.join(save_dir, "model_best.pth.tar"))

        with open(log_path, "a") as f:
            f.write(
                f"{epoch+1:03d},{elapsed:.1f},{train_loss:.6f},"
                f"{test_loss:.5f},{train_acc:.4f},{test_acc:.4f}\n"
            )

        print(
            f"Epoch {epoch+1:3d} | "
            f"Time {int(elapsed):5d}s | "
            f"Train Loss {train_loss:.4f} | "
            f"Test Loss {test_loss:.3f} | "
            f"Train Acc {100.*train_acc:.2f}% | "
            f"Test Acc {100.*test_acc:.2f}%"
        )

    _, adv_acc = test(net, test_loader, adv=adversary)
    print(f"Final Adversarial Test Error: {100 - 100.*adv_acc:.3f}%")


if __name__ == "__main__":
    main()
