#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


class Normalize(nn.Module):
    def __init__(self, mean=CIFAR_MEAN, std=CIFAR_STD):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, drop_prob=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.drop_prob = drop_prob
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        if self.training and self.drop_prob > 0.0:
            out = F.dropout(out, p=self.drop_prob, inplace=False)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


class SmallResNet(nn.Module):
    def __init__(self, num_classes=10, width=96, drop_prob=0.08):
        super().__init__()
        self.normalize = Normalize()
        self.in_planes = width
        self.conv1 = nn.Conv2d(3, width, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.layer1 = self._make_layer(width, 4, stride=1, drop_prob=drop_prob * 0.5)
        self.layer2 = self._make_layer(width * 2, 4, stride=2, drop_prob=drop_prob)
        self.layer3 = self._make_layer(width * 4, 4, stride=2, drop_prob=drop_prob)
        self.head_bn = nn.BatchNorm1d(width * 4)
        self.fc = nn.Linear(width * 4, num_classes)

    def _make_layer(self, planes, blocks, stride, drop_prob):
        layers = [BasicBlock(self.in_planes, planes, stride=stride, drop_prob=drop_prob)]
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_planes, planes, stride=1, drop_prob=drop_prob))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.normalize(x)
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        x = self.head_bn(x)
        return self.fc(x)


def safe_torch_load(path, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def ensure_cifar_pt(data_dir: str, train: bool):
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    pt_path = root / ("train.pt" if train else "test.pt")
    if not pt_path.exists():
        ds = torchvision.datasets.CIFAR10(root=str(root), train=train, download=True)
        images = torch.tensor(ds.data)
        labels = torch.tensor(ds.targets)
        torch.save({"images": images, "labels": labels}, pt_path)
    return pt_path


def load_cifar_tensors(data_dir: str):
    train_data = safe_torch_load(ensure_cifar_pt(data_dir, True), map_location="cpu")
    test_data = safe_torch_load(ensure_cifar_pt(data_dir, False), map_location="cpu")
    x_train = train_data["images"].permute(0, 3, 1, 2).contiguous().float().div_(255.0)
    y_train = train_data["labels"].long()
    x_test = test_data["images"].permute(0, 3, 1, 2).contiguous().float().div_(255.0)
    y_test = test_data["labels"].long()
    return x_train, y_train, x_test, y_test


def random_crop_flip(images: torch.Tensor, pad: int = 4):
    n, c, h, w = images.shape
    x = F.pad(images, (pad, pad, pad, pad), mode="reflect")
    ys = torch.randint(0, 2 * pad + 1, (n,), device=images.device)
    xs = torch.randint(0, 2 * pad + 1, (n,), device=images.device)
    out = torch.empty((n, c, h, w), device=images.device, dtype=images.dtype)
    for oy in range(2 * pad + 1):
        mask_y = ys == oy
        if not mask_y.any():
            continue
        rows = x[mask_y, :, oy : oy + h, :]
        xs_sub = xs[mask_y]
        tmp = torch.empty((rows.size(0), c, h, w), device=images.device, dtype=images.dtype)
        for ox in range(2 * pad + 1):
            mask_x = xs_sub == ox
            if mask_x.any():
                tmp[mask_x] = rows[mask_x, :, :, ox : ox + w]
        out[mask_y] = tmp
    flip = torch.rand(n, device=images.device) < 0.5
    out[flip] = out[flip].flip(-1)
    return out


def cutmix(images, targets, alpha=1.0):
    if alpha <= 0:
        return images, targets, targets, 1.0
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    n, _, h, w = images.shape
    index = torch.randperm(n, device=images.device)
    cut_rat = math.sqrt(1.0 - lam)
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)
    cx = torch.randint(w, (1,), device=images.device).item()
    cy = torch.randint(h, (1,), device=images.device).item()
    x1 = max(cx - cut_w // 2, 0)
    x2 = min(cx + cut_w // 2, w)
    y1 = max(cy - cut_h // 2, 0)
    y2 = min(cy + cut_h // 2, h)
    images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
    lam = 1.0 - ((x2 - x1) * (y2 - y1) / (w * h))
    return images, targets, targets[index], lam


def accuracy_from_logits(logits, targets):
    return (logits.argmax(dim=1) == targets).float().mean().item()


@torch.no_grad()
def evaluate(model, x_test_gpu, y_test_gpu, batch_size=1000, tta=True):
    model.eval()
    total = 0
    correct = 0
    for i in range(0, x_test_gpu.size(0), batch_size):
        x = x_test_gpu[i : i + batch_size]
        y = y_test_gpu[i : i + batch_size]
        logits = model(x)
        if tta:
            logits = 0.5 * (logits + model(x.flip(-1)))
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total


def make_optimizer(model, lr, weight_decay):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or name.endswith("bias") or "bn" in name or "normalize" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.95),
        fused=True,
    )


def cosine_lr(step, total_steps, warmup_steps, base_lr):
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * (0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * t)))


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_trial(data_dir: str, use_compile: bool, seed: int, use_dummy_labels: bool = False):
    set_seed(seed)
    device = torch.device("cuda")
    x_train, y_train, x_test, y_test = load_cifar_tensors(data_dir)
    if use_dummy_labels:
        y_train = torch.randint(0, 10, y_train.shape, dtype=y_train.dtype)

    x_train_gpu = x_train.to(device=device, non_blocking=True, memory_format=torch.channels_last)
    y_train_gpu = y_train.to(device=device, non_blocking=True)
    x_test_gpu = x_test.to(device=device, non_blocking=True, memory_format=torch.channels_last)
    y_test_gpu = y_test.to(device=device, non_blocking=True)

    model = SmallResNet(width=96, drop_prob=0.08).to(device)
    model.to(memory_format=torch.channels_last)
    model = model.to(dtype=torch.float32)

    if use_compile:
        try:
            model = torch.compile(model, mode="default")
            compiled = True
        except Exception:
            compiled = False
    else:
        compiled = False

    epochs = 30
    batch_size = 768
    base_lr = 3e-3
    weight_decay = 0.02
    warmup_steps = 20
    mix_epochs = 20
    label_smoothing = 0.05

    optimizer = make_optimizer(model, base_lr, weight_decay)
    scaler = torch.amp.GradScaler("cuda")

    steps_per_epoch = x_train_gpu.size(0) // batch_size
    total_steps = steps_per_epoch * epochs
    ema = None
    ema_decay = 0.999

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    starter.record()

    step = 0
    n = x_train_gpu.size(0)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for bi in range(steps_per_epoch):
            idx = perm[bi * batch_size : (bi + 1) * batch_size]
            x = x_train_gpu[idx]
            y = y_train_gpu[idx]
            x = random_crop_flip(x, pad=4)
            if epoch < mix_epochs and torch.rand(()) < 0.5:
                x, ya, yb, lam = cutmix(x, y, alpha=1.0)
                mixed = True
            else:
                mixed = False

            lr = cosine_lr(step, total_steps, warmup_steps, base_lr)
            for g in optimizer.param_groups:
                g["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(x)
                if mixed:
                    loss = lam * F.cross_entropy(logits, ya, label_smoothing=label_smoothing) + (1.0 - lam) * F.cross_entropy(logits, yb, label_smoothing=label_smoothing)
                else:
                    loss = F.cross_entropy(logits, y, label_smoothing=label_smoothing)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if ema is None:
                ema = {k: v.detach().float().clone() for k, v in model.state_dict().items()}
            else:
                state = model.state_dict()
                for k in ema.keys():
                    ema[k].mul_(ema_decay).add_(state[k].detach().float(), alpha=1.0 - ema_decay)
            step += 1

    eval_model = model
    if ema is not None:
        raw = model.state_dict()
        backup = {k: v.detach().clone() for k, v in raw.items()}
        ema_cast = {k: ema[k].to(device=raw[k].device, dtype=raw[k].dtype) for k in raw.keys()}
        model.load_state_dict(ema_cast, strict=True)
    else:
        backup = None

    acc = evaluate(eval_model, x_test_gpu, y_test_gpu, batch_size=1000, tta=True)

    if backup is not None:
        model.load_state_dict(backup, strict=True)

    ender.record()
    torch.cuda.synchronize()
    time_seconds = starter.elapsed_time(ender) / 1000.0

    return {
        "tta_val_accuracy": float(acc),
        "time_seconds": float(time_seconds),
        "compiled": compiled,
    }


def run_preflight(data_dir: str, use_compile: bool, seed: int):
    set_seed(seed)
    device = torch.device("cuda")
    x_train, y_train, x_test, y_test = load_cifar_tensors(data_dir)
    x = x_train[:256].to(device=device, non_blocking=True, memory_format=torch.channels_last)
    y = y_train[:256].to(device=device, non_blocking=True)
    xt = x_test[:512].to(device=device, non_blocking=True, memory_format=torch.channels_last)
    yt = y_test[:512].to(device=device, non_blocking=True)

    model = SmallResNet(width=64, drop_prob=0.05).to(device).to(memory_format=torch.channels_last)
    if use_compile:
        try:
            model = torch.compile(model, mode="default")
        except Exception:
            pass
    optimizer = make_optimizer(model, 1e-3, 0.01)
    scaler = torch.amp.GradScaler("cuda")
    model.train()
    x = random_crop_flip(x)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        logits = model(x)
        loss = F.cross_entropy(logits, y)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    acc = evaluate(model, xt, yt, batch_size=256, tta=True)
    torch.cuda.synchronize()
    return {
        "loss": float(loss.item()),
        "sample_tta_accuracy": float(acc),
        "train_batch_size": 256,
        "eval_batch_size": 256,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default="/vol/cifar10")
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--warmup-trials", type=int, default=1)
    p.add_argument("--target-accuracy", type=float, default=0.94)
    p.add_argument("--json-only", action="store_true")
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--disable-compile", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    target_accuracy = args.target_accuracy / 100.0 if args.target_accuracy > 1.0 else args.target_accuracy
    use_compile = not args.disable_compile
    result = None
    exit_code = 0
    try:
        preflight = None
        if args.preflight:
            preflight = run_preflight(args.data_dir, use_compile, args.seed)

        for i in range(args.warmup_trials):
            train_one_trial(args.data_dir, use_compile, args.seed + i, use_dummy_labels=True)

        trials = []
        compiled_any = False
        for i in range(args.trials):
            r = train_one_trial(args.data_dir, use_compile, args.seed + args.warmup_trials + i, use_dummy_labels=False)
            compiled_any = compiled_any or r.get("compiled", False)
            trials.append({"tta_val_accuracy": r["tta_val_accuracy"], "time_seconds": r["time_seconds"]})

        accs = torch.tensor([t["tta_val_accuracy"] for t in trials], dtype=torch.float64)
        times = torch.tensor([t["time_seconds"] for t in trials], dtype=torch.float64)
        result = {
            "mean_accuracy": float(accs.mean().item()),
            "std_accuracy": float(accs.std(unbiased=False).item()),
            "mean_time_seconds": float(times.mean().item()),
            "std_time_seconds": float(times.std(unbiased=False).item()),
            "meets_target": bool(float(accs.mean().item()) >= target_accuracy),
            "target_accuracy": float(target_accuracy),
            "trials": int(args.trials),
            "warmup_trials": int(args.warmup_trials),
            "per_trial": trials,
            "torch_version": torch.__version__,
            "device_name": torch.cuda.get_device_name(0),
            "compiled": bool(compiled_any),
        }
        if preflight is not None:
            result["preflight"] = preflight
    except Exception as exc:
        exit_code = 1
        result = {
            "mean_accuracy": 0.0,
            "mean_time_seconds": 1.0,
            "trials": int(args.trials),
            "warmup_trials": int(args.warmup_trials),
            "per_trial": [],
            "meets_target": False,
            "target_accuracy": float(target_accuracy),
            "torch_version": torch.__version__,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "compiled": False,
            "failure_type": exc.__class__.__name__,
            "failure_message": str(exc),
        }

    print(json.dumps(result, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
