#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import v2 as T


torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        nll = -log_probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth = -log_probs.mean(dim=-1)
        return ((1.0 - self.smoothing) * nll + self.smoothing * smooth).mean()


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1, drop_p: float = 0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.drop_p = drop_p
        self.shortcut = None
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.silu(self.bn1(x), inplace=True)
        shortcut = x if self.shortcut is None else self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.silu(self.bn2(out), inplace=True))
        if self.drop_p > 0.0 and self.training:
            out = F.dropout(out, p=self.drop_p)
        return out + shortcut


class WideResNet(nn.Module):
    def __init__(self, depth: int = 28, widen_factor: int = 10, num_classes: int = 10, drop_p: float = 0.0):
        super().__init__()
        assert (depth - 4) % 6 == 0
        n = (depth - 4) // 6
        widths = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        self.conv1 = nn.Conv2d(3, widths[0], 3, stride=1, padding=1, bias=False)
        self.block1 = self._make_layer(widths[0], widths[1], n, stride=1, drop_p=drop_p)
        self.block2 = self._make_layer(widths[1], widths[2], n, stride=2, drop_p=drop_p)
        self.block3 = self._make_layer(widths[2], widths[3], n, stride=2, drop_p=drop_p)
        self.bn = nn.BatchNorm2d(widths[3])
        self.fc = nn.Linear(widths[3], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0.0)

    def _make_layer(self, in_planes: int, planes: int, num_blocks: int, stride: int, drop_p: float):
        layers = [BasicBlock(in_planes, planes, stride=stride, drop_p=drop_p)]
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(planes, planes, stride=1, drop_p=drop_p))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = F.silu(self.bn(x), inplace=True)
        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        return self.fc(x)


def get_data_dir(root: str) -> str:
    p = Path(root)
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


def build_datasets(data_dir: str):
    data_dir = get_data_dir(data_dir)
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4, padding_mode="reflect"),
        T.RandomHorizontalFlip(),
        T.AutoAugment(policy=T.AutoAugmentPolicy.CIFAR10),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
        T.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value="random"),
    ])
    test_tf = T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_tf)
    test_set = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_tf)
    return train_set, test_set


def make_loaders(data_dir: str, batch_size: int = 512, eval_batch_size: int = 1024):
    train_set, test_set = build_datasets(data_dir)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    return train_loader, test_loader


def tta_logits(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    logits = model(x)
    logits += model(torch.flip(x, dims=[3]))
    xpad = F.pad(x, (2, 2, 2, 2), mode="reflect")
    logits += model(xpad[:, :, 0:32, 0:32])
    logits += model(torch.flip(xpad[:, :, 0:32, 0:32], dims=[3]))
    logits += model(xpad[:, :, 4:36, 4:36])
    logits += model(torch.flip(xpad[:, :, 4:36, 4:36], dims=[3]))
    return logits / 6.0


@torch.no_grad()
def evaluate(model: nn.Module, loader, use_tta: bool = True) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.cuda(non_blocking=True).contiguous(memory_format=torch.channels_last)
        y = y.cuda(non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = tta_logits(model, x) if use_tta else model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def maybe_compile(model: nn.Module, enable: bool):
    if not enable:
        return model
    try:
        return torch.compile(model, mode="reduce-overhead")
    except Exception:
        return model


def train_one_trial(data_dir: str, disable_compile: bool, seed: int, epochs: int = 25, use_dummy_labels: bool = False):
    set_seed(seed)
    train_loader, test_loader = make_loaders(data_dir)

    model = WideResNet(depth=28, widen_factor=10, drop_p=0.0).cuda().to(memory_format=torch.channels_last)
    model = maybe_compile(model, not disable_compile)

    criterion = LabelSmoothingCrossEntropy(0.1)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.4,
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=True,
        fused=True,
    )
    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch
    warmup_steps = 5 * steps_per_epoch

    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        t = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + torch.cos(torch.tensor(t * 3.141592653589793))).item()

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    starter.record()

    global_step = 0
    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            x = x.cuda(non_blocking=True).contiguous(memory_format=torch.channels_last)
            y = y.cuda(non_blocking=True)
            if use_dummy_labels:
                y = torch.randint(0, 10, y.shape, device=y.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1

    ender.record()
    torch.cuda.synchronize()
    train_time = starter.elapsed_time(ender) / 1000.0

    eval_start = time.perf_counter()
    acc = evaluate(model, test_loader, use_tta=True)
    eval_time = time.perf_counter() - eval_start
    return {"tta_val_accuracy": float(acc), "time_seconds": float(train_time + eval_time)}


def run_preflight(data_dir: str, disable_compile: bool, seed: int):
    set_seed(seed)
    train_loader, test_loader = make_loaders(data_dir, batch_size=128, eval_batch_size=256)
    model = WideResNet(depth=16, widen_factor=2, drop_p=0.0).cuda().to(memory_format=torch.channels_last)
    model = maybe_compile(model, not disable_compile)
    criterion = LabelSmoothingCrossEntropy(0.1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, fused=True)
    x, y = next(iter(train_loader))
    x = x.cuda(non_blocking=True).contiguous(memory_format=torch.channels_last)
    y = y.cuda(non_blocking=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        logits = model(x)
        loss = criterion(logits, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    x2, _ = next(iter(test_loader))
    x2 = x2.cuda(non_blocking=True).contiguous(memory_format=torch.channels_last)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        _ = model(x2)
    torch.cuda.synchronize()
    return {"train_batch_size": float(x.shape[0]), "eval_batch_size": float(x2.shape[0]), "loss": float(loss.item())}


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
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    target_accuracy = args.target_accuracy / 100.0 if args.target_accuracy > 1.0 else args.target_accuracy

    if args.preflight:
        pf = run_preflight(args.data_dir, args.disable_compile, args.seed)
        if not args.json_only:
            print(json.dumps({"preflight": pf}))

    for i in range(args.warmup_trials):
        _ = train_one_trial(args.data_dir, args.disable_compile, args.seed + i, epochs=1, use_dummy_labels=True)

    per_trial = []
    for i in range(args.trials):
        result = train_one_trial(
            args.data_dir,
            args.disable_compile,
            args.seed + args.warmup_trials + i,
            epochs=25,
            use_dummy_labels=False,
        )
        per_trial.append(result)
        if args.verbose and not args.json_only:
            print(json.dumps({"trial": i + 1, **result}))

    accs = torch.tensor([x["tta_val_accuracy"] for x in per_trial], dtype=torch.float64) if per_trial else torch.tensor([0.0], dtype=torch.float64)
    times = torch.tensor([x["time_seconds"] for x in per_trial], dtype=torch.float64) if per_trial else torch.tensor([1e-9], dtype=torch.float64)
    mean_acc = float(accs.mean().item()) if per_trial else 0.0
    result = {
        "mean_accuracy": mean_acc,
        "mean_time_seconds": float(times.mean().item()) if per_trial else 1e-9,
        "std_accuracy": float(accs.std(unbiased=False).item()) if per_trial else 0.0,
        "std_time_seconds": float(times.std(unbiased=False).item()) if per_trial else 0.0,
        "trials": int(args.trials),
        "warmup_trials": int(args.warmup_trials),
        "target_accuracy": float(target_accuracy),
        "meets_target": bool(mean_acc >= target_accuracy),
        "per_trial": per_trial,
        "torch_version": torch.__version__,
        "device_name": torch.cuda.get_device_name(0),
        "compiled": bool(not args.disable_compile),
    }
    if args.json_only:
        print(json.dumps(result))
    else:
        print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())