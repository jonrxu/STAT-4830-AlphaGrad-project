#!/usr/bin/env python3
"""Fast CIFAR-10 trainer targeting >=94% TTA accuracy on one A100.

Keeps the evaluator CLI/JSON contract and uses an AirBench-style recipe with
slightly tuned widths/steps for a bit more margin while preserving speed.
"""

from __future__ import annotations

import argparse
import json
from math import ceil
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch import nn


torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True


@torch.compile(mode="max-autotune")
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 3, eps: float = 1e-7):
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.to(torch.bfloat16)
    X = X / (X.norm() + eps)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0.0, nesterov=False):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                g_eff = g.add(buf, alpha=momentum) if nesterov else buf
                p_norm = p.data.norm()
                if p_norm > 0:
                    p.data.mul_(len(p.data) ** 0.5 / p_norm)
                update = zeropower_via_newtonschulz5(g_eff.reshape(len(g_eff), -1)).view_as(g_eff)
                p.data.add_(update, alpha=-lr)


CIFAR_MEAN = torch.tensor((0.4914, 0.4822, 0.4465), device="cuda").view(1, 3, 1, 1)
CIFAR_STD = torch.tensor((0.2470, 0.2435, 0.2616), device="cuda").view(1, 3, 1, 1)


def batch_flip_lr(inputs: torch.Tensor) -> torch.Tensor:
    flip_mask = (torch.rand(len(inputs), device=inputs.device) < 0.5).view(-1, 1, 1, 1)
    return torch.where(flip_mask, inputs.flip(-1), inputs)


def batch_crop(images: torch.Tensor, crop_size: int) -> torch.Tensor:
    r = (images.size(-1) - crop_size) // 2
    shifts = torch.randint(-r, r + 1, size=(len(images), 2), device=images.device)
    images_out = torch.empty(
        (len(images), images.size(1), crop_size, crop_size),
        device=images.device,
        dtype=images.dtype,
    )
    if r <= 2:
        for sy in range(-r, r + 1):
            for sx in range(-r, r + 1):
                mask = (shifts[:, 0] == sy) & (shifts[:, 1] == sx)
                if mask.any():
                    images_out[mask] = images[mask, :, r + sy : r + sy + crop_size, r + sx : r + sx + crop_size]
    else:
        images_tmp = torch.empty(
            (len(images), images.size(1), crop_size, crop_size + 2 * r),
            device=images.device,
            dtype=images.dtype,
        )
        for s in range(-r, r + 1):
            mask = shifts[:, 0] == s
            if mask.any():
                images_tmp[mask] = images[mask, :, r + s : r + s + crop_size, :]
        for s in range(-r, r + 1):
            mask = shifts[:, 1] == s
            if mask.any():
                images_out[mask] = images_tmp[mask, :, :, r + s : r + s + crop_size]
    return images_out


class CifarLoader:
    def __init__(self, path: str, train: bool = True, batch_size: int = 500, aug=None):
        data_path = Path(path) / ("train.pt" if train else "test.pt")
        if not data_path.exists():
            dset = torchvision.datasets.CIFAR10(path, download=True, train=train)
            images = torch.tensor(dset.data)
            labels = torch.tensor(dset.targets)
            data_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"images": images, "labels": labels, "classes": dset.classes}, data_path)

        data = torch.load(data_path, map_location="cuda", weights_only=False)
        self.images = (data["images"].to(device="cuda", dtype=torch.float16) / 255.0).permute(0, 3, 1, 2)
        self.images = self.images.contiguous(memory_format=torch.channels_last)
        self.labels = data["labels"].to(device="cuda")
        self.classes = data.get("classes", None)

        self.proc_images = {}
        self.epoch = 0
        self.aug = aug or {}
        self.batch_size = batch_size
        self.drop_last = train
        self.shuffle = train

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - CIFAR_MEAN) / CIFAR_STD

    def __len__(self):
        return len(self.images) // self.batch_size if self.drop_last else ceil(len(self.images) / self.batch_size)

    def __iter__(self):
        if self.epoch == 0:
            images = self.proc_images["norm"] = self.normalize(self.images)
            if self.aug.get("flip", False):
                images = self.proc_images["flip"] = batch_flip_lr(images)
            pad = self.aug.get("translate", 0)
            if pad > 0:
                self.proc_images["pad"] = F.pad(images, (pad, pad, pad, pad), mode="reflect")

        if self.aug.get("translate", 0) > 0:
            images = batch_crop(self.proc_images["pad"], self.images.shape[-2])
        elif self.aug.get("flip", False):
            images = self.proc_images["flip"]
        else:
            images = self.proc_images["norm"]

        if self.aug.get("flip", False) and (self.epoch % 2 == 1):
            images = images.flip(-1)

        self.epoch += 1
        indices = torch.randperm(len(images), device=images.device) if self.shuffle else torch.arange(len(images), device=images.device)
        for i in range(len(self)):
            idxs = indices[i * self.batch_size : (i + 1) * self.batch_size]
            yield images[idxs], self.labels[idxs]


class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, momentum=0.6, eps=1e-12):
        super().__init__(num_features, eps=eps, momentum=1 - momentum)
        self.weight.requires_grad = False


class Conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels, kernel_size=3, padding="same", bias=False)

    def reset_parameters(self):
        super().reset_parameters()
        w = self.weight.data
        torch.nn.init.dirac_(w[: w.size(1)])


class ConvGroup(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.conv1 = Conv(channels_in, channels_out)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(channels_out)
        self.conv2 = Conv(channels_out, channels_out)
        self.norm2 = BatchNorm(channels_out)
        self.activ = nn.GELU()

    def forward(self, x):
        x = self.activ(self.norm1(self.pool(self.conv1(x))))
        x = self.activ(self.norm2(self.conv2(x)))
        return x


class CifarNet(nn.Module):
    def __init__(self, widths=(64, 256, 320)):
        super().__init__()
        whiten_kernel_size = 2
        whiten_width = 2 * 3 * whiten_kernel_size**2
        self.whiten = nn.Conv2d(3, whiten_width, whiten_kernel_size, padding=0, bias=True)
        self.whiten.weight.requires_grad = False
        self.layers = nn.Sequential(
            nn.GELU(),
            ConvGroup(whiten_width, widths[0]),
            ConvGroup(widths[0], widths[1]),
            ConvGroup(widths[1], widths[2]),
            nn.MaxPool2d(3),
        )
        self.head = nn.Linear(widths[2], 10, bias=False)
        for mod in self.modules():
            if isinstance(mod, BatchNorm):
                mod.float()
            else:
                mod.half()

    def reset(self):
        for mod in self.modules():
            if type(mod) in (nn.Conv2d, Conv, BatchNorm, nn.Linear):
                mod.reset_parameters()
        self.head.weight.data.mul_(1 / self.head.weight.data.std())

    @torch.no_grad()
    def init_whiten(self, train_images: torch.Tensor, eps: float = 5e-4):
        c, (h, w) = train_images.shape[1], self.whiten.weight.shape[2:]
        patches = train_images.unfold(2, h, 1).unfold(3, w, 1).transpose(1, 3).reshape(-1, c, h, w).float()
        patches_flat = patches.view(len(patches), -1)
        cov = (patches_flat.T @ patches_flat) / len(patches_flat)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov, UPLO="U")
        eigenvectors_scaled = eigenvectors.T.reshape(-1, c, h, w) / torch.sqrt(eigenvalues.view(-1, 1, 1, 1) + eps)
        self.whiten.weight.data.copy_(torch.cat((eigenvectors_scaled, -eigenvectors_scaled)).to(self.whiten.weight.dtype))
        self.whiten.bias.data.zero_()

    def forward(self, x, whiten_bias_grad: bool = True):
        bias = self.whiten.bias if whiten_bias_grad else self.whiten.bias.detach()
        x = F.conv2d(x, self.whiten.weight, bias)
        x = self.layers(x)
        x = x.view(len(x), -1)
        return self.head(x) / x.size(-1)


def infer(model: nn.Module, loader: CifarLoader, tta_level: int = 0):
    def basic(inp):
        return model(inp).clone()

    def mirror(inp):
        return 0.5 * model(inp) + 0.5 * model(inp.flip(-1))

    def mirror_translate(inp):
        logits = mirror(inp)
        padded = F.pad(inp, (1, 1, 1, 1), mode="reflect")
        translated = [padded[:, :, 0:32, 0:32], padded[:, :, 2:34, 2:34]]
        logits_translate = torch.stack([mirror(z) for z in translated]).mean(0)
        return 0.5 * logits + 0.5 * logits_translate

    infer_fn = [basic, mirror, mirror_translate][tta_level]
    model.eval()
    test_images = loader.normalize(loader.images)
    with torch.no_grad():
        return torch.cat([infer_fn(inp) for inp in test_images.split(2000)])


def evaluate(model: nn.Module, loader: CifarLoader, tta_level: int = 0) -> float:
    logits = infer(model, loader, tta_level)
    return (logits.argmax(1) == loader.labels).float().mean().item()


def set_trial_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_single_trial(model: CifarNet, data_dir: str, use_dummy_labels: bool = False) -> dict[str, float]:
    batch_size = 2000
    bias_lr = 0.053
    head_lr = 0.70
    wd = 2e-6 * batch_size

    test_loader = CifarLoader(data_dir, train=False, batch_size=2000)
    train_loader = CifarLoader(data_dir, train=True, batch_size=batch_size, aug=dict(flip=True, translate=2))
    if use_dummy_labels:
        train_loader.labels = torch.randint(0, 10, size=train_loader.labels.shape, device=train_loader.labels.device)

    total_train_steps = ceil(8.5 * len(train_loader))
    whiten_bias_train_steps = ceil(3.0 * len(train_loader))

    filter_params = [p for p in model.parameters() if p.requires_grad and p.ndim == 4]
    norm_biases = [p for name, p in model.named_parameters() if ("norm" in name and p.requires_grad)]

    optimizer1 = torch.optim.SGD(
        [
            dict(params=[model.whiten.bias], lr=bias_lr, weight_decay=wd / bias_lr),
            dict(params=norm_biases, lr=bias_lr, weight_decay=wd / bias_lr),
            dict(params=[model.head.weight], lr=head_lr, weight_decay=wd / head_lr),
        ],
        momentum=0.85,
        nesterov=True,
        fused=True,
    )
    optimizer2 = Muon(filter_params, lr=0.24, momentum=0.6, nesterov=True)
    optimizers = [optimizer1, optimizer2]
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    time_seconds = 0.0

    def start_timer():
        starter.record()

    def stop_timer():
        nonlocal time_seconds
        ender.record()
        torch.cuda.synchronize()
        time_seconds += 1e-3 * starter.elapsed_time(ender)

    model.reset()
    step = 0

    start_timer()
    model.init_whiten(train_loader.normalize(train_loader.images[:5000]))
    stop_timer()

    epochs = ceil(total_train_steps / len(train_loader))
    for _ in range(epochs):
        start_timer()
        model.train()
        for inputs, labels in train_loader:
            outputs = model(inputs, whiten_bias_grad=(step < whiten_bias_train_steps))
            F.cross_entropy(outputs, labels, label_smoothing=0.2, reduction="sum").backward()

            frac_whiten = max(0.0, 1 - step / whiten_bias_train_steps)
            frac_main = max(0.0, 1 - step / total_train_steps)
            optimizer1.param_groups[0]["lr"] = optimizer1.param_groups[0]["initial_lr"] * frac_whiten
            for group in optimizer1.param_groups[1:]:
                group["lr"] = group["initial_lr"] * frac_main
            for group in optimizer2.param_groups:
                group["lr"] = group["initial_lr"] * frac_main

            for opt in optimizers:
                opt.step()
            model.zero_grad(set_to_none=True)
            step += 1
            if step >= total_train_steps:
                break
        stop_timer()
        if step >= total_train_steps:
            break

    start_timer()
    tta_val_acc = evaluate(model, test_loader, tta_level=2)
    stop_timer()

    return {"tta_val_accuracy": float(tta_val_acc), "time_seconds": float(time_seconds)}


def run_preflight(model: CifarNet, data_dir: str) -> dict[str, float]:
    batch_size = 1024
    train_loader = CifarLoader(data_dir, train=True, batch_size=batch_size, aug=dict(flip=True, translate=2))
    test_loader = CifarLoader(data_dir, train=False, batch_size=batch_size)

    model.reset()
    model.init_whiten(train_loader.normalize(train_loader.images[:2048]))

    model.train()
    inputs, labels = next(iter(train_loader))
    outputs = model(inputs, whiten_bias_grad=True)
    loss = F.cross_entropy(outputs, labels, label_smoothing=0.2, reduction="mean")
    loss.backward()
    model.zero_grad(set_to_none=True)

    model.eval()
    with torch.no_grad():
        eval_inputs = test_loader.normalize(test_loader.images[:batch_size])
        _ = model(eval_inputs)
    torch.cuda.synchronize()
    return {
        "train_batch_size": float(len(inputs)),
        "eval_batch_size": float(len(eval_inputs)),
        "loss": float(loss.item()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=str, default="/vol/cifar10")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--warmup-trials", type=int, default=1)
    parser.add_argument("--target-accuracy", type=float, default=0.94)
    parser.add_argument("--json-only", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--disable-compile", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    target_accuracy = args.target_accuracy / 100.0 if args.target_accuracy > 1.0 else args.target_accuracy

    model = CifarNet().cuda().to(memory_format=torch.channels_last)
    if not args.disable_compile:
        model = torch.compile(model, mode="max-autotune")

    if args.verbose:
        print(
            f"[config] data_dir={args.data_dir} trials={args.trials} "
            f"warmup_trials={args.warmup_trials} target_accuracy={target_accuracy:.4f}"
        )

    if args.preflight:
        set_trial_seed(args.seed)
        preflight_result = run_preflight(model, args.data_dir)
        if args.verbose:
            print(
                "[preflight] ok "
                f"train_batch_size={preflight_result['train_batch_size']:.0f} "
                f"eval_batch_size={preflight_result['eval_batch_size']:.0f} "
                f"loss={preflight_result['loss']:.4f}"
            )

    for warmup_idx in range(args.warmup_trials):
        set_trial_seed(args.seed + warmup_idx)
        warmup_result = run_single_trial(model, args.data_dir, use_dummy_labels=True)
        if args.verbose:
            print(f"[warmup {warmup_idx + 1}/{args.warmup_trials}] time_seconds={warmup_result['time_seconds']:.4f}")

    per_trial = []
    for trial_idx in range(args.trials):
        set_trial_seed(args.seed + args.warmup_trials + trial_idx)
        trial_result = run_single_trial(model, args.data_dir, use_dummy_labels=False)
        per_trial.append(trial_result)
        if args.verbose:
            print(
                f"[trial {trial_idx + 1}/{args.trials}] "
                f"tta_val_accuracy={trial_result['tta_val_accuracy']:.4f} "
                f"time_seconds={trial_result['time_seconds']:.4f}"
            )

    accuracies = torch.tensor([x["tta_val_accuracy"] for x in per_trial], dtype=torch.float64)
    times = torch.tensor([x["time_seconds"] for x in per_trial], dtype=torch.float64)
    result = {
        "mean_accuracy": float(accuracies.mean().item()),
        "std_accuracy": float(accuracies.std(unbiased=False).item()),
        "mean_time_seconds": float(times.mean().item()),
        "std_time_seconds": float(times.std(unbiased=False).item()),
        "meets_target": bool(float(accuracies.mean().item()) >= target_accuracy),
        "target_accuracy": float(target_accuracy),
        "trials": int(args.trials),
        "warmup_trials": int(args.warmup_trials),
        "per_trial": per_trial,
        "torch_version": torch.__version__,
        "device_name": torch.cuda.get_device_name(0),
        "compiled": bool(not args.disable_compile),
    }

    if args.json_only:
        print(json.dumps(result))
    else:
        print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
