#!/usr/bin/env python3
"""AirBench 94 baseline with a CLI and JSON output contract."""

from __future__ import annotations

import argparse
import json
from math import ceil

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch import nn


torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True


@torch.compile
def zeropower_via_newtonschulz5(G, steps=3, eps=1e-7):
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    orig_dtype = G.dtype
    work_dtype = torch.bfloat16 if G.is_cuda else torch.float32
    X = G.to(dtype=work_dtype)
    X = X / (X.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(dtype=orig_dtype)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0, nesterov=False):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if nesterov and momentum <= 0:
            raise ValueError("Nesterov momentum requires a momentum")
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                if not torch.isfinite(g).all():
                    raise RuntimeError("Muon received non-finite gradients")
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                g_eff = g.add(buf, alpha=momentum) if group["nesterov"] else buf

                p_norm = p.data.float().norm()
                if torch.isfinite(p_norm) and p_norm.item() > 0.0:
                    p.data.mul_(len(p.data) ** 0.5 / p_norm.to(dtype=p.data.dtype))

                update = zeropower_via_newtonschulz5(g_eff.reshape(len(g_eff), -1)).view(g_eff.shape)
                if update.dtype != p.data.dtype:
                    update = update.to(dtype=p.data.dtype)
                p.data.add_(update, alpha=-lr)


CIFAR_MEAN = torch.tensor((0.4914, 0.4822, 0.4465))
CIFAR_STD = torch.tensor((0.2470, 0.2435, 0.2616))


def safe_torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def batch_flip_lr(inputs):
    flip_mask = (torch.rand(len(inputs), device=inputs.device) < 0.5).view(-1, 1, 1, 1)
    return torch.where(flip_mask, inputs.flip(-1), inputs)


def batch_crop(images, crop_size):
    r = (images.size(-1) - crop_size) // 2
    shifts = torch.randint(-r, r + 1, size=(len(images), 2), device=images.device)
    images_out = torch.empty((len(images), 3, crop_size, crop_size), device=images.device, dtype=images.dtype)
    if r <= 2:
        for sy in range(-r, r + 1):
            for sx in range(-r, r + 1):
                mask = (shifts[:, 0] == sy) & (shifts[:, 1] == sx)
                images_out[mask] = images[mask, :, r + sy : r + sy + crop_size, r + sx : r + sx + crop_size]
    else:
        images_tmp = torch.empty(
            (len(images), 3, crop_size, crop_size + 2 * r),
            device=images.device,
            dtype=images.dtype,
        )
        for s in range(-r, r + 1):
            mask = shifts[:, 0] == s
            images_tmp[mask] = images[mask, :, r + s : r + s + crop_size, :]
        for s in range(-r, r + 1):
            mask = shifts[:, 1] == s
            images_out[mask] = images_tmp[mask, :, :, r + s : r + s + crop_size]
    return images_out
class CifarLoader:
    def __init__(self, path, train=True, batch_size=500, aug=None):
        from pathlib import Path

        data_path = Path(path) / ("train.pt" if train else "test.pt")
        if not data_path.exists():
            dset = torchvision.datasets.CIFAR10(path, download=True, train=train)
            images = torch.tensor(dset.data)
            labels = torch.tensor(dset.targets)
            torch.save({"images": images, "labels": labels, "classes": dset.classes}, data_path)

        data = safe_torch_load(data_path, map_location=torch.device("cuda"))
        self.images, self.labels, self.classes = data["images"], data["labels"], data["classes"]
        self.images = (self.images.half() / 255).permute(0, 3, 1, 2).to(memory_format=torch.channels_last)

        self.normalize = T.Normalize(CIFAR_MEAN, CIFAR_STD)
        self.proc_images = {}
        self.epoch = 0
        self.aug = aug or {}
        for key in self.aug:
            if key not in {"flip", "translate", "cutout"}:
                raise ValueError(f"Unrecognized augmentation key: {key}")
        self.batch_size = batch_size
        self.drop_last = train
        self.shuffle = train

    def __len__(self):
        if self.drop_last:
            return len(self.images) // self.batch_size
        return ceil(len(self.images) / self.batch_size)

    def __iter__(self):
        if self.epoch == 0:
            images = self.proc_images["norm"] = self.normalize(self.images)
            if self.aug.get("flip", False):
                images = self.proc_images["flip"] = batch_flip_lr(images)
            pad = self.aug.get("translate", 0)
            if pad > 0:
                self.proc_images["pad"] = F.pad(images, (pad,) * 4, "reflect")

        if self.aug.get("translate", 0) > 0:
            images = batch_crop(self.proc_images["pad"], self.images.shape[-2])
        elif self.aug.get("flip", False):
            images = self.proc_images["flip"]
        else:
            images = self.proc_images["norm"]

        if self.aug.get("flip", False) and self.epoch % 2 == 1:
            images = images.flip(-1)

        cutout = int(self.aug.get("cutout", 0) or 0)
        if cutout > 0:
            n, c, h, w = images.shape
            max_y = h - cutout + 1
            max_x = w - cutout + 1
            ys = torch.randint(max_y, (n,), device=images.device)
            xs = torch.randint(max_x, (n,), device=images.device)
            mask = torch.ones((n, 1, h, w), device=images.device, dtype=images.dtype)
            ar = torch.arange(cutout, device=images.device)
            yy = ys[:, None] + ar[None, :]
            xx = xs[:, None] + ar[None, :]
            mask[
                torch.arange(n, device=images.device)[:, None, None],
                0,
                yy[:, :, None],
                xx[:, None, :],
            ] = 0
            images = images * mask

        self.epoch += 1
        indices = (torch.randperm if self.shuffle else torch.arange)(len(images), device=images.device)
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
        x = self.conv1(x)
        x = self.pool(x)
        x = self.norm1(x)
        x = self.activ(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activ(x)
        return x


class CifarNet(nn.Module):
    def __init__(self):
        super().__init__()
        widths = dict(block1=64, block2=256, block3=256)
        whiten_kernel_size = 2
        whiten_width = 2 * 3 * whiten_kernel_size**2
        self.whiten = nn.Conv2d(3, whiten_width, whiten_kernel_size, padding=0, bias=True)
        self.whiten.weight.requires_grad = False
        self.layers = nn.Sequential(
            nn.GELU(),
            ConvGroup(whiten_width, widths["block1"]),
            ConvGroup(widths["block1"], widths["block2"]),
            ConvGroup(widths["block2"], widths["block3"]),
            nn.MaxPool2d(3),
        )
        self.head = nn.Linear(widths["block3"], 10, bias=False)
        for mod in self.modules():
            if isinstance(mod, BatchNorm):
                mod.float()
            else:
                mod.half()

    def reset(self):
        for mod in self.modules():
            if type(mod) in (nn.Conv2d, Conv, BatchNorm, nn.Linear):
                mod.reset_parameters()
        w = self.head.weight.data
        w *= 1 / w.std()

    def init_whiten(self, train_images, eps=5e-4):
        c, (h, w) = train_images.shape[1], self.whiten.weight.shape[2:]
        patches = train_images.unfold(2, h, 1).unfold(3, w, 1).transpose(1, 3).reshape(-1, c, h, w).float()
        patches_flat = patches.view(len(patches), -1)
        est_patch_covariance = (patches_flat.T @ patches_flat) / len(patches_flat)
        eigenvalues, eigenvectors = torch.linalg.eigh(est_patch_covariance, UPLO="U")
        eigenvectors_scaled = eigenvectors.T.reshape(-1, c, h, w) / torch.sqrt(eigenvalues.view(-1, 1, 1, 1) + eps)
        self.whiten.weight.data[:] = torch.cat((eigenvectors_scaled, -eigenvectors_scaled))

    def forward(self, x, whiten_bias_grad=True):
        bias = self.whiten.bias
        x = F.conv2d(x, self.whiten.weight, bias if whiten_bias_grad else bias.detach())
        x = self.layers(x)
        x = x.view(len(x), -1)
        return self.head(x) / x.size(-1)


def infer(model, loader, tta_level=0):
    def infer_basic(inputs, net):
        return net(inputs).clone()

    def infer_mirror(inputs, net):
        flipped = inputs.flip(-1)
        return 0.5 * (net(inputs) + net(flipped))

    def infer_mirror_translate(inputs, net):
        logits = infer_mirror(inputs, net)
        pad = 1
        padded_inputs = F.pad(inputs, (pad,) * 4, "reflect")
        translated = (
            padded_inputs[:, :, 0:32, 0:32],
            padded_inputs[:, :, 0:32, 2:34],
            padded_inputs[:, :, 2:34, 0:32],
            padded_inputs[:, :, 2:34, 2:34],
        )
        logits_translate = torch.stack([infer_mirror(inp, net) for inp in translated], dim=0).mean(0)
        return 0.5 * logits + 0.5 * logits_translate

    model.eval()
    test_images = loader.normalize(loader.images)
    infer_fn = [infer_basic, infer_mirror, infer_mirror_translate][tta_level]
    with torch.no_grad():
        return torch.cat([infer_fn(inputs, model) for inputs in test_images.split(2000)])


def evaluate(model, loader, tta_level=0):
    logits = infer(model, loader, tta_level)
    return (logits.argmax(1) == loader.labels).float().mean().item()


def set_trial_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_single_trial(model, data_dir: str, use_dummy_labels: bool = False) -> dict[str, float]:
    batch_size = 2000
    bias_lr = 0.053
    head_lr = 0.67
    wd = 2e-6 * batch_size
    label_smoothing = 0.18
    ema_decay = 0.995

    test_loader = CifarLoader(data_dir, train=False, batch_size=2000)
    train_loader = CifarLoader(data_dir, train=True, batch_size=batch_size, aug=dict(flip=True, translate=3, cutout=4))
    if use_dummy_labels:
        train_loader.labels = torch.randint(0, 10, size=(len(train_loader.labels),), device=train_loader.labels.device)

    total_train_steps = ceil(8 * len(train_loader))
    whiten_bias_train_steps = ceil(3 * len(train_loader))

    filter_params = [p for p in model.parameters() if len(p.shape) == 4 and p.requires_grad]
    norm_biases = [p for name, p in model.named_parameters() if "norm" in name and p.requires_grad]
    param_configs = [
        dict(params=[model.whiten.bias], lr=bias_lr, weight_decay=wd / bias_lr),
        dict(params=norm_biases, lr=bias_lr, weight_decay=wd / bias_lr),
        dict(params=[model.head.weight], lr=head_lr, weight_decay=wd / head_lr),
    ]
    optimizer1 = torch.optim.SGD(param_configs, momentum=0.85, nesterov=True, fused=True)
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
        ender.record()
        torch.cuda.synchronize()
        nonlocal time_seconds
        time_seconds += 1e-3 * starter.elapsed_time(ender)

    model.reset()
    step = 0
    ema_head_weight = model.head.weight.detach().float().clone()

    start_timer()
    train_images = train_loader.normalize(train_loader.images[:5000])
    model.init_whiten(train_images)
    stop_timer()

    for _epoch in range(ceil(total_train_steps / len(train_loader))):
        start_timer()
        model.train()
        for inputs, labels in train_loader:
            outputs = model(inputs, whiten_bias_grad=(step < whiten_bias_train_steps))
            F.cross_entropy(outputs, labels, label_smoothing=label_smoothing, reduction="sum").backward()
            for group in optimizer1.param_groups[:1]:
                group["lr"] = group["initial_lr"] * (1 - step / whiten_bias_train_steps)
            for group in optimizer1.param_groups[1:] + optimizer2.param_groups:
                group["lr"] = group["initial_lr"] * (1 - step / total_train_steps)
            for opt in optimizers:
                opt.step()
            ema_head_weight.mul_(ema_decay).add_(model.head.weight.detach().float(), alpha=1.0 - ema_decay)
            model.zero_grad(set_to_none=True)
            step += 1
            if step >= total_train_steps:
                break
        stop_timer()

    start_timer()
    saved_head = model.head.weight.data
    model.head.weight.data = ema_head_weight.to(dtype=saved_head.dtype)
    tta_val_acc = evaluate(model, test_loader, tta_level=2)
    model.head.weight.data = saved_head
    stop_timer()

    return {
        "tta_val_accuracy": float(tta_val_acc),
        "time_seconds": float(time_seconds),
    }


def run_preflight(model, data_dir: str) -> dict[str, float]:
    batch_size = 2000
    train_loader = CifarLoader(data_dir, train=True, batch_size=batch_size, aug=dict(flip=True, translate=3, cutout=4))
    test_loader = CifarLoader(data_dir, train=False, batch_size=batch_size)

    model.reset()
    train_images = train_loader.normalize(train_loader.images[:5000])
    model.init_whiten(train_images)

    model.train()
    inputs, labels = next(iter(train_loader))
    outputs = model(inputs, whiten_bias_grad=True)
    loss = F.cross_entropy(outputs, labels, label_smoothing=0.18, reduction="mean")
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
    parser.add_argument("--data-dir", type=str, default="cifar10")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--warmup-trials", type=int, default=1)
    parser.add_argument("--target-accuracy", type=float, default=0.94)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--disable-compile", action="store_true")
    parser.add_argument("--json-only", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    target_accuracy = args.target_accuracy / 100.0 if args.target_accuracy > 1.0 else args.target_accuracy
    verbose = bool(args.verbose and not args.json_only)

    if args.trials < 1:
        raise ValueError("--trials must be >= 1")
    if args.warmup_trials < 0:
        raise ValueError("--warmup-trials must be >= 0")
    if not (0.0 <= target_accuracy <= 1.0):
        raise ValueError("--target-accuracy must be in [0, 1] or [0, 100]")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    result = None
    exit_code = 0

    try:
        set_trial_seed(args.seed)
        model = CifarNet().cuda().to(memory_format=torch.channels_last)
        if not args.disable_compile:
            model.compile(mode="max-autotune")

        if verbose:
            print(
                f"[config] data_dir={args.data_dir} trials={args.trials} "
                f"warmup_trials={args.warmup_trials} target_accuracy={target_accuracy:.4f}"
            )

        preflight_result = None
        if args.preflight:
            set_trial_seed(args.seed)
            preflight_result = run_preflight(model, args.data_dir)
            if verbose:
                print(
                    "[preflight] ok "
                    f"train_batch_size={preflight_result['train_batch_size']:.0f} "
                    f"eval_batch_size={preflight_result['eval_batch_size']:.0f} "
                    f"loss={preflight_result['loss']:.4f}"
                )

        for warmup_idx in range(args.warmup_trials):
            warmup_seed = args.seed + warmup_idx
            set_trial_seed(warmup_seed)
            warmup_result = run_single_trial(model, args.data_dir, use_dummy_labels=True)
            if verbose:
                print(
                    f"[warmup {warmup_idx + 1}/{args.warmup_trials}] "
                    f"time_seconds={warmup_result['time_seconds']:.4f}"
                )

        per_trial: list[dict[str, float]] = []
        for trial_idx in range(args.trials):
            trial_seed = args.seed + args.warmup_trials + trial_idx
            set_trial_seed(trial_seed)
            trial_result = run_single_trial(model, args.data_dir, use_dummy_labels=False)
            per_trial.append(trial_result)
            if verbose:
                print(
                    f"[trial {trial_idx + 1}/{args.trials}] "
                    f"tta_val_accuracy={trial_result['tta_val_accuracy']:.4f} "
                    f"time_seconds={trial_result['time_seconds']:.4f}"
                )

        accuracies = torch.tensor([row["tta_val_accuracy"] for row in per_trial], dtype=torch.float64)
        times = torch.tensor([row["time_seconds"] for row in per_trial], dtype=torch.float64)
        mean_accuracy = float(accuracies.mean().item())

        result = {
            "mean_accuracy": mean_accuracy,
            "std_accuracy": float(accuracies.std(unbiased=False).item()),
            "mean_time_seconds": float(times.mean().item()),
            "std_time_seconds": float(times.std(unbiased=False).item()),
            "meets_target": bool(mean_accuracy >= target_accuracy),
            "target_accuracy": float(target_accuracy),
            "trials": int(args.trials),
            "warmup_trials": int(args.warmup_trials),
            "per_trial": per_trial,
            "torch_version": torch.__version__,
            "device_name": torch.cuda.get_device_name(0),
            "compiled": bool(not args.disable_compile),
        }
        if preflight_result is not None:
            result["preflight"] = preflight_result

    except Exception as exc:
        exit_code = 1
        result = {
            "mean_accuracy": None,
            "std_accuracy": None,
            "mean_time_seconds": None,
            "std_time_seconds": None,
            "meets_target": False,
            "target_accuracy": float(target_accuracy),
            "trials": int(args.trials),
            "warmup_trials": int(args.warmup_trials),
            "per_trial": [],
            "torch_version": torch.__version__,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "compiled": bool(not args.disable_compile),
            "failure_type": exc.__class__.__name__,
            "failure_message": str(exc),
        }
        if verbose:
            print(f"[error] {exc.__class__.__name__}: {exc}")

    print(json.dumps(result, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
