#!/usr/bin/env python3
"""Fast CIFAR-10 trainer tuned for single A100 with AirBench-style model/training.

Outputs final line as JSON with mean_accuracy, mean_time_seconds, trials.
"""

from __future__ import annotations

import argparse
import json
from math import ceil
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision
from torch import nn


torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


@torch.compile(mode="max-autotune")
def zeropower_via_newtonschulz5(G, steps=3, eps=1e-7):
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X = X / (X.norm() + eps)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + (B @ X)
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
                p.mul_(len(p) ** 0.5 / (p.norm() + 1e-8))
                update = zeropower_via_newtonschulz5(g_eff.reshape(len(g_eff), -1)).view_as(g_eff)
                p.add_(update, alpha=-lr)


CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def _ensure_cache(path: str, train: bool):
    data_path = Path(path) / ("train.pt" if train else "test.pt")
    if data_path.exists():
        return data_path
    dset = torchvision.datasets.CIFAR10(path, download=True, train=train)
    images = torch.tensor(dset.data)
    labels = torch.tensor(dset.targets)
    torch.save({"images": images, "labels": labels, "classes": dset.classes}, data_path)
    return data_path


class CifarLoader:
    def __init__(self, path, train=True, batch_size=500, aug=None):
        data_path = _ensure_cache(path, train)
        data = torch.load(data_path, map_location="cuda", weights_only=False)
        self.images = data["images"]
        self.labels = data["labels"]
        self.classes = data["classes"]
        self.images = (self.images.cuda(non_blocking=True).half() / 255.0).permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)
        self.labels = self.labels.cuda(non_blocking=True)
        mean = torch.tensor(CIFAR_MEAN, device="cuda", dtype=torch.float16).view(1, 3, 1, 1)
        std = torch.tensor(CIFAR_STD, device="cuda", dtype=torch.float16).view(1, 3, 1, 1)
        self.mean = mean
        self.std = std
        self.proc_images = {}
        self.epoch = 0
        self.aug = aug or {}
        self.batch_size = batch_size
        self.drop_last = train
        self.shuffle = train

    def normalize(self, x):
        return (x - self.mean) / self.std

    def __len__(self):
        return len(self.images) // self.batch_size if self.drop_last else ceil(len(self.images) / self.batch_size)

    def __iter__(self):
        if self.epoch == 0:
            images = self.proc_images["norm"] = self.normalize(self.images)
            if self.aug.get("flip", False):
                flip_mask = (torch.rand(len(images), device=images.device) < 0.5).view(-1, 1, 1, 1)
                images = self.proc_images["flip"] = torch.where(flip_mask, images.flip(-1), images)
            pad = self.aug.get("translate", 0)
            if pad > 0:
                self.proc_images["pad"] = F.pad(images, (pad, pad, pad, pad), mode="reflect")

        if self.aug.get("translate", 0) > 0:
            padded = self.proc_images["pad"]
            pad = self.aug["translate"]
            crop = self.images.shape[-1]
            shifts = torch.randint(-pad, pad + 1, (len(padded), 2), device=padded.device)
            out = torch.empty((len(padded), 3, crop, crop), device=padded.device, dtype=padded.dtype).contiguous(memory_format=torch.channels_last)
            for sy in range(-pad, pad + 1):
                ym = shifts[:, 0] == sy
                if ym.any():
                    tmp = padded[ym, :, pad + sy:pad + sy + crop, :]
                    xs = shifts[ym, 1]
                    o = out[ym]
                    for sx in range(-pad, pad + 1):
                        xm = xs == sx
                        if xm.any():
                            o[xm] = tmp[xm, :, :, pad + sx:pad + sx + crop]
                    out[ym] = o
            images = out
        elif self.aug.get("flip", False):
            images = self.proc_images["flip"]
        else:
            images = self.proc_images["norm"]

        if self.aug.get("flip", False) and self.epoch % 2 == 1:
            images = images.flip(-1)

        self.epoch += 1
        indices = torch.randperm(len(images), device=images.device) if self.shuffle else torch.arange(len(images), device=images.device)
        for i in range(len(self)):
            idx = indices[i * self.batch_size:(i + 1) * self.batch_size]
            yield images[idx], self.labels[idx]


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
    def __init__(self, cin, cout):
        super().__init__()
        self.conv1 = Conv(cin, cout)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(cout)
        self.conv2 = Conv(cout, cout)
        self.norm2 = BatchNorm(cout)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act(x)
        return x


class CifarNet(nn.Module):
    def __init__(self, widths=(64, 256, 256)):
        super().__init__()
        k = 2
        ww = 2 * 3 * k * k
        self.whiten = nn.Conv2d(3, ww, k, padding=0, bias=True)
        self.whiten.weight.requires_grad = False
        self.layers = nn.Sequential(
            nn.GELU(),
            ConvGroup(ww, widths[0]),
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
    def init_whiten(self, train_images, eps=5e-4):
        c, (h, w) = train_images.shape[1], self.whiten.weight.shape[2:]
        patches = train_images.unfold(2, h, 1).unfold(3, w, 1).transpose(1, 3).reshape(-1, c, h, w).float()
        flat = patches.reshape(len(patches), -1)
        cov = (flat.T @ flat) / len(flat)
        evals, evecs = torch.linalg.eigh(cov, UPLO="U")
        evecs_scaled = evecs.T.reshape(-1, c, h, w) / torch.sqrt(evals.view(-1, 1, 1, 1) + eps)
        self.whiten.weight.data[:] = torch.cat((evecs_scaled, -evecs_scaled)).to(self.whiten.weight.dtype)
        if self.whiten.bias is not None:
            self.whiten.bias.zero_()

    def forward(self, x, whiten_bias_grad=True):
        b = self.whiten.bias
        x = F.conv2d(x, self.whiten.weight, b if whiten_bias_grad else b.detach())
        x = self.layers(x)
        x = x.reshape(len(x), -1)
        return self.head(x) / x.size(-1)


def infer(model, loader, tta_level=2):
    @torch.no_grad()
    def mirror(inp):
        return 0.5 * model(inp, whiten_bias_grad=False) + 0.5 * model(inp.flip(-1), whiten_bias_grad=False)

    model.eval()
    test_images = loader.normalize(loader.images)
    outs = []
    for inputs in test_images.split(2000):
        if tta_level <= 0:
            logits = model(inputs, whiten_bias_grad=False)
        elif tta_level == 1:
            logits = mirror(inputs)
        else:
            logits = mirror(inputs)
            padded = F.pad(inputs, (1, 1, 1, 1), mode="reflect")
            t1 = mirror(padded[:, :, 0:32, 0:32])
            t2 = mirror(padded[:, :, 2:34, 2:34])
            logits = 0.5 * logits + 0.25 * t1 + 0.25 * t2
        outs.append(logits)
    return torch.cat(outs, dim=0)


def evaluate(model, loader, tta_level=2):
    logits = infer(model, loader, tta_level)
    return (logits.argmax(1) == loader.labels).float().mean().item()


def set_trial_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def maybe_compile_model(model, enable=True):
    if not enable:
        return model
    try:
        return torch.compile(model, mode="max-autotune")
    except Exception:
        return model


def run_single_trial(model, data_dir: str, use_dummy_labels: bool = False):
    batch_size = 2000
    bias_lr = 0.053
    head_lr = 0.70
    wd = 2e-6 * batch_size

    test_loader = CifarLoader(data_dir, train=False, batch_size=2000)
    train_loader = CifarLoader(data_dir, train=True, batch_size=batch_size, aug={"flip": True, "translate": 2})
    if use_dummy_labels:
        train_loader.labels = torch.randint(0, 10, (len(train_loader.labels),), device=train_loader.labels.device)

    total_train_steps = ceil(8.5 * len(train_loader))
    whiten_bias_train_steps = ceil(3.0 * len(train_loader))

    filter_params = [p for p in model.parameters() if p.requires_grad and p.ndim == 4]
    norm_biases = [p for n, p in model.named_parameters() if "norm" in n and p.requires_grad]
    param_configs = [
        dict(params=[model.whiten.bias], lr=bias_lr, weight_decay=wd / bias_lr),
        dict(params=norm_biases, lr=bias_lr, weight_decay=wd / bias_lr),
        dict(params=[model.head.weight], lr=head_lr, weight_decay=wd / head_lr),
    ]
    optimizer1 = torch.optim.SGD(param_configs, momentum=0.85, nesterov=True, fused=True)
    optimizer2 = Muon(filter_params, lr=0.24, momentum=0.6, nesterov=True)
    for opt in (optimizer1, optimizer2):
        for g in opt.param_groups:
            g["initial_lr"] = g["lr"]

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
    train_images = train_loader.normalize(train_loader.images[:5000])
    model.init_whiten(train_images)
    stop_timer()

    for _ in range(ceil(total_train_steps / len(train_loader))):
        start_timer()
        model.train()
        for inputs, labels in train_loader:
            outputs = model(inputs, whiten_bias_grad=(step < whiten_bias_train_steps))
            F.cross_entropy(outputs, labels, label_smoothing=0.15, reduction="sum").backward()

            wb_scale = max(0.0, 1.0 - step / max(1, whiten_bias_train_steps))
            main_scale = max(0.0, 1.0 - step / max(1, total_train_steps))
            optimizer1.param_groups[0]["lr"] = optimizer1.param_groups[0]["initial_lr"] * wb_scale
            for g in optimizer1.param_groups[1:]:
                g["lr"] = g["initial_lr"] * main_scale
            for g in optimizer2.param_groups:
                g["lr"] = g["initial_lr"] * main_scale

            optimizer1.step()
            optimizer2.step()
            model.zero_grad(set_to_none=True)
            step += 1
            if step >= total_train_steps:
                break
        stop_timer()
        if step >= total_train_steps:
            break

    start_timer()
    acc = evaluate(model, test_loader, tta_level=2)
    stop_timer()
    return {"tta_val_accuracy": float(acc), "time_seconds": float(time_seconds)}


def run_preflight(model, data_dir: str):
    train_loader = CifarLoader(data_dir, train=True, batch_size=512, aug={"flip": True, "translate": 2})
    test_loader = CifarLoader(data_dir, train=False, batch_size=512)
    model.reset()
    model.init_whiten(train_loader.normalize(train_loader.images[:2048]))
    model.train()
    x, y = next(iter(train_loader))
    out = model(x, whiten_bias_grad=True)
    loss = F.cross_entropy(out, y, label_smoothing=0.15)
    loss.backward()
    model.zero_grad(set_to_none=True)
    model.eval()
    with torch.no_grad():
        _ = model(test_loader.normalize(test_loader.images[:512]), whiten_bias_grad=False)
    torch.cuda.synchronize()
    return {"train_batch_size": float(len(x)), "eval_batch_size": 512.0, "loss": float(loss.item())}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
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

    base_model = CifarNet().cuda().to(memory_format=torch.channels_last)
    model = maybe_compile_model(base_model, enable=not args.disable_compile)

    if args.verbose:
        print(f"[config] data_dir={args.data_dir} trials={args.trials} warmup_trials={args.warmup_trials} target_accuracy={target_accuracy:.4f}")

    if args.preflight:
        set_trial_seed(args.seed)
        pf = run_preflight(base_model, args.data_dir)
        if args.verbose:
            print(f"[preflight] ok train_batch_size={pf['train_batch_size']:.0f} eval_batch_size={pf['eval_batch_size']:.0f} loss={pf['loss']:.4f}")

    for i in range(args.warmup_trials):
        set_trial_seed(args.seed + i)
        warm = run_single_trial(base_model, args.data_dir, use_dummy_labels=True)
        if args.verbose:
            print(f"[warmup {i + 1}/{args.warmup_trials}] time_seconds={warm['time_seconds']:.4f}")

    # compile after warmup so measured trial excludes compilation overhead as much as possible
    model = maybe_compile_model(base_model, enable=not args.disable_compile)

    per_trial = []
    for i in range(args.trials):
        set_trial_seed(args.seed + args.warmup_trials + i)
        res = run_single_trial(base_model, args.data_dir, use_dummy_labels=False)
        per_trial.append(res)
        if args.verbose:
            print(f"[trial {i + 1}/{args.trials}] tta_val_accuracy={res['tta_val_accuracy']:.4f} time_seconds={res['time_seconds']:.4f}")

    accuracies = torch.tensor([r["tta_val_accuracy"] for r in per_trial], dtype=torch.float64)
    times = torch.tensor([r["time_seconds"] for r in per_trial], dtype=torch.float64)
    result = {
        "mean_accuracy": float(accuracies.mean().item()),
        "std_accuracy": float(accuracies.std(unbiased=False).item()) if len(per_trial) else 0.0,
        "mean_time_seconds": float(times.mean().item()),
        "std_time_seconds": float(times.std(unbiased=False).item()) if len(per_trial) else 0.0,
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
