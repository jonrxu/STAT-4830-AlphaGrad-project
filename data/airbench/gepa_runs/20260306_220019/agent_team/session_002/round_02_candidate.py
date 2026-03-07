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
torch.backends.cudnn.deterministic = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


def _safe_torch_load(path, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _maybe_compile(fn=None, **compile_kwargs):
    if fn is None:
        return lambda f: _maybe_compile(f, **compile_kwargs)
    if hasattr(torch, "compile"):
        try:
            return torch.compile(fn, **compile_kwargs)
        except Exception:
            return fn
    return fn


def _safe_compile_model(model, **compile_kwargs):
    if not hasattr(torch, "compile"):
        return model, False, None
    try:
        compiled_model = torch.compile(model, **compile_kwargs)
        return compiled_model, True, None
    except Exception as exc:
        return model, False, f"compile_failed: {exc}"


@_maybe_compile(fullgraph=False, dynamic=False)
def zeropower_via_newtonschulz5(G, steps=3, eps=1e-7):
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    orig_dtype = G.dtype
    X = G.to(dtype=torch.bfloat16)
    X = X / (X.norm() + eps)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
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
            lr = float(group["lr"])
            momentum = float(group["momentum"])
            nesterov = bool(group["nesterov"])
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                if g.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients")

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g, memory_format=torch.preserve_format)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                g_eff = g.add(buf, alpha=momentum) if nesterov else buf

                p_norm = p.norm()
                if torch.isfinite(p_norm) and p_norm > 0:
                    p.mul_((p.numel() ** 0.5) / p_norm)

                update = zeropower_via_newtonschulz5(g_eff.reshape(g_eff.shape[0], -1)).view_as(g_eff)
                update = update.to(device=p.device, dtype=p.dtype)
                p.add_(update, alpha=-lr)


CIFAR_MEAN = torch.tensor((0.4914, 0.4822, 0.4465), dtype=torch.float32)
CIFAR_STD = torch.tensor((0.2470, 0.2435, 0.2616), dtype=torch.float32)


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

        data = _safe_torch_load(data_path, map_location=torch.device("cuda"))
        self.images, self.labels, self.classes = data["images"], data["labels"], data["classes"]
        self.images = (self.images.half() / 255).permute(0, 3, 1, 2).to(memory_format=torch.channels_last)

        self.normalize = T.Normalize(CIFAR_MEAN, CIFAR_STD)
        self.proc_images = {}
        self.epoch = 0
        self.aug = aug or {}
        for key in self.aug:
            if key not in {"flip", "translate"}:
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


def _model_device_dtype(model):
    p = next(model.parameters())
    return p.device, p.dtype


def _align_inputs(inputs, model):
    device, model_dtype = _model_device_dtype(model)
    if inputs.device != device:
        inputs = inputs.to(device=device, non_blocking=True)
    target_dtype = torch.float32 if model_dtype in (torch.float16, torch.bfloat16) else model_dtype
    if inputs.dtype != target_dtype:
        inputs = inputs.to(dtype=target_dtype)
    return inputs.contiguous(memory_format=torch.channels_last)


def infer(model, loader, tta_level=0):
    def infer_basic(inputs, net):
        return net(inputs)

    def infer_mirror(inputs, net):
        return 0.5 * (net(inputs) + net(inputs.flip(-1)))

    def infer_mirror_translate(inputs, net):
        logits = infer_mirror(inputs, net)
        pad = 1
        padded_inputs = F.pad(inputs, (pad,) * 4, "reflect")
        translated = [
            padded_inputs[:, :, 0:32, 0:32],
            padded_inputs[:, :, 0:32, 2:34],
            padded_inputs[:, :, 2:34, 0:32],
            padded_inputs[:, :, 2:34, 2:34],
        ]
        logits_translate = torch.stack([infer_mirror(inp, net) for inp in translated]).mean(0)
        return 0.5 * logits + 0.5 * logits_translate

    def infer_mirror_translate_center(inputs, net):
        logits = infer_mirror(inputs, net)
        pad = 1
        padded_inputs = F.pad(inputs, (pad,) * 4, "reflect")
        translated = [
            padded_inputs[:, :, 0:32, 0:32],
            padded_inputs[:, :, 0:32, 1:33],
            padded_inputs[:, :, 0:32, 2:34],
            padded_inputs[:, :, 1:33, 0:32],
            padded_inputs[:, :, 1:33, 1:33],
            padded_inputs[:, :, 1:33, 2:34],
            padded_inputs[:, :, 2:34, 0:32],
            padded_inputs[:, :, 2:34, 1:33],
            padded_inputs[:, :, 2:34, 2:34],
        ]
        logits_translate = torch.stack([infer_mirror(inp, net) for inp in translated]).mean(0)
        return 0.35 * logits + 0.65 * logits_translate

    infer_fns = [infer_basic, infer_mirror, infer_mirror_translate, infer_mirror_translate_center]
    if not (0 <= int(tta_level) < len(infer_fns)):
        raise ValueError(f"Invalid tta_level={tta_level}")

    model.eval()
    infer_fn = infer_fns[int(tta_level)]
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
        test_images = loader.normalize(loader.images).to(dtype=torch.float32)
        test_images = _align_inputs(test_images, model)
        return torch.cat([infer_fn(inputs, model) for inputs in test_images.split(2000)], dim=0)


def evaluate(model, loader, tta_level=0):
    logits = infer(model, loader, tta_level)
    labels = loader.labels
    if labels.device != logits.device:
        labels = labels.to(logits.device, non_blocking=True)
    return (logits.argmax(1) == labels).float().mean().item()


def set_trial_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_single_trial(model, data_dir: str, use_dummy_labels: bool = False) -> dict[str, float]:
    batch_size = 2000
    bias_lr = 0.053
    head_lr = 0.70
    wd = 1.6e-6 * batch_size
    label_smoothing = 0.12
    ema_decay = 0.96

    test_loader = CifarLoader(data_dir, train=False, batch_size=2000)
    train_loader = CifarLoader(data_dir, train=True, batch_size=batch_size, aug=dict(flip=True, translate=2))
    if use_dummy_labels:
        train_loader.labels = torch.randint(0, 10, size=(len(train_loader.labels),), device=train_loader.labels.device)

    total_train_steps = ceil(8.5 * len(train_loader))
    whiten_bias_train_steps = max(1, ceil(3 * len(train_loader)))

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
            group["initial_lr"] = float(group["lr"])

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    ema_buffers = [p.detach().clone() for p in trainable_params]

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
    model.zero_grad(set_to_none=True)
    step = 0

    start_timer()
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
        train_images = train_loader.normalize(train_loader.images[:5000]).to(dtype=torch.float32)
        train_images = _align_inputs(train_images, model)
        model.init_whiten(train_images)
    stop_timer()

    for _epoch in range(ceil(total_train_steps / len(train_loader))):
        start_timer()
        model.train()
        for inputs, labels in train_loader:
            inputs = inputs.to(dtype=torch.float32)
            inputs = _align_inputs(inputs, model)
            if labels.device != inputs.device:
                labels = labels.to(device=inputs.device, non_blocking=True)

            outputs = model(inputs, whiten_bias_grad=(step < whiten_bias_train_steps))
            F.cross_entropy(outputs, labels, label_smoothing=label_smoothing, reduction="sum").backward()

            warm_frac = max(0.0, 1.0 - step / whiten_bias_train_steps)
            total_frac = max(0.0, 1.0 - step / total_train_steps)
            for group in optimizer1.param_groups[:1]:
                group["lr"] = group["initial_lr"] * warm_frac
            for group in optimizer1.param_groups[1:] + optimizer2.param_groups:
                group["lr"] = group["initial_lr"] * total_frac

            for opt in optimizers:
                opt.step()

            with torch.no_grad():
                if step == 0:
                    for ema, p in zip(ema_buffers, trainable_params):
                        ema.copy_(p)
                else:
                    for ema, p in zip(ema_buffers, trainable_params):
                        ema.lerp_(p, 1.0 - ema_decay)

            model.zero_grad(set_to_none=True)
            step += 1
            if step >= total_train_steps:
                break
        stop_timer()
        if step >= total_train_steps:
            break

    saved_params = [p.detach().clone() for p in trainable_params]
    with torch.no_grad():
        for p, ema in zip(trainable_params, ema_buffers):
            p.copy_(ema)

    start_timer()
    tta_val_acc = evaluate(model, test_loader, tta_level=3)
    stop_timer()

    with torch.no_grad():
        for p, saved in zip(trainable_params, saved_params):
            p.copy_(saved)

    return {
        "tta_val_accuracy": float(tta_val_acc),
        "time_seconds": float(time_seconds),
    }


def run_preflight(model, data_dir: str) -> dict[str, float]:
    batch_size = 2000
    train_loader = CifarLoader(data_dir, train=True, batch_size=batch_size, aug=dict(flip=True, translate=2))
    test_loader = CifarLoader(data_dir, train=False, batch_size=batch_size)

    model.reset()
    model.zero_grad(set_to_none=True)

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
        train_images = train_loader.normalize(train_loader.images[:5000]).to(dtype=torch.float32)
        train_images = _align_inputs(train_images, model)
        model.init_whiten(train_images)

    model.train()
    inputs, labels = next(iter(train_loader))
    inputs = inputs.to(dtype=torch.float32)
    inputs = _align_inputs(inputs, model)
    if labels.device != inputs.device:
        labels = labels.to(device=inputs.device, non_blocking=True)
    outputs = model(inputs, whiten_bias_grad=True)
    loss = F.cross_entropy(outputs, labels, label_smoothing=0.12, reduction="mean")
    loss.backward()
    model.zero_grad(set_to_none=True)

    model.eval()
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
        eval_inputs = test_loader.normalize(test_loader.images[:batch_size]).to(dtype=torch.float32)
        eval_inputs = _align_inputs(eval_inputs, model)
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


def _maybe_compile_model(model):
    compiled_model, compiled, _reason = _safe_compile_model(model, mode="max-autotune")
    return compiled_model, compiled


def main() -> int:
    args = parse_args()
    target_accuracy = args.target_accuracy / 100.0 if args.target_accuracy > 1.0 else args.target_accuracy

    exit_code = 0
    per_trial: list[dict[str, float]] = []
    preflight_result = None
    failure_type = None
    failure_message = None
    compiled = False
    compile_downgraded = False
    compile_failure_message = None

    def _log(msg: str) -> None:
        if args.verbose and not args.json_only:
            print(msg, flush=True)

    try:
        base_model = CifarNet().cuda().to(dtype=torch.float32, memory_format=torch.channels_last)
        model = base_model
        if not args.disable_compile:
            model, compiled, compile_failure_message = _safe_compile_model(base_model, mode="max-autotune")

        _log(
            f"[config] data_dir={args.data_dir} trials={args.trials} "
            f"warmup_trials={args.warmup_trials} target_accuracy={target_accuracy:.4f}"
        )

        if args.preflight:
            set_trial_seed(args.seed)
            try:
                preflight_result = run_preflight(model, args.data_dir)
            except Exception as exc:
                if compiled and not args.disable_compile:
                    compile_downgraded = True
                    compile_failure_message = f"compiled_preflight_failed: {exc}"
                    model = base_model
                    compiled = False
                    set_trial_seed(args.seed)
                    preflight_result = run_preflight(model, args.data_dir)
                else:
                    raise
            _log(
                "[preflight] ok "
                f"train_batch_size={preflight_result['train_batch_size']:.0f} "
                f"eval_batch_size={preflight_result['eval_batch_size']:.0f} "
                f"loss={preflight_result['loss']:.4f}"
            )

        for warmup_idx in range(args.warmup_trials):
            set_trial_seed(args.seed + warmup_idx)
            try:
                warmup_result = run_single_trial(model, args.data_dir, use_dummy_labels=True)
            except Exception as exc:
                if compiled and not args.disable_compile:
                    compile_downgraded = True
                    compile_failure_message = f"compiled_warmup_failed: {exc}"
                    model = base_model
                    compiled = False
                    set_trial_seed(args.seed + warmup_idx)
                    warmup_result = run_single_trial(model, args.data_dir, use_dummy_labels=True)
                else:
                    raise
            _log(
                f"[warmup {warmup_idx + 1}/{args.warmup_trials}] "
                f"time_seconds={warmup_result['time_seconds']:.4f}"
            )

        for trial_idx in range(args.trials):
            set_trial_seed(args.seed + args.warmup_trials + trial_idx)
            try:
                trial_result = run_single_trial(model, args.data_dir, use_dummy_labels=False)
            except Exception as exc:
                if compiled and not args.disable_compile:
                    compile_downgraded = True
                    compile_failure_message = f"compiled_trial_failed: {exc}"
                    model = base_model
                    compiled = False
                    set_trial_seed(args.seed + args.warmup_trials + trial_idx)
                    trial_result = run_single_trial(model, args.data_dir, use_dummy_labels=False)
                else:
                    raise
            per_trial.append(trial_result)
            _log(
                f"[trial {trial_idx + 1}/{args.trials}] "
                f"tta_val_accuracy={trial_result['tta_val_accuracy']:.4f} "
                f"time_seconds={trial_result['time_seconds']:.4f}"
            )
    except RuntimeError as exc:
        exit_code = 1
        failure_type = "runtime_error"
        failure_message = str(exc)
    except Exception as exc:
        exit_code = 1
        failure_type = exc.__class__.__name__
        failure_message = str(exc)

    if per_trial:
        accuracies = torch.tensor([row["tta_val_accuracy"] for row in per_trial], dtype=torch.float64)
        times = torch.tensor([row["time_seconds"] for row in per_trial], dtype=torch.float64)
        mean_accuracy = float(accuracies.mean().item())
        std_accuracy = float(accuracies.std(unbiased=False).item())
        mean_time_seconds = float(times.mean().item())
        std_time_seconds = float(times.std(unbiased=False).item())
    else:
        mean_accuracy = 0.0
        std_accuracy = 0.0
        mean_time_seconds = 0.0
        std_time_seconds = 0.0

    try:
        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        device_name = "unknown"

    result = {
        "mean_accuracy": mean_accuracy,
        "std_accuracy": std_accuracy,
        "mean_time_seconds": mean_time_seconds,
        "std_time_seconds": std_time_seconds,
        "meets_target": bool(per_trial and mean_accuracy >= target_accuracy),
        "target_accuracy": float(target_accuracy),
        "trials": int(args.trials),
        "warmup_trials": int(args.warmup_trials),
        "per_trial": per_trial,
        "torch_version": torch.__version__,
        "device_name": device_name,
        "compiled": bool(compiled),
    }
    if preflight_result is not None:
        result["preflight"] = preflight_result
    if compile_downgraded:
        result["compile_downgraded"] = True
    if compile_failure_message is not None:
        result["compile_failure_message"] = compile_failure_message
    if failure_type is not None:
        result["failure_type"] = failure_type
        result["failure_message"] = failure_message

    print(json.dumps(result, sort_keys=True), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
