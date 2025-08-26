import argparse
import csv
import hashlib
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .data import Windows, complementary_mask, fit_scaler, load_entity, scale
from .metrics import ensemble, evaluate
from .model import MODEM

ROOT = Path(__file__).resolve().parents[1]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def validate_config(cfg):
    m, d, t, inf = (cfg[k] for k in ("model", "data", "train", "inference"))
    for key in ("channels", "prior_channels"):
        if m[key] < 4 or m[key] % 2 or m[key] % m["heads"]:
            raise ValueError(f"{key} must be even and divisible by heads")
    if not 0 < m["beta_start"] <= m["beta_end"] < 1:
        raise ValueError("Require 0 < beta_start <= beta_end < 1")
    if m["resolutions"] < 1 or 2 ** (m["resolutions"] - 1) > d["window"]:
        raise ValueError("Resolution pooling size must fit the window")
    if m["kernel_size"] % 2 != 1 or m["kernel_size"] < 1:
        raise ValueError("kernel_size must be positive and odd")
    if not 0 < m["frequency_fraction"] <= 1:
        raise ValueError("frequency_fraction must lie in (0, 1]")
    if not 0 < m["hop_length"] <= m["n_fft"] // 2:
        raise ValueError("Require 0 < hop_length <= n_fft / 2 for STFT overlap-add")
    if not 0 < d["validation_fraction"] < 1:
        raise ValueError("validation_fraction must lie in (0, 1)")
    if not 1 <= inf["sampling_steps"] <= m["diffusion_steps"]:
        raise ValueError("Invalid DDIM sampling_steps")
    votes = m["resolutions"] * min(inf["keep_steps"], inf["sampling_steps"])
    if not 0 <= inf["vote_threshold"] < votes:
        raise ValueError(f"vote_threshold must be below the available {votes} votes")
    if not 1 <= inf["mask_block"] < d["window"]:
        raise ValueError("Invalid mask_block")
    if not 0 <= inf["eta"] <= 1 or inf["samples"] < 1 or inf["keep_steps"] < 1:
        raise ValueError("Invalid inference settings")
    if t["epochs"] < 1 or t["batch_size"] < 1:
        raise ValueError("epochs and batch_size must be positive")


def train(cfg, raw_train, output, device, resume=False):
    seed_everything(cfg["seed"])
    d, settings = cfg["data"], cfg["train"]
    split = int(len(raw_train) * (1 - d["validation_fraction"]))
    if min(split, len(raw_train) - split) < d["window"]:
        raise ValueError("Chronological train/validation partitions must each contain at least one window")

    scaler = fit_scaler(raw_train[:split])
    training = Windows(scale(raw_train[:split], scaler), d["window"], d["train_stride"])
    validation = Windows(scale(raw_train[split:], scaler), d["window"], d["test_stride"], cover_tail=True)
    train_loader = DataLoader(training, settings["batch_size"], shuffle=True, num_workers=settings["workers"])
    val_loader = DataLoader(validation, settings["batch_size"], num_workers=settings["workers"])
    model = MODEM(raw_train.shape[1], cfg["model"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"])
    best, start_epoch, history = float("inf"), 0, []
    checkpoint_path = output / "last.pt"
    if resume:
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        old_config = json.loads(json.dumps(saved["config"]))
        new_config = json.loads(json.dumps(cfg))
        old_config["train"].pop("epochs")
        new_config["train"].pop("epochs")
        if old_config != new_config or saved["scaler"] != scaler:
            raise ValueError("Resume requires the same data/config; only total epochs may change")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
        start_epoch, best, history = saved["epoch"], saved["best_validation_loss"], saved["history"]
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(saved["cuda_rng"], device)
    elif checkpoint_path.exists() or (output / "best.pt").exists():
        raise FileExistsError(f"{output} already has checkpoints; use --resume or a new --output")
    write_json(output / "scaler.json", scaler)
    started = time.monotonic()
    for epoch in range(start_epoch, settings["epochs"]):
        model.train()
        total, count = 0.0, 0
        epoch_start = time.monotonic()
        for batch_index, (x, _, _) in enumerate(train_loader):
            x = x.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(x)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch + 1}, batch {batch_index}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            total += loss.item() * len(x)
            count += len(x)
            if (batch_index + 1) % 25 == 0:
                print(f"epoch={epoch + 1} batch={batch_index + 1}/{len(train_loader)} loss={total / count:.6f}", flush=True)
        model.eval()
        val_total, val_count = 0.0, 0

        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=cuda_devices), torch.no_grad():
            torch.manual_seed(cfg["seed"] + 10000)
            for x, _, _ in val_loader:
                loss = model.loss(x.to(device))
                val_total += loss.item() * len(x)
                val_count += len(x)
        value = val_total / val_count
        if not np.isfinite(value):
            raise FloatingPointError("Non-finite validation loss")
        row = {"epoch": epoch + 1, "train_loss": total / count, "validation_loss": value,
               "training_examples": count, "seconds": time.monotonic() - epoch_start}
        history.append(row)
        improved = value < best
        best = min(best, value)
        checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "config": cfg, "features": raw_train.shape[1], "scaler": scaler,
                      "epoch": epoch + 1, "best_validation_loss": best, "history": history,
                      "torch_rng": torch.get_rng_state(),
                      "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}
        torch.save(checkpoint, checkpoint_path)
        if improved:
            torch.save(checkpoint, output / "best.pt")
        write_json(output / "history.json", history)
        print(json.dumps(row), flush=True)
    with (output / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    return {"seconds": time.monotonic() - started, "epochs_completed": len(history),
            "training_windows": len(training), "validation_windows": len(validation),
            "parameters": sum(p.numel() for p in model.parameters()), "best_validation_loss": best}


@torch.no_grad()
def infer(output, raw_test, labels, device, batch_size=None):

    saved = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    cfg = saved["config"]
    seed_everything(cfg["seed"] + 20000)
    model = MODEM(saved["features"], cfg["model"]).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    x = scale(raw_test, saved["scaler"])
    d, inf = cfg["data"], cfg["inference"]
    windows = Windows(x, d["window"], d["test_stride"], cover_tail=True)
    loader = DataLoader(windows, batch_size or cfg["train"]["batch_size"])
    n = len(x)
    keep = min(inf["sampling_steps"], inf["keep_steps"])
    errors = np.zeros((cfg["model"]["resolutions"], keep, n), np.float32)
    reconstruction = np.zeros_like(x)
    counts = np.zeros(n, np.int32)
    started = time.monotonic()
    for batch_index, (batch, starts, valid) in enumerate(loader):
        batch = batch.to(device)
        mask = complementary_mask(batch, inf["mask_block"])
        batch_errors = torch.zeros(len(batch), *errors.shape[:2], d["window"], device=device)
        batch_reconstruction = torch.zeros_like(batch)
        for _ in range(inf["samples"]):
            for cond_mask in (mask, 1 - mask):
                rec, err = model.reconstruct(batch, cond_mask, inf["sampling_steps"], inf["keep_steps"], inf["eta"])
                target = 1 - cond_mask
                batch_reconstruction += rec * target / inf["samples"]
                batch_errors += err * target[:, :1, None, :] / inf["samples"]
        batch_errors = batch_errors.cpu().numpy()
        batch_reconstruction = batch_reconstruction.transpose(1, 2).cpu().numpy()
        if not np.isfinite(batch_errors).all() or not np.isfinite(batch_reconstruction).all():
            raise FloatingPointError("Non-finite inference result")
        for i, (start, length) in enumerate(zip(starts.tolist(), valid.tolist())):
            errors[:, :, start:start + length] += batch_errors[i, :, :, :length]
            reconstruction[start:start + length] += batch_reconstruction[i, :length]
            counts[start:start + length] += 1
        if (batch_index + 1) % 5 == 0 or batch_index + 1 == len(loader):
            print(f"inference={batch_index + 1}/{len(loader)} elapsed={time.monotonic() - started:.1f}s", flush=True)
    if not np.all(counts > 0):
        raise RuntimeError("Uncovered test timestamps")
    errors /= counts[None, None]
    reconstruction /= counts[:, None]
    np.savez_compressed(output / "scores.npz", errors=errors, reconstruction=reconstruction,
                        normalized_input=x, labels=labels, coverage=counts)
    return {"seconds": time.monotonic() - started, "test_points": n, "test_windows": len(windows),
            "error_shape": list(errors.shape), "checkpoint_epoch": saved["epoch"],
            "all_points_covered": bool(np.all(counts > 0)), "seed": cfg["seed"] + 20000}


def score(output):
    saved = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    cfg = saved["config"]
    with np.load(output / "scores.npz") as arrays:
        errors, labels = arrays["errors"], arrays["labels"]
    prediction, votes, thresholds, fractions = ensemble(
        errors, cfg["inference"]["final_fraction"], cfg["inference"]["vote_threshold"])
    metrics = evaluate(prediction, labels)
    metrics.update({"test_points": len(labels), "positive_labels": int(labels.sum()),
                    "predicted_positive": int(prediction.sum()),
                    "threshold_selection": "unlabeled_test_error_percentiles_fixed_paper_fraction",
                    "vote_threshold": cfg["inference"]["vote_threshold"],
                    "final_fraction": cfg["inference"]["final_fraction"],
                    "available_votes": int(np.prod(errors.shape[:2]))})
    np.savez_compressed(output / "predictions.npz", prediction=prediction, votes=votes,
                        thresholds=thresholds, fractions=fractions)
    write_json(output / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2), flush=True)
    plot_results(output)
    return metrics


def plot_results(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .metrics import spans
    with np.load(output / "scores.npz") as data:
        observed, reconstructed, labels = data["normalized_input"], data["reconstruction"], data["labels"]
        error = data["errors"][0, -1]
    with np.load(output / "predictions.npz") as data:
        votes, prediction = data["votes"], data["prediction"]
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(observed[:, 0], label="Observed (feature 0)", lw=0.7)
    axes[0].plot(reconstructed[:, 0], label="Reconstructed", lw=0.7)
    axes[0].legend(loc="upper right")
    axes[1].plot(error, label="Finest resolution final error", lw=0.7)
    axes[1].legend(loc="upper right")
    axes[2].plot(votes, label="Anomaly votes", lw=0.7)
    axes[2].scatter(np.flatnonzero(prediction), votes[prediction.astype(bool)], s=3, color="red", label="Prediction")
    axes[2].legend(loc="upper right")
    for begin, end in spans(labels):
        for ax in axes:
            ax.axvspan(begin, end, color="orange", alpha=0.2)
    axes[2].set_xlabel("Test timestamp (orange: labeled anomaly)")
    fig.tight_layout()
    fig.savefig(output / "diagnostics.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["all", "train", "infer", "evaluate"], default="all")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/paper.yaml")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/Machine")
    parser.add_argument("--entity", default="machine-1-1")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/smd_machine-1-1_seed1")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", action="store_true", help="Resume last.pt up to the configured total epochs")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.stage == "evaluate":
        score(output)
        return
    if args.stage in ("all", "train"):
        cfg = yaml.safe_load(args.config.read_text())
        if args.epochs is not None:
            cfg["train"]["epochs"] = args.epochs
        if args.batch_size is not None:
            cfg["train"]["batch_size"] = args.batch_size
        if args.seed is not None:
            cfg["seed"] = args.seed
        validate_config(cfg)
    else:
        cfg = torch.load(output / "best.pt", map_location="cpu", weights_only=False)["config"]
    raw_train, raw_test, labels = load_entity(args.data_root, args.entity)
    manifest_path = output / "manifest.json"
    if args.stage == "infer" and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["entity"] != args.entity:
            raise ValueError("Inference entity differs from training manifest")
    else:
        manifest = {"entity": args.entity, "data_root": str(args.data_root.resolve()),
                    "config": cfg, "command_args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    "environment": {"python": platform.python_version(), "torch": torch.__version__,
                                    "numpy": np.__version__, "cuda": torch.version.cuda,
                                    "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
                    "data_shapes": {"train": list(raw_train.shape), "test": list(raw_test.shape)},
                    "data_sha256": {"train": hashlib.sha256(raw_train.tobytes()).hexdigest(),
                                    "test": hashlib.sha256(raw_test.tobytes()).hexdigest()},
                    "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in sorted((ROOT / "modem").glob("*.py"))},
                    "status": "running"}
    if args.stage in ("all", "train"):
        manifest["training"] = train(cfg, raw_train, output, device, args.resume)
        (output / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        write_json(manifest_path, manifest)
    if args.stage in ("all", "infer"):
        manifest["inference"] = infer(output, raw_test, labels, device, args.batch_size)
        manifest["metrics"] = score(output)
    manifest["status"] = "complete"
    write_json(manifest_path, manifest)
    print(f"Completed {args.stage}: {output}", flush=True)


if __name__ == "__main__":
    main()
