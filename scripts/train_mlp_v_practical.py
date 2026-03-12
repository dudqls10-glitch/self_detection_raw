#!/usr/bin/env python3
"""
MLP 학습 스크립트 (Velocity 포함, 실용 메트릭/손실 강화 버전)

기존 train_mlp_v.py의 핵심 파이프라인은 유지하면서 아래를 추가합니다.
1. raw-scale 기준 임계값(기본 2000)에 맞춘 robust Huber loss
2. raw-scale RMSE / 채널별 RMSE, MAE / within-threshold 메트릭
3. best checkpoint 기준: within_2000
4. 조기 종료 기준: validation rmse
4. 최종 메트릭 요약 저장 및 간단한 학습 곡선 플롯

실행:
    python3 scripts/train_mlp_v_practical.py
"""

import os
import sys
import random
from pathlib import Path
from datetime import datetime
import argparse

# make sure package root is on sys.path when running script directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting is optional at runtime
    plt = None

from self_detection_raw.data.loader_v import load_multiple_files_v
from self_detection_raw.models.mlp_b_v import ModelBV
from self_detection_raw.train.train import SelfDetectionDataset, train_epoch
from self_detection_raw.utils.io import ensure_dir, save_json
from self_detection_raw.utils.metrics import compute_channel_metrics, format_metrics_report

# ---------------------------------------------------------------------------
# 기본 설정값
# ---------------------------------------------------------------------------
DATA_DIR = "/home/song/rb10_Proximity/src/self_detection_raw/dataset"

TRAIN_FILES = [
    "1dataset_50_25.txt",
    "2dataset_50_25.txt",
    "3dataset_100_25.txt",
    "4dataset_100_25.txt",
    "5dataset_50_50.txt",
    "6dataset_50_50.txt",
    "7dataset_100_50.txt"
]

VAL_FILES = [
    "8dataset_100_50.txt",
]

script_root = Path(__file__).parent.absolute()
OUT_DIR = str(script_root / "model")
NAME_PREFIX = "mlp_vel_practical"

# 학습 하이퍼파라미터
USE_VEL = True
VEL_WINDOW = 10
STD_FLOOR = 1e-2
EPOCHS = 300
BATCH = 256
LR = 1e-3
WD = 1e-4
HIDDEN = 128
HEAD_HIDDEN = 64
DROPOUT = 0.1
SEED = 42
NUM_WORKERS = 4

# 실용 목표 하이퍼파라미터
RAW_ERROR_THRESHOLD = 2000.0
BEST_METRIC_NAME = "within_2000"
EARLY_STOP_PATIENCE = 30
EARLY_STOP_MIN_DELTA = 0.0
EARLY_STOP_WARMUP_EPOCHS = 20
SAVE_PLOTS = True

CHANNEL_NAMES = [f"raw{i}" for i in range(1, 9)]
CHANNEL_LOSS_WEIGHTS = [1.0] * len(CHANNEL_NAMES)  # Edit here if some channels should matter more.
WITHIN_THRESHOLDS = (1000.0, 2000.0, 5000.0, 10000.0)
# ---------------------------------------------------------------------------


def full_paths(names, base_dir):
    paths = []
    for fn in names:
        p = Path(fn)
        if not p.is_absolute():
            paths.append(str(Path(base_dir) / fn))
        else:
            paths.append(str(p))
    return paths


def build_dataloader(dataset, batch_size, shuffle, num_workers, pin_memory):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


class ChannelwiseHuberLoss(nn.Module):
    """
    Channel-wise Huber loss in normalized space.

    The raw-scale target threshold is mapped to each normalized output channel as:
        delta_norm[c] = raw_threshold / Y_std[c]

    This preserves the engineering threshold location even when targets are normalized.
    """

    def __init__(self, delta_norm):
        super().__init__()
        delta_tensor = torch.as_tensor(delta_norm, dtype=torch.float32)
        self.register_buffer("delta", torch.clamp(delta_tensor, min=1e-8))

    def forward(self, pred, target):
        delta = self.delta.to(device=pred.device, dtype=pred.dtype)
        error = pred - target
        abs_error = error.abs()
        quadratic = torch.minimum(abs_error, delta)
        linear = abs_error - quadratic
        loss = 0.5 * quadratic.pow(2) + delta * linear
        return loss.mean()


class WeightedChannelwiseHuberLoss(nn.Module):
    """
    Weighted variant of ChannelwiseHuberLoss.

    Channel weights are normalized to have mean 1.0 so that changing weights does
    not implicitly rescale the overall loss magnitude too aggressively.
    """

    def __init__(self, delta_norm, channel_weights):
        super().__init__()
        delta_tensor = torch.as_tensor(delta_norm, dtype=torch.float32)
        weight_tensor = torch.as_tensor(channel_weights, dtype=torch.float32)
        if delta_tensor.shape != weight_tensor.shape:
            raise ValueError(
                f"delta_norm shape {tuple(delta_tensor.shape)} must match "
                f"channel_weights shape {tuple(weight_tensor.shape)}"
            )
        normalized_weights = weight_tensor / torch.clamp(weight_tensor.mean(), min=1e-8)
        self.register_buffer("delta", torch.clamp(delta_tensor, min=1e-8))
        self.register_buffer("weights", normalized_weights)

    def forward(self, pred, target):
        delta = self.delta.to(device=pred.device, dtype=pred.dtype)
        weights = self.weights.to(device=pred.device, dtype=pred.dtype)
        error = pred - target
        abs_error = error.abs()
        quadratic = torch.minimum(abs_error, delta)
        linear = abs_error - quadratic
        loss = 0.5 * quadratic.pow(2) + delta * linear
        weighted_loss = loss * weights.unsqueeze(0)
        return weighted_loss.mean()


def get_metric_direction(metric_name):
    if metric_name == "within_2000":
        return "max"
    if metric_name in {"rmse", "avg_std"}:
        return "min"
    raise ValueError(f"Unsupported best metric: {metric_name}")


def metric_value_for_selection(metrics, metric_name):
    if metric_name not in metrics:
        raise KeyError(f"Metric '{metric_name}' not found in validation metrics")
    return float(metrics[metric_name])


def is_better_metric(candidate, current_best, metric_name):
    if current_best is None:
        return True
    direction = get_metric_direction(metric_name)
    if direction == "max":
        return candidate > current_best
    return candidate < current_best


def compute_practical_metrics(preds, targets):
    residuals = targets - preds
    abs_errors = np.abs(residuals)
    sq_errors = residuals ** 2

    per_channel_rmse = np.sqrt(np.mean(sq_errors, axis=0))
    per_channel_mae = np.mean(abs_errors, axis=0)
    per_channel_within = {
        int(th): np.mean(abs_errors <= th, axis=0)
        for th in WITHIN_THRESHOLDS
    }

    channel_metrics = compute_channel_metrics(targets, residuals)
    ranked_indices = np.argsort(per_channel_rmse)
    ranked_channels = [
        {
            "channel": CHANNEL_NAMES[idx],
            "rmse": float(per_channel_rmse[idx]),
            "mae": float(per_channel_mae[idx]),
            "within_2000": float(per_channel_within[2000][idx]),
            "residual_std": float(channel_metrics["residual_std"][idx]),
        }
        for idx in ranked_indices
    ]

    metrics = {
        "mae": float(np.mean(abs_errors)),
        "rmse": float(np.sqrt(np.mean(sq_errors))),
        "avg_std": float(np.mean(channel_metrics["residual_std"])),
        "per_channel_rmse": per_channel_rmse,
        "per_channel_mae": per_channel_mae,
        "residual_std": channel_metrics["residual_std"],
        "raw_std": channel_metrics["raw_std"],
        "improvement": channel_metrics["improvement"],
        "ranked_channels": ranked_channels,
        "best_channel_rmse": float(per_channel_rmse[ranked_indices[0]]),
        "best_channel_name": CHANNEL_NAMES[ranked_indices[0]],
        "worst_channel_rmse": float(per_channel_rmse[ranked_indices[-1]]),
        "worst_channel_name": CHANNEL_NAMES[ranked_indices[-1]],
    }

    for threshold in WITHIN_THRESHOLDS:
        key = f"within_{int(threshold)}"
        metrics[key] = float(np.mean(abs_errors <= threshold))
        metrics[f"per_channel_{key}"] = per_channel_within[int(threshold)]

    return metrics


@torch.no_grad()
def evaluate_practical(model, dataloader, dataset, device):
    model.eval()

    all_preds = []
    all_targets = []

    for X, Y in dataloader:
        X = X.to(device)
        pred_norm = model(X)
        all_preds.append(pred_norm.cpu().numpy())
        all_targets.append(Y.cpu().numpy())

    preds_norm = np.concatenate(all_preds, axis=0)
    targets_norm = np.concatenate(all_targets, axis=0)

    preds = preds_norm * dataset.Y_std + dataset.Y_mean
    targets = targets_norm * dataset.Y_std + dataset.Y_mean

    return compute_practical_metrics(preds, targets)


def history_entry(metrics, history):
    history["val_rmse"].append(metrics["rmse"])
    history["val_mae"].append(metrics["mae"])
    history["val_avg_std"].append(metrics["avg_std"])
    history["val_within_1000"].append(metrics["within_1000"])
    history["val_within_2000"].append(metrics["within_2000"])
    history["val_within_5000"].append(metrics["within_5000"])
    history["val_within_10000"].append(metrics["within_10000"])
    history["val_worst_channel_rmse"].append(metrics["worst_channel_rmse"])
    history["val_best_channel_rmse"].append(metrics["best_channel_rmse"])


def to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {key: to_serializable(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(value) for value in obj]
    return obj


def format_practical_report(metrics):
    lines = []
    lines.append(
        "Overall: "
        f"RMSE={metrics['rmse']:.2f}, "
        f"MAE={metrics['mae']:.2f}, "
        f"AvgSTD={metrics['avg_std']:.2f}, "
        f"W1000={metrics['within_1000']:.3f}, "
        f"W2000={metrics['within_2000']:.3f}, "
        f"W5000={metrics['within_5000']:.3f}, "
        f"W10000={metrics['within_10000']:.3f}"
    )
    lines.append(
        f"Best channel RMSE: {metrics['best_channel_name']}={metrics['best_channel_rmse']:.2f}"
    )
    lines.append(
        f"Worst channel RMSE: {metrics['worst_channel_name']}={metrics['worst_channel_rmse']:.2f}"
    )
    lines.append("")
    lines.append("Channels ranked by RMSE (best -> worst)")
    for rank, item in enumerate(metrics["ranked_channels"], start=1):
        lines.append(
            f"{rank:2d}. {item['channel']:<6} "
            f"RMSE={item['rmse']:.2f} "
            f"MAE={item['mae']:.2f} "
            f"W2000={item['within_2000']:.3f} "
            f"STD={item['residual_std']:.2f}"
        )
    lines.append("")
    lines.append(format_metrics_report(
        {
            "raw_std": metrics["raw_std"],
            "residual_std": metrics["residual_std"],
            "improvement": metrics["improvement"],
        },
        CHANNEL_NAMES,
    ))
    return "\n".join(lines)


def save_metrics_summary(run_dir, metrics, args, delta_norm):
    summary = {
        "best_metric_name": args.best_metric_name,
        "raw_error_threshold": args.raw_threshold,
        "normalized_threshold_per_channel": delta_norm,
        "final_validation_metrics": metrics,
    }
    save_json(to_serializable(summary), str(Path(run_dir) / "final_metrics.json"))
    with open(Path(run_dir) / "final_metrics.txt", "w", encoding="utf-8") as f:
        f.write(format_practical_report(metrics))
        f.write("\n")


def plot_training_curves(history, run_dir, raw_threshold):
    if plt is None:
        print("matplotlib not available; skipping plots")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(history["train_loss"], label="Train Loss")
    axes[0, 0].set_title("Training Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True)

    axes[0, 1].plot(history["val_rmse"], label="Val RMSE", color="orange")
    axes[0, 1].axhline(y=raw_threshold, color="red", linestyle="--", label=f"RMSE {raw_threshold:.0f}")
    axes[0, 1].set_title("Validation Raw RMSE")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("RMSE")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    axes[1, 0].plot(history["val_within_2000"], label="Within 2000", color="green")
    axes[1, 0].plot(history["val_within_1000"], label="Within 1000", color="steelblue", alpha=0.7)
    axes[1, 0].set_title("Within-Threshold Ratio")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Ratio")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    axes[1, 1].plot(history["val_avg_std"], label="Avg STD", color="purple")
    axes[1, 1].plot(history["val_worst_channel_rmse"], label="Worst Channel RMSE", color="brown", alpha=0.8)
    axes[1, 1].plot(history["val_best_channel_rmse"], label="Best Channel RMSE", color="teal", alpha=0.8)
    axes[1, 1].axhline(y=raw_threshold, color="red", linestyle="--", label=f"{raw_threshold:.0f} Ref")
    axes[1, 1].set_title("Validation Spread / Channel RMSE")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Raw Scale")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(Path(run_dir) / "training_curves.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train MLP with smoothed velocity and practical raw-scale metrics")
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--train-files", default=",".join(TRAIN_FILES))
    parser.add_argument("--val-files", default=",".join(VAL_FILES))
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--name-prefix", default=NAME_PREFIX)

    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--wd", type=float, default=WD)
    parser.add_argument("--hidden", type=int, default=HIDDEN)
    parser.add_argument("--head-hidden", type=int, default=HEAD_HIDDEN)
    parser.add_argument("--dropout", type=float, default=DROPOUT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--vel-window", type=int, default=VEL_WINDOW, help="Smoothing window size for velocity")
    parser.add_argument(
        "--raw-threshold",
        type=float,
        default=RAW_ERROR_THRESHOLD,
        help="Raw-scale Huber threshold in original sensor units",
    )
    parser.add_argument(
        "--best-metric-name",
        default=BEST_METRIC_NAME,
        choices=["rmse", "within_2000", "avg_std"],
        help="Validation metric used for best checkpoint selection",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=EARLY_STOP_PATIENCE,
        help="Stop training if validation RMSE does not improve for this many epochs",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=EARLY_STOP_MIN_DELTA,
        help="Minimum RMSE improvement required to reset early stopping",
    )
    parser.add_argument(
        "--early-stop-warmup",
        type=int,
        default=EARLY_STOP_WARMUP_EPOCHS,
        help="Do not apply early stopping before this epoch",
    )
    parser.add_argument(
        "--save-plots",
        type=int,
        default=int(SAVE_PLOTS),
        help="Save training plots (1) or skip (0)",
    )

    args = parser.parse_args()

    data_dir = args.data_dir
    train_files = args.train_files.split(",") if args.train_files else []
    val_files = args.val_files.split(",") if args.val_files else []
    vel_window = args.vel_window

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_workers = args.num_workers
    if device.type != "cuda" and num_workers == NUM_WORKERS:
        num_workers = 0

    train_paths = full_paths(train_files, data_dir)
    val_paths = full_paths(val_files, data_dir)

    print(f"\n데이터 로딩 중... (Velocity Smoothing Window: {vel_window})")
    X_train, Y_train = load_multiple_files_v(train_paths, use_vel=USE_VEL, vel_window=vel_window)
    if val_paths:
        X_val, Y_val = load_multiple_files_v(val_paths, use_vel=USE_VEL, vel_window=vel_window)
    else:
        X_val = np.array([]).reshape(0, X_train.shape[1])
        Y_val = np.array([]).reshape(0, Y_train.shape[1])

    print(f"Train Input Shape: {X_train.shape} (Expected N x 18)")

    train_ds = SelfDetectionDataset(
        X_train,
        Y_train,
        std_floor=STD_FLOOR,
    )
    val_ds = SelfDetectionDataset(
        X_val,
        Y_val,
        X_mean=train_ds.X_mean,
        X_std=train_ds.X_std,
        Y_mean=train_ds.Y_mean,
        Y_std=train_ds.Y_std,
        std_floor=STD_FLOOR,
    )

    pin_memory = device.type == "cuda"
    train_loader = build_dataloader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = build_dataloader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model = ModelBV(
        in_dim=X_train.shape[1],
        trunk_hidden=args.hidden,
        head_hidden=args.head_hidden,
        out_dim=Y_train.shape[1],
        dropout=args.dropout,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    delta_norm = (args.raw_threshold / np.maximum(train_ds.Y_std, train_ds.std_floor)).astype(np.float32)
    channel_loss_weights = np.asarray(CHANNEL_LOSS_WEIGHTS, dtype=np.float32)
    if channel_loss_weights.shape[0] != Y_train.shape[1]:
        raise ValueError(
            f"CHANNEL_LOSS_WEIGHTS length ({channel_loss_weights.shape[0]}) must match "
            f"number of output channels ({Y_train.shape[1]})"
        )

    # criterion = ChannelwiseHuberLoss(delta_norm).to(device)
    criterion = WeightedChannelwiseHuberLoss(delta_norm, channel_loss_weights).to(device)

    try:
        from torch.optim.lr_scheduler import ReduceLROnPlateau

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=25,
            verbose=True,
        )
    except Exception:
        scheduler = None

    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    run_dir = ensure_dir(os.path.join(args.out_dir, f"{args.name_prefix}_{timestamp}"))

    best_metric_value = None
    best_epoch = None
    best_rmse_for_early_stop = None
    no_improve_epochs = 0
    history = {
        "train_loss": [],
        "val_rmse": [],
        "val_mae": [],
        "val_avg_std": [],
        "val_within_1000": [],
        "val_within_2000": [],
        "val_within_5000": [],
        "val_within_10000": [],
        "val_worst_channel_rmse": [],
        "val_best_channel_rmse": [],
    }

    run_config = vars(args).copy()
    run_config["normalized_threshold_per_channel"] = delta_norm.tolist()
    run_config["channel_loss_weights"] = channel_loss_weights.tolist()
    save_json(to_serializable(run_config), str(Path(run_dir) / "config.json"))

    print("\n학습 시작...")
    print(f"Raw threshold: {args.raw_threshold:.1f}")
    print("Normalized threshold per channel: " + ", ".join(f"{value:.4f}" for value in delta_norm))
    print("Channel loss weights: " + ", ".join(f"{value:.3f}" for value in channel_loss_weights))

    for epoch in range(1, args.epochs + 1):
        try:
            loss = train_epoch(model, train_loader, criterion, optimizer, device)
        except PermissionError as exc:
            if num_workers > 0:
                print(
                    f"DataLoader worker startup failed ({exc}). "
                    "Retrying with --num-workers 0."
                )
                num_workers = 0
                train_loader = build_dataloader(
                    train_ds,
                    batch_size=args.batch,
                    shuffle=True,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                )
                val_loader = build_dataloader(
                    val_ds,
                    batch_size=args.batch,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                )
                loss = train_epoch(model, train_loader, criterion, optimizer, device)
            else:
                raise

        metrics = evaluate_practical(model, val_loader, val_ds, device) if len(val_paths) > 0 else {}

        history["train_loss"].append(loss)
        if metrics:
            history_entry(metrics, history)

            selection_value = metric_value_for_selection(metrics, args.best_metric_name)
            if scheduler:
                scheduler.step(metrics["rmse"])

            if is_better_metric(selection_value, best_metric_value, args.best_metric_name):
                best_metric_value = selection_value
                best_epoch = epoch
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_metrics": to_serializable(metrics),
                        "normalization": train_ds.get_norm_params(),
                        "args": run_config,
                    },
                    Path(run_dir) / "model.pt",
                )
                print(
                    f"Saved best model to {Path(run_dir) / 'model.pt'} "
                    f"({args.best_metric_name}={selection_value:.4f})"
                )

            rmse_value = metrics["rmse"]
            if (
                best_rmse_for_early_stop is None
                or rmse_value < (best_rmse_for_early_stop - args.early_stop_min_delta)
            ):
                best_rmse_for_early_stop = rmse_value
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

        if epoch % 10 == 0 or epoch == 1:
            msg = f"Epoch {epoch}/{args.epochs} | Train loss: {loss:.6f}"
            if metrics:
                msg += (
                    f" | Val RMSE: {metrics['rmse']:.2f}"
                    f" | W2000: {metrics['within_2000']:.3f}"
                    f" | AvgSTD: {metrics['avg_std']:.2f}"
                    f" | BestCh: {metrics['best_channel_name']}={metrics['best_channel_rmse']:.2f}"
                    f" | WorstCh: {metrics['worst_channel_name']}={metrics['worst_channel_rmse']:.2f}"
                )
            print(msg)

        if (
            metrics
            and epoch >= args.early_stop_warmup
            and no_improve_epochs >= args.early_stop_patience
        ):
            print(
                f"Early stopping at epoch {epoch}: "
                f"validation RMSE did not improve for {args.early_stop_patience} epochs "
                f"(best_rmse={best_rmse_for_early_stop:.2f})"
            )
            break

    final_metrics = {}
    if len(val_paths) > 0 and best_epoch is not None:
        checkpoint = torch.load(Path(run_dir) / "model.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        final_metrics = evaluate_practical(model, val_loader, val_ds, device)

    if final_metrics:
        save_metrics_summary(run_dir, final_metrics, args, delta_norm.tolist())
        print("\nFinal validation summary")
        print(format_practical_report(final_metrics))
        if args.save_plots:
            plot_training_curves(history, run_dir, args.raw_threshold)

    print(
        f"\nFinished training. Best {args.best_metric_name}: "
        f"{best_metric_value if best_metric_value is not None else 'n/a'}"
    )
    if best_rmse_for_early_stop is not None:
        print(f"Best RMSE for early stopping: {best_rmse_for_early_stop:.2f}")
    if best_epoch is not None:
        print(f"Best epoch: {best_epoch}")
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
