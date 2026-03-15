#!/usr/bin/env python3
"""
Train a base + bounded-correction self-detection model.

The design goal is:
    final_pred = base_pred + bounded_correction

where:
    - base_pred is the main self-prediction model
    - bounded_correction is small, history-based, and explicitly constrained

Base branch input:
    current joint sin/cos [+ current smoothed velocity]

Correction branch input:
    previous joint history and joint deltas only
    - sin/cos(q_{t-1})
    - sin/cos(q_{t-2})
    - dq_t
    - dq_{t-1}

This avoids feeding previous sensor values into the correction branch.
"""

import argparse
import os
import random
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

from self_detection_raw.data.loader import load_file
from self_detection_raw.data.loader_v import smooth_data
from self_detection_raw.models.mlp_b_v import ModelBV
from self_detection_raw.train.train import SelfDetectionDataset
from self_detection_raw.utils.io import ensure_dir, save_json
from self_detection_raw.utils.metrics import compute_channel_metrics, format_metrics_report

# ---------------------------------------------------------------------------
# Editable defaults
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
NAME_PREFIX = "mlp_vel_base_corr"

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

CORR_HIDDEN = 32
CORR_DROPOUT = 0.05
CORRECTION_LIMIT_RAW = 1500.0
CORRECTION_PENALTY_BETA = 0.01
RAW_ERROR_THRESHOLD = 2000.0

BEST_METRIC_NAME = "within_2000"
EARLY_STOP_PATIENCE = 15
EARLY_STOP_MIN_DELTA = 0.0
EARLY_STOP_WARMUP_EPOCHS = 10
SAVE_PLOTS = True

CHANNEL_NAMES = [f"raw{i}" for i in range(1, 9)]
CHANNEL_LOSS_WEIGHTS = [1.0] * len(CHANNEL_NAMES)
WITHIN_THRESHOLDS = (1000.0, 2000.0, 5000.0, 10000.0)
# ---------------------------------------------------------------------------


def full_paths(names, base_dir):
    paths = []
    for fn in names:
        p = Path(fn)
        if p.is_absolute():
            paths.append(str(p))
        else:
            paths.append(str(Path(base_dir) / fn))
    return paths


def build_dataloader(dataset, batch_size, shuffle, num_workers, pin_memory):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def encode_joint_positions_deg(joint_pos_deg):
    joint_pos_rad = np.deg2rad(joint_pos_deg)
    return np.concatenate([np.sin(joint_pos_rad), np.cos(joint_pos_rad)], axis=1)


def extract_base_correction_features(data, use_vel=True, vel_window=10):
    if len(data) <= 2:
        return (
            np.empty((0, 0), dtype=np.float32),
            np.empty((0, 8), dtype=np.float32),
            0,
            0,
        )

    joint_pos_deg = data[:, 1:7]
    joint_pos_rad = np.deg2rad(joint_pos_deg)
    joint_encoded = encode_joint_positions_deg(joint_pos_deg)

    base_parts = [joint_encoded[2:]]
    if use_vel:
        joint_vel = smooth_data(data[:, 7:13], window_size=vel_window, polyorder=2)
        base_parts.append(joint_vel[2:])
    base_x = np.concatenate(base_parts, axis=1).astype(np.float32)

    prev1 = joint_encoded[1:-1]
    prev2 = joint_encoded[:-2]
    dq_t = (joint_pos_rad[2:] - joint_pos_rad[1:-1]).astype(np.float32)
    dq_tm1 = (joint_pos_rad[1:-1] - joint_pos_rad[:-2]).astype(np.float32)
    corr_x = np.concatenate([prev1, prev2, dq_t, dq_tm1], axis=1).astype(np.float32)

    y = data[2:, 21:29].astype(np.float32)
    x = np.concatenate([base_x, corr_x], axis=1)
    return x, y, base_x.shape[1], corr_x.shape[1]


def load_multiple_files_base_correction(filepaths, use_vel=True, vel_window=10):
    x_list = []
    y_list = []
    base_dim = None
    corr_dim = None

    for filepath in filepaths:
        if not os.path.exists(filepath):
            print(f"[WARN] File not found: {filepath}")
            continue

        data = load_file(filepath)
        x, y, file_base_dim, file_corr_dim = extract_base_correction_features(
            data,
            use_vel=use_vel,
            vel_window=vel_window,
        )
        if len(x) == 0:
            print(f"[WARN] Not enough samples for 2-step history: {filepath}")
            continue
        if base_dim is None:
            base_dim = file_base_dim
            corr_dim = file_corr_dim
        x_list.append(x)
        y_list.append(y)

    if not x_list:
        return np.array([]), np.array([]), 0, 0

    return np.concatenate(x_list, axis=0), np.concatenate(y_list, axis=0), base_dim, corr_dim


class WeightedChannelwiseHuberLoss(nn.Module):
    def __init__(self, delta_norm, channel_weights):
        super().__init__()
        delta_tensor = torch.as_tensor(delta_norm, dtype=torch.float32)
        weight_tensor = torch.as_tensor(channel_weights, dtype=torch.float32)
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
        return (loss * weights.unsqueeze(0)).mean()


class BaseCorrectionLoss(nn.Module):
    def __init__(self, delta_norm, channel_weights, correction_penalty_beta):
        super().__init__()
        self.pred_loss = WeightedChannelwiseHuberLoss(delta_norm, channel_weights)
        self.correction_penalty_beta = correction_penalty_beta

    def forward(self, pred, target, bounded_correction):
        pred_loss = self.pred_loss(pred, target)
        correction_penalty = bounded_correction.pow(2).mean()
        total = pred_loss + self.correction_penalty_beta * correction_penalty
        return total, pred_loss.detach(), correction_penalty.detach()


class BaseCorrectionModel(nn.Module):
    def __init__(
        self,
        base_in_dim,
        corr_in_dim,
        out_dim,
        trunk_hidden,
        head_hidden,
        dropout,
        corr_hidden,
        corr_dropout,
        correction_limit_norm,
    ):
        super().__init__()
        self.base_in_dim = base_in_dim
        self.corr_in_dim = corr_in_dim
        self.base_model = ModelBV(
            in_dim=base_in_dim,
            trunk_hidden=trunk_hidden,
            head_hidden=head_hidden,
            out_dim=out_dim,
            dropout=dropout,
        )
        self.corr_model = nn.Sequential(
            nn.Linear(corr_in_dim, corr_hidden),
            nn.ReLU(),
            nn.Dropout(corr_dropout),
            nn.Linear(corr_hidden, corr_hidden),
            nn.ReLU(),
            nn.Linear(corr_hidden, out_dim),
        )
        self.register_buffer(
            "correction_limit",
            torch.as_tensor(correction_limit_norm, dtype=torch.float32),
        )

    def forward(self, x):
        base_x = x[:, :self.base_in_dim]
        corr_x = x[:, self.base_in_dim:self.base_in_dim + self.corr_in_dim]
        base_pred = self.base_model(base_x)
        corr_raw = self.corr_model(corr_x)
        correction_limit = self.correction_limit.to(device=x.device, dtype=x.dtype)
        bounded_correction = torch.tanh(corr_raw) * correction_limit
        pred = base_pred + bounded_correction
        return pred, base_pred, bounded_correction


def get_metric_direction(metric_name):
    if metric_name == "within_2000":
        return "max"
    if metric_name in {"rmse", "avg_std"}:
        return "min"
    raise ValueError(f"Unsupported best metric: {metric_name}")


def is_better_metric(candidate, current_best, metric_name):
    if current_best is None:
        return True
    if get_metric_direction(metric_name) == "max":
        return candidate > current_best
    return candidate < current_best


def metric_value_for_selection(metrics, metric_name):
    return float(metrics[metric_name])


def compute_practical_metrics(preds, targets, base_preds=None):
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
        "raw_std": channel_metrics["raw_std"],
        "residual_std": channel_metrics["residual_std"],
        "improvement": channel_metrics["improvement"],
        "ranked_channels": ranked_channels,
        "best_channel_rmse": float(per_channel_rmse[ranked_indices[0]]),
        "best_channel_name": CHANNEL_NAMES[ranked_indices[0]],
        "worst_channel_rmse": float(per_channel_rmse[ranked_indices[-1]]),
        "worst_channel_name": CHANNEL_NAMES[ranked_indices[-1]],
    }

    if base_preds is not None:
        base_residuals = targets - base_preds
        metrics["base_rmse"] = float(np.sqrt(np.mean(base_residuals ** 2)))
        correction_raw = preds - base_preds
        metrics["mean_abs_correction"] = float(np.mean(np.abs(correction_raw)))
        metrics["max_abs_correction"] = float(np.max(np.abs(correction_raw)))

    for threshold in WITHIN_THRESHOLDS:
        key = f"within_{int(threshold)}"
        metrics[key] = float(np.mean(abs_errors <= threshold))
        metrics[f"per_channel_{key}"] = per_channel_within[int(threshold)]

    return metrics


@torch.no_grad()
def evaluate_practical(model, dataloader, dataset, device):
    model.eval()

    pred_list = []
    base_pred_list = []
    target_list = []

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        pred_norm, base_pred_norm, _ = model(x_batch)
        pred_list.append(pred_norm.cpu().numpy())
        base_pred_list.append(base_pred_norm.cpu().numpy())
        target_list.append(y_batch.cpu().numpy())

    preds_norm = np.concatenate(pred_list, axis=0)
    base_preds_norm = np.concatenate(base_pred_list, axis=0)
    targets_norm = np.concatenate(target_list, axis=0)

    preds = preds_norm * dataset.Y_std + dataset.Y_mean
    base_preds = base_preds_norm * dataset.Y_std + dataset.Y_mean
    targets = targets_norm * dataset.Y_std + dataset.Y_mean
    return compute_practical_metrics(preds, targets, base_preds=base_preds)


def train_epoch_base_correction(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_pred_loss = 0.0
    total_corr_penalty = 0.0
    n_batches = 0

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        pred, _, bounded_correction = model(x_batch)
        loss, pred_loss, corr_penalty = criterion(pred, y_batch, bounded_correction)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_pred_loss += pred_loss.item()
        total_corr_penalty += corr_penalty.item()
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "pred_loss": total_pred_loss / n_batches,
        "corr_penalty": total_corr_penalty / n_batches,
    }


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
    history["val_base_rmse"].append(metrics.get("base_rmse", np.nan))
    history["val_mean_abs_correction"].append(metrics.get("mean_abs_correction", np.nan))


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
    if "base_rmse" in metrics:
        lines.append(
            "Base vs final: "
            f"base_RMSE={metrics['base_rmse']:.2f}, "
            f"mean|corr|={metrics['mean_abs_correction']:.2f}, "
            f"max|corr|={metrics['max_abs_correction']:.2f}"
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
    lines.append(
        format_metrics_report(
            {
                "raw_std": metrics["raw_std"],
                "residual_std": metrics["residual_std"],
                "improvement": metrics["improvement"],
            },
            CHANNEL_NAMES,
        )
    )
    return "\n".join(lines)


def plot_training_curves(history, run_dir, raw_threshold):
    if plt is None:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(history["train_loss"], label="Train Loss")
    axes[0, 0].plot(history["train_pred_loss"], label="Pred Loss", alpha=0.7)
    axes[0, 0].plot(history["train_corr_penalty"], label="Corr Penalty", alpha=0.7)
    axes[0, 0].set_title("Training Loss Terms")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(history["val_rmse"], label="Val RMSE", color="orange")
    axes[0, 1].plot(history["val_base_rmse"], label="Base RMSE", color="steelblue", alpha=0.8)
    axes[0, 1].axhline(y=raw_threshold, color="red", linestyle="--", label=f"RMSE {raw_threshold:.0f}")
    axes[0, 1].set_title("Validation RMSE")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    axes[1, 0].plot(history["val_within_2000"], label="Within 2000", color="green")
    axes[1, 0].plot(history["val_within_1000"], label="Within 1000", color="purple", alpha=0.8)
    axes[1, 0].set_title("Within-Threshold Ratio")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    axes[1, 1].plot(history["val_mean_abs_correction"], label="Mean |Correction|", color="brown")
    axes[1, 1].axhline(y=raw_threshold, color="red", linestyle="--", label=f"{raw_threshold:.0f} Ref")
    axes[1, 1].set_title("Correction Magnitude (raw scale)")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(Path(run_dir) / "training_curves.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train base + bounded-correction self-detection model")
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
    parser.add_argument("--corr-hidden", type=int, default=CORR_HIDDEN)
    parser.add_argument("--corr-dropout", type=float, default=CORR_DROPOUT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--use-vel", type=int, default=1 if USE_VEL else 0)
    parser.add_argument("--vel-window", type=int, default=VEL_WINDOW)
    parser.add_argument("--raw-threshold", type=float, default=RAW_ERROR_THRESHOLD)
    parser.add_argument("--correction-limit-raw", type=float, default=CORRECTION_LIMIT_RAW)
    parser.add_argument("--correction-penalty-beta", type=float, default=CORRECTION_PENALTY_BETA)
    parser.add_argument(
        "--best-metric-name",
        default=BEST_METRIC_NAME,
        choices=["rmse", "within_2000", "avg_std"],
    )
    parser.add_argument("--early-stop-patience", type=int, default=EARLY_STOP_PATIENCE)
    parser.add_argument("--early-stop-min-delta", type=float, default=EARLY_STOP_MIN_DELTA)
    parser.add_argument("--early-stop-warmup", type=int, default=EARLY_STOP_WARMUP_EPOCHS)
    parser.add_argument("--save-plots", type=int, default=int(SAVE_PLOTS))
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    use_vel = bool(args.use_vel)
    num_workers = args.num_workers
    if device.type != "cuda" and num_workers == NUM_WORKERS:
        num_workers = 0

    train_paths = full_paths(args.train_files.split(",") if args.train_files else [], args.data_dir)
    val_paths = full_paths(args.val_files.split(",") if args.val_files else [], args.data_dir)

    print(f"\n데이터 로딩 중... (Velocity Smoothing Window: {args.vel_window})")
    print("Structure: final = base(current joints[/vel]) + bounded correction(joint history)")
    print("Alignment note: first 2 samples of each file are dropped for joint history.")
    x_train, y_train, base_in_dim, corr_in_dim = load_multiple_files_base_correction(
        train_paths,
        use_vel=use_vel,
        vel_window=args.vel_window,
    )
    if val_paths:
        x_val, y_val, _, _ = load_multiple_files_base_correction(
            val_paths,
            use_vel=use_vel,
            vel_window=args.vel_window,
        )
    else:
        x_val = np.array([]).reshape(0, x_train.shape[1])
        y_val = np.array([]).reshape(0, y_train.shape[1])

    if x_train.size == 0 or y_train.size == 0:
        raise ValueError("Training dataset is empty. Check file paths and history alignment.")

    print(f"Train Input Shape: {x_train.shape}")
    print(f"Base input dim: {base_in_dim}")
    print(f"Correction input dim: {corr_in_dim}")

    train_ds = SelfDetectionDataset(x_train, y_train, std_floor=STD_FLOOR)
    val_ds = SelfDetectionDataset(
        x_val,
        y_val,
        X_mean=train_ds.X_mean,
        X_std=train_ds.X_std,
        Y_mean=train_ds.Y_mean,
        Y_std=train_ds.Y_std,
        std_floor=STD_FLOOR,
    )

    delta_norm = (args.raw_threshold / np.maximum(train_ds.Y_std, train_ds.std_floor)).astype(np.float32)
    correction_limit_norm = (
        args.correction_limit_raw / np.maximum(train_ds.Y_std, train_ds.std_floor)
    ).astype(np.float32)
    channel_loss_weights = np.asarray(CHANNEL_LOSS_WEIGHTS, dtype=np.float32)

    model = BaseCorrectionModel(
        base_in_dim=base_in_dim,
        corr_in_dim=corr_in_dim,
        out_dim=y_train.shape[1],
        trunk_hidden=args.hidden,
        head_hidden=args.head_hidden,
        dropout=args.dropout,
        corr_hidden=args.corr_hidden,
        corr_dropout=args.corr_dropout,
        correction_limit_norm=correction_limit_norm,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = BaseCorrectionLoss(
        delta_norm=delta_norm,
        channel_weights=channel_loss_weights,
        correction_penalty_beta=args.correction_penalty_beta,
    ).to(device)

    try:
        from torch.optim.lr_scheduler import ReduceLROnPlateau

        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=25, verbose=True)
    except Exception:
        scheduler = None

    pin_memory = device.type == "cuda"
    train_loader = build_dataloader(train_ds, args.batch, True, num_workers, pin_memory)
    val_loader = build_dataloader(val_ds, args.batch, False, num_workers, pin_memory)

    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    run_dir = ensure_dir(os.path.join(args.out_dir, f"{args.name_prefix}_{timestamp}"))

    run_config = vars(args).copy()
    run_config["use_vel"] = use_vel
    run_config["base_in_dim"] = base_in_dim
    run_config["corr_in_dim"] = corr_in_dim
    run_config["normalized_threshold_per_channel"] = delta_norm.tolist()
    run_config["correction_limit_norm_per_channel"] = correction_limit_norm.tolist()
    save_json(to_serializable(run_config), str(Path(run_dir) / "config.json"))

    history = {
        "train_loss": [],
        "train_pred_loss": [],
        "train_corr_penalty": [],
        "val_rmse": [],
        "val_mae": [],
        "val_avg_std": [],
        "val_within_1000": [],
        "val_within_2000": [],
        "val_within_5000": [],
        "val_within_10000": [],
        "val_worst_channel_rmse": [],
        "val_best_channel_rmse": [],
        "val_base_rmse": [],
        "val_mean_abs_correction": [],
    }

    best_metric_value = None
    best_rmse_for_early_stop = None
    no_improve_epochs = 0

    print("\n학습 시작...")
    print(f"Raw threshold: {args.raw_threshold:.1f}")
    print(f"Correction raw limit: {args.correction_limit_raw:.1f}")
    print(f"Correction penalty beta: {args.correction_penalty_beta:.5f}")
    print("Normalized threshold per channel: " + ", ".join(f"{v:.4f}" for v in delta_norm))
    print("Correction limit per channel: " + ", ".join(f"{v:.4f}" for v in correction_limit_norm))

    for epoch in range(1, args.epochs + 1):
        batch_losses = train_epoch_base_correction(model, train_loader, criterion, optimizer, device)
        metrics = evaluate_practical(model, val_loader, val_ds, device) if val_paths else {}

        history["train_loss"].append(batch_losses["loss"])
        history["train_pred_loss"].append(batch_losses["pred_loss"])
        history["train_corr_penalty"].append(batch_losses["corr_penalty"])

        if metrics:
            history_entry(metrics, history)
            if scheduler:
                scheduler.step(metrics["rmse"])

            selection_value = metric_value_for_selection(metrics, args.best_metric_name)
            if is_better_metric(selection_value, best_metric_value, args.best_metric_name):
                best_metric_value = selection_value
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

            current_rmse = metrics["rmse"]
            if best_rmse_for_early_stop is None or current_rmse < (best_rmse_for_early_stop - args.early_stop_min_delta):
                best_rmse_for_early_stop = current_rmse
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            if epoch % 10 == 0 or epoch == 1:
                print(
                    f"Epoch {epoch}/{args.epochs} | "
                    f"Train loss: {batch_losses['loss']:.6f} | "
                    f"Pred loss: {batch_losses['pred_loss']:.6f} | "
                    f"Corr pen: {batch_losses['corr_penalty']:.6f} | "
                    f"Val RMSE: {metrics['rmse']:.2f} | "
                    f"W2000: {metrics['within_2000']:.3f} | "
                    f"Base RMSE: {metrics['base_rmse']:.2f} | "
                    f"Mean|corr|: {metrics['mean_abs_correction']:.2f}"
                )

            if epoch >= args.early_stop_warmup and no_improve_epochs >= args.early_stop_patience:
                print(
                    f"Early stopping at epoch {epoch}: validation RMSE did not improve "
                    f"for {args.early_stop_patience} epochs (best_rmse={best_rmse_for_early_stop:.2f})"
                )
                break

    final_metrics = evaluate_practical(model, val_loader, val_ds, device) if val_paths else {}
    if final_metrics:
        summary = {
            "best_metric_name": args.best_metric_name,
            "raw_error_threshold": args.raw_threshold,
            "correction_limit_raw": args.correction_limit_raw,
            "correction_penalty_beta": args.correction_penalty_beta,
            "normalized_threshold_per_channel": delta_norm,
            "correction_limit_norm_per_channel": correction_limit_norm,
            "final_validation_metrics": final_metrics,
        }
        save_json(to_serializable(summary), str(Path(run_dir) / "final_metrics.json"))
        with open(Path(run_dir) / "final_metrics.txt", "w", encoding="utf-8") as f:
            f.write(format_practical_report(final_metrics))
            f.write("\n")

    if args.save_plots:
        plot_training_curves(history, run_dir, args.raw_threshold)

    print(f"Finished training. Best {args.best_metric_name}: {best_metric_value}")
    print(f"Best RMSE for early stopping: {best_rmse_for_early_stop}")
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
