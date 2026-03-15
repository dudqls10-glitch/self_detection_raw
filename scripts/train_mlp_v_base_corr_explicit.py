#!/usr/bin/env python3
"""
Explicit base + correction training script.

This version follows the conceptual structure directly:
    S_hat_t = S_base_t + lambda * C_t

    S_base_t = BasePredictor(base_input_t)
    C_t = tanh(CorrectionPredictor(corr_input_t))

Each batch contains:
    - base_input
    - corr_input
    - target
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
from torch.utils.data import DataLoader, Dataset

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

from self_detection_raw.data.loader import load_file
from self_detection_raw.data.loader_v import smooth_data
from self_detection_raw.models.mlp_b_v import ModelBV
from self_detection_raw.utils.io import ensure_dir, save_json
from self_detection_raw.utils.metrics import compute_channel_metrics, format_metrics_report

# ---------------------------------------------------------------------------
# Editable defaults
# ---------------------------------------------------------------------------
DATA_DIR = "/home/song/rb10_Proximity/src/self_detection_raw/dataset"

TRAIN_FILES = [
    "[1]dataset_50_25_new.txt",
    "[2]dataset_50_25_new.txt",
    "[3]dataset_100_25_new.txt",
 
 
]

VAL_FILES = [
    "[4]dataset_100_25_new.txt",
]

script_root = Path(__file__).parent.absolute()
OUT_DIR = str(script_root / "model")
NAME_PREFIX = "mlp_vel_base_corr_explicit"

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


def compute_stats(x, std_floor):
    mean = np.mean(x, axis=0).astype(np.float32)
    std = np.std(x, axis=0).astype(np.float32)
    std = np.maximum(std, std_floor).astype(np.float32)
    return mean, std


class BaseCorrectionDataset(Dataset):
    def __init__(
        self,
        base_x,
        corr_x,
        y,
        base_mean=None,
        base_std=None,
        corr_mean=None,
        corr_std=None,
        y_mean=None,
        y_std=None,
        std_floor=1e-2,
    ):
        self.base_x = base_x.astype(np.float32)
        self.corr_x = corr_x.astype(np.float32)
        self.y_raw = y.astype(np.float32)
        self.std_floor = std_floor

        if base_mean is None:
            base_mean, base_std = compute_stats(self.base_x, std_floor)
        if corr_mean is None:
            corr_mean, corr_std = compute_stats(self.corr_x, std_floor)
        if y_mean is None:
            y_mean, y_std = compute_stats(self.y_raw, std_floor)

        self.base_mean = base_mean.astype(np.float32)
        self.base_std = base_std.astype(np.float32)
        self.corr_mean = corr_mean.astype(np.float32)
        self.corr_std = corr_std.astype(np.float32)
        self.y_mean = y_mean.astype(np.float32)
        self.y_std = y_std.astype(np.float32)

        self.base_norm = (self.base_x - self.base_mean) / self.base_std
        self.corr_norm = (self.corr_x - self.corr_mean) / self.corr_std
        self.y_norm = (self.y_raw - self.y_mean) / self.y_std

    def __len__(self):
        return len(self.base_x)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.base_norm[idx]),
            torch.from_numpy(self.corr_norm[idx]),
            torch.from_numpy(self.y_norm[idx]),
        )

    def get_norm_params(self):
        return {
            "base_mean": self.base_mean,
            "base_std": self.base_std,
            "corr_mean": self.corr_mean,
            "corr_std": self.corr_std,
            "Y_mean": self.y_mean,
            "Y_std": self.y_std,
        }


def full_paths(names, base_dir):
    return [str((Path(base_dir) / fn) if not Path(fn).is_absolute() else Path(fn)) for fn in names]


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


def build_sequence_examples(data, use_vel=True, vel_window=10):
    if len(data) <= 2:
        return (
            np.empty((0, 0), dtype=np.float32),
            np.empty((0, 0), dtype=np.float32),
            np.empty((0, 8), dtype=np.float32),
        )

    q_deg = data[:, 1:7]
    q_rad = np.deg2rad(q_deg)
    q_enc = encode_joint_positions_deg(q_deg)

    dq = np.zeros_like(q_rad, dtype=np.float32)
    dq[1:] = (q_rad[1:] - q_rad[:-1]).astype(np.float32)

    base_parts = [q_enc[2:]]
    if use_vel:
        jv = smooth_data(data[:, 7:13], window_size=vel_window, polyorder=2)
        base_parts.append(jv[2:].astype(np.float32))
    base_x = np.concatenate(base_parts, axis=1).astype(np.float32)

    corr_x = np.concatenate(
        [
            q_enc[1:-1],      # q_{t-1}
            q_enc[:-2],       # q_{t-2}
            dq[2:],           # dq_t
            dq[1:-1],         # dq_{t-1}
        ],
        axis=1,
    ).astype(np.float32)

    y = data[2:, 21:29].astype(np.float32)
    return base_x, corr_x, y


def load_multiple_sequence_files(filepaths, use_vel=True, vel_window=10):
    base_list = []
    corr_list = []
    y_list = []

    for filepath in filepaths:
        if not os.path.exists(filepath):
            print(f"[WARN] File not found: {filepath}")
            continue
        data = load_file(filepath)
        base_x, corr_x, y = build_sequence_examples(data, use_vel=use_vel, vel_window=vel_window)
        if len(base_x) == 0:
            print(f"[WARN] Not enough samples for history features: {filepath}")
            continue
        base_list.append(base_x)
        corr_list.append(corr_x)
        y_list.append(y)

    if not base_list:
        return np.array([]), np.array([]), np.array([])

    return (
        np.concatenate(base_list, axis=0),
        np.concatenate(corr_list, axis=0),
        np.concatenate(y_list, axis=0),
    )


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


class CorrectionPredictor(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, corr_input):
        return torch.tanh(self.net(corr_input))


class FullPredictor(nn.Module):
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
        lambda_norm,
    ):
        super().__init__()
        self.base_predictor = ModelBV(
            in_dim=base_in_dim,
            trunk_hidden=trunk_hidden,
            head_hidden=head_hidden,
            out_dim=out_dim,
            dropout=dropout,
        )
        self.correction_predictor = CorrectionPredictor(
            in_dim=corr_in_dim,
            hidden_dim=corr_hidden,
            out_dim=out_dim,
            dropout=corr_dropout,
        )
        self.register_buffer("lambda_norm", torch.as_tensor(lambda_norm, dtype=torch.float32))

    def forward(self, base_input, corr_input):
        s_base = self.base_predictor(base_input)
        c = self.correction_predictor(corr_input)
        lambda_norm = self.lambda_norm.to(device=s_base.device, dtype=s_base.dtype)
        s_hat = s_base + lambda_norm * c
        return s_hat, s_base, c


class BaseCorrLoss(nn.Module):
    def __init__(self, delta_norm, channel_weights, beta):
        super().__init__()
        self.pred_loss = WeightedChannelwiseHuberLoss(delta_norm, channel_weights)
        self.beta = beta

    def forward(self, s_hat, target, c):
        pred_loss = self.pred_loss(s_hat, target)
        corr_penalty = c.pow(2).mean()
        total = pred_loss + self.beta * corr_penalty
        return total, pred_loss.detach(), corr_penalty.detach()


def compute_practical_metrics(preds, targets, base_preds=None, corr_raw=None):
    residuals = targets - preds
    abs_errors = np.abs(residuals)
    sq_errors = residuals ** 2
    per_channel_rmse = np.sqrt(np.mean(sq_errors, axis=0))
    per_channel_mae = np.mean(abs_errors, axis=0)
    per_channel_within = {int(th): np.mean(abs_errors <= th, axis=0) for th in WITHIN_THRESHOLDS}
    channel_metrics = compute_channel_metrics(targets, residuals)
    ranked_indices = np.argsort(per_channel_rmse)

    metrics = {
        "mae": float(np.mean(abs_errors)),
        "rmse": float(np.sqrt(np.mean(sq_errors))),
        "avg_std": float(np.mean(channel_metrics["residual_std"])),
        "per_channel_rmse": per_channel_rmse,
        "per_channel_mae": per_channel_mae,
        "raw_std": channel_metrics["raw_std"],
        "residual_std": channel_metrics["residual_std"],
        "improvement": channel_metrics["improvement"],
        "best_channel_name": CHANNEL_NAMES[ranked_indices[0]],
        "best_channel_rmse": float(per_channel_rmse[ranked_indices[0]]),
        "worst_channel_name": CHANNEL_NAMES[ranked_indices[-1]],
        "worst_channel_rmse": float(per_channel_rmse[ranked_indices[-1]]),
        "ranked_channels": [
            {
                "channel": CHANNEL_NAMES[idx],
                "rmse": float(per_channel_rmse[idx]),
                "mae": float(per_channel_mae[idx]),
                "within_2000": float(per_channel_within[2000][idx]),
                "residual_std": float(channel_metrics["residual_std"][idx]),
            }
            for idx in ranked_indices
        ],
    }

    for threshold in WITHIN_THRESHOLDS:
        key = f"within_{int(threshold)}"
        metrics[key] = float(np.mean(abs_errors <= threshold))
        metrics[f"per_channel_{key}"] = per_channel_within[int(threshold)]

    if base_preds is not None:
        metrics["base_rmse"] = float(np.sqrt(np.mean((targets - base_preds) ** 2)))
    if corr_raw is not None:
        metrics["mean_abs_correction"] = float(np.mean(np.abs(corr_raw)))
        metrics["max_abs_correction"] = float(np.max(np.abs(corr_raw)))

    return metrics


@torch.no_grad()
def evaluate_model(model, dataloader, dataset, device):
    model.eval()
    pred_list = []
    base_list = []
    corr_list = []
    target_list = []

    for base_input, corr_input, target in dataloader:
        base_input = base_input.to(device)
        corr_input = corr_input.to(device)
        s_hat, s_base, c = model(base_input, corr_input)
        pred_list.append(s_hat.cpu().numpy())
        base_list.append(s_base.cpu().numpy())
        corr_list.append(c.cpu().numpy())
        target_list.append(target.cpu().numpy())

    preds_norm = np.concatenate(pred_list, axis=0)
    base_preds_norm = np.concatenate(base_list, axis=0)
    corr_norm = np.concatenate(corr_list, axis=0)
    targets_norm = np.concatenate(target_list, axis=0)

    preds = preds_norm * dataset.y_std + dataset.y_mean
    base_preds = base_preds_norm * dataset.y_std + dataset.y_mean
    targets = targets_norm * dataset.y_std + dataset.y_mean
    corr_raw = corr_norm * (model.lambda_norm.cpu().numpy() * dataset.y_std)
    return compute_practical_metrics(preds, targets, base_preds=base_preds, corr_raw=corr_raw)


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_pred_loss = 0.0
    total_corr_penalty = 0.0
    n_batches = 0

    for base_input, corr_input, target in dataloader:
        base_input = base_input.to(device)
        corr_input = corr_input.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        s_hat, _, c = model(base_input, corr_input)
        loss, pred_loss, corr_penalty = criterion(s_hat, target, c)
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


def get_metric_direction(metric_name):
    return "max" if metric_name == "within_2000" else "min"


def is_better_metric(candidate, best_value, metric_name):
    if best_value is None:
        return True
    if get_metric_direction(metric_name) == "max":
        return candidate > best_value
    return candidate < best_value


def history_entry(metrics, history):
    history["val_rmse"].append(metrics["rmse"])
    history["val_mae"].append(metrics["mae"])
    history["val_within_1000"].append(metrics["within_1000"])
    history["val_within_2000"].append(metrics["within_2000"])
    history["val_base_rmse"].append(metrics.get("base_rmse", np.nan))
    history["val_mean_abs_correction"].append(metrics.get("mean_abs_correction", np.nan))


def to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def format_report(metrics):
    lines = [
        (
            f"RMSE={metrics['rmse']:.2f}, MAE={metrics['mae']:.2f}, "
            f"W2000={metrics['within_2000']:.3f}, AvgSTD={metrics['avg_std']:.2f}"
        )
    ]
    if "base_rmse" in metrics:
        lines.append(
            f"BaseRMSE={metrics['base_rmse']:.2f}, "
            f"Mean|corr|={metrics['mean_abs_correction']:.2f}, "
            f"Max|corr|={metrics['max_abs_correction']:.2f}"
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


def plot_training_curves(history, run_dir, raw_threshold):
    if plt is None:
        return
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].plot(history["train_loss"], label="Train loss")
    axes[0, 0].plot(history["train_pred_loss"], label="Pred loss")
    axes[0, 0].plot(history["train_corr_penalty"], label="Corr penalty")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(history["val_rmse"], label="Val RMSE")
    axes[0, 1].plot(history["val_base_rmse"], label="Base RMSE")
    axes[0, 1].axhline(y=raw_threshold, color="red", linestyle="--")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    axes[1, 0].plot(history["val_within_2000"], label="Within 2000")
    axes[1, 0].plot(history["val_within_1000"], label="Within 1000")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    axes[1, 1].plot(history["val_mean_abs_correction"], label="Mean |corr|")
    axes[1, 1].axhline(y=raw_threshold, color="red", linestyle="--")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(Path(run_dir) / "training_curves.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train explicit base + correction model")
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
    parser.add_argument("--best-metric-name", choices=["rmse", "within_2000", "avg_std"], default=BEST_METRIC_NAME)
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
    num_workers = 0 if (device.type != "cuda" and args.num_workers == NUM_WORKERS) else args.num_workers

    train_paths = full_paths(args.train_files.split(",") if args.train_files else [], args.data_dir)
    val_paths = full_paths(args.val_files.split(",") if args.val_files else [], args.data_dir)

    print(f"\n데이터 로딩 중... (Velocity Smoothing Window: {args.vel_window})")
    print("Structure: S_hat = S_base(base_input) + lambda * C(corr_input)")
    print("History features: [q_(t-1), q_(t-2), dq_t, dq_(t-1)]")
    print("Alignment note: first 2 samples of each file are dropped.")

    train_base, train_corr, train_y = load_multiple_sequence_files(
        train_paths,
        use_vel=use_vel,
        vel_window=args.vel_window,
    )
    val_base, val_corr, val_y = load_multiple_sequence_files(
        val_paths,
        use_vel=use_vel,
        vel_window=args.vel_window,
    ) if val_paths else (np.array([]), np.array([]), np.array([]))

    if train_base.size == 0:
        raise ValueError("Training dataset is empty. Check file paths and history alignment.")

    print(f"Train base input shape: {train_base.shape}")
    print(f"Train corr input shape: {train_corr.shape}")
    print(f"Train target shape: {train_y.shape}")

    train_ds = BaseCorrectionDataset(train_base, train_corr, train_y, std_floor=STD_FLOOR)
    val_ds = BaseCorrectionDataset(
        val_base,
        val_corr,
        val_y,
        base_mean=train_ds.base_mean,
        base_std=train_ds.base_std,
        corr_mean=train_ds.corr_mean,
        corr_std=train_ds.corr_std,
        y_mean=train_ds.y_mean,
        y_std=train_ds.y_std,
        std_floor=STD_FLOOR,
    )

    delta_norm = (args.raw_threshold / np.maximum(train_ds.y_std, train_ds.std_floor)).astype(np.float32)
    lambda_norm = (args.correction_limit_raw / np.maximum(train_ds.y_std, train_ds.std_floor)).astype(np.float32)
    channel_weights = np.asarray(CHANNEL_LOSS_WEIGHTS, dtype=np.float32)

    model = FullPredictor(
        base_in_dim=train_base.shape[1],
        corr_in_dim=train_corr.shape[1],
        out_dim=train_y.shape[1],
        trunk_hidden=args.hidden,
        head_hidden=args.head_hidden,
        dropout=args.dropout,
        corr_hidden=args.corr_hidden,
        corr_dropout=args.corr_dropout,
        lambda_norm=lambda_norm,
    ).to(device)
    criterion = BaseCorrLoss(delta_norm, channel_weights, args.correction_penalty_beta).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

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
    run_config["base_in_dim"] = train_base.shape[1]
    run_config["corr_in_dim"] = train_corr.shape[1]
    run_config["lambda_norm"] = lambda_norm.tolist()
    run_config["normalized_threshold_per_channel"] = delta_norm.tolist()
    save_json(to_serializable(run_config), str(Path(run_dir) / "config.json"))

    history = {
        "train_loss": [],
        "train_pred_loss": [],
        "train_corr_penalty": [],
        "val_rmse": [],
        "val_mae": [],
        "val_within_1000": [],
        "val_within_2000": [],
        "val_base_rmse": [],
        "val_mean_abs_correction": [],
    }

    best_metric_value = None
    best_rmse = None
    no_improve_epochs = 0

    print("\n학습 시작...")
    print(f"Raw threshold: {args.raw_threshold:.1f}")
    print(f"Correction raw limit: {args.correction_limit_raw:.1f}")
    print(f"Correction penalty beta: {args.correction_penalty_beta:.5f}")

    for epoch in range(1, args.epochs + 1):
        losses = train_epoch(model, train_loader, criterion, optimizer, device)
        metrics = evaluate_model(model, val_loader, val_ds, device) if val_paths else {}

        history["train_loss"].append(losses["loss"])
        history["train_pred_loss"].append(losses["pred_loss"])
        history["train_corr_penalty"].append(losses["corr_penalty"])

        if metrics:
            history_entry(metrics, history)
            if scheduler:
                scheduler.step(metrics["rmse"])

            selected = float(metrics[args.best_metric_name])
            if is_better_metric(selected, best_metric_value, args.best_metric_name):
                best_metric_value = selected
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
                print(f"Saved best model to {Path(run_dir) / 'model.pt'} ({args.best_metric_name}={selected:.4f})")

            current_rmse = metrics["rmse"]
            if best_rmse is None or current_rmse < (best_rmse - args.early_stop_min_delta):
                best_rmse = current_rmse
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            if epoch % 10 == 0 or epoch == 1:
                print(
                    f"Epoch {epoch}/{args.epochs} | "
                    f"Train loss: {losses['loss']:.6f} | "
                    f"Pred loss: {losses['pred_loss']:.6f} | "
                    f"Corr pen: {losses['corr_penalty']:.6f} | "
                    f"Val RMSE: {metrics['rmse']:.2f} | "
                    f"W2000: {metrics['within_2000']:.3f} | "
                    f"Base RMSE: {metrics['base_rmse']:.2f} | "
                    f"Mean|corr|: {metrics['mean_abs_correction']:.2f}"
                )

            if epoch >= args.early_stop_warmup and no_improve_epochs >= args.early_stop_patience:
                print(
                    f"Early stopping at epoch {epoch}: validation RMSE did not improve "
                    f"for {args.early_stop_patience} epochs (best_rmse={best_rmse:.2f})"
                )
                break

    final_metrics = evaluate_model(model, val_loader, val_ds, device) if val_paths else {}
    if final_metrics:
        save_json(
            to_serializable(
                {
                    "best_metric_name": args.best_metric_name,
                    "raw_error_threshold": args.raw_threshold,
                    "correction_limit_raw": args.correction_limit_raw,
                    "correction_penalty_beta": args.correction_penalty_beta,
                    "final_validation_metrics": final_metrics,
                }
            ),
            str(Path(run_dir) / "final_metrics.json"),
        )
        with open(Path(run_dir) / "final_metrics.txt", "w", encoding="utf-8") as f:
            f.write(format_report(final_metrics))
            f.write("\n")

    if args.save_plots:
        plot_training_curves(history, run_dir, args.raw_threshold)

    print(f"Finished training. Best {args.best_metric_name}: {best_metric_value}")
    print(f"Best RMSE for early stopping: {best_rmse}")
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
