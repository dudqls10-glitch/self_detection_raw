#!/usr/bin/env python3
"""
MLP training with optional previous-sensor input.

Target structure:
    input_t  = [joint_features_t, sensor_{t-1}, sensor_{t-2}, ...]
    target_t = sensor_t

This keeps the current project structure close to train_mlp_v.py:
- same explicit train/val file list style
- same ModelBV usage
- same SelfDetectionDataset normalization pipeline
- optional velocity input
- optional previous sensor channels / previous sensor steps

Examples:
    joints only:
        python3 scripts/train_mlp_v_prev_sensor.py --use-vel 0 --use-prev-sensor 0

    joints + velocity:
        python3 scripts/train_mlp_v_prev_sensor.py --use-vel 1 --use-prev-sensor 0

    joints + velocity + previous sensors:
        python3 scripts/train_mlp_v_prev_sensor.py --use-vel 1 --use-prev-sensor 1

    joints + velocity + selected previous sensors:
        python3 scripts/train_mlp_v_prev_sensor.py --use-prev-sensor 1 --prev-sensor-channels raw1,raw3,raw6
"""

import os
import sys
import random
from pathlib import Path
from datetime import datetime
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW

from self_detection_raw.data.loader import load_file, extract_features
from self_detection_raw.data.loader_v import extract_features_v
from self_detection_raw.models.mlp_b_v import ModelBV
from self_detection_raw.train.train import SelfDetectionDataset, train_epoch
from self_detection_raw.utils.io import ensure_dir, save_json
from self_detection_raw.utils.metrics import compute_channel_metrics

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------
DATA_DIR = "/home/song/rb10_Proximity/src/self_detection_raw/dataset"

TRAIN_FILES = [
    "1dataset_50_25_1.txt",
    "2dataset_50_25_1.txt",
    "3dataset_50_25_1.txt",

]

VAL_FILES = [
    "4dataset_50_25_1.txt",
]

script_root = Path(__file__).parent.absolute()
OUT_DIR = str(script_root / "model")
NAME_PREFIX = "mlp_vel_prev"

USE_VEL = True
USE_PREV_SENSOR = True
PREV_SENSOR_CHANNELS = ""  # Empty = use all channels.
PREV_SENSOR_STEPS = 2
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

CHANNEL_NAMES = [f"raw{i}" for i in range(1, 9)]
RAW_ERROR_THRESHOLD = 2000.0
CHANNEL_LOSS_WEIGHTS = [1.0] * len(CHANNEL_NAMES)  # Edit here if some channels should matter more.
BEST_METRIC_NAME = "within_2000"
EARLY_STOP_PATIENCE = 15
EARLY_STOP_MIN_DELTA = 0.0
EARLY_STOP_WARMUP_EPOCHS = 10
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


def compute_practical_metrics(preds, targets):
    residuals = targets - preds
    abs_errors = np.abs(residuals)
    sq_errors = residuals ** 2

    channel_metrics = compute_channel_metrics(targets, residuals)
    per_channel_rmse = np.sqrt(np.mean(sq_errors, axis=0))
    per_channel_mae = np.mean(abs_errors, axis=0)
    ranked_indices = np.argsort(per_channel_rmse)

    return {
        "mae": float(np.mean(abs_errors)),
        "rmse": float(np.sqrt(np.mean(sq_errors))),
        "avg_std": float(np.mean(channel_metrics["residual_std"])),
        "within_2000": float(np.mean(abs_errors <= 2000.0)),
        "within_1000": float(np.mean(abs_errors <= 1000.0)),
        "within_5000": float(np.mean(abs_errors <= 5000.0)),
        "within_10000": float(np.mean(abs_errors <= 10000.0)),
        "per_channel_rmse": per_channel_rmse,
        "per_channel_mae": per_channel_mae,
        "best_channel_name": CHANNEL_NAMES[ranked_indices[0]],
        "best_channel_rmse": float(per_channel_rmse[ranked_indices[0]]),
        "worst_channel_name": CHANNEL_NAMES[ranked_indices[-1]],
        "worst_channel_rmse": float(per_channel_rmse[ranked_indices[-1]]),
        "channel_metrics": channel_metrics,
    }


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


def is_better_metric(candidate, current_best, metric_name):
    if current_best is None:
        return True
    if metric_name == "within_2000":
        return candidate > current_best
    return candidate < current_best


def parse_prev_sensor_channels(value):
    if value is None or value == "":
        return list(range(len(CHANNEL_NAMES)))

    indices = []
    for item in value.split(","):
        token = item.strip()
        if not token:
            continue
        if token.startswith("raw"):
            idx = int(token[3:]) - 1
        else:
            idx = int(token)
            if idx >= 1:
                idx -= 1
        if idx < 0 or idx >= len(CHANNEL_NAMES):
            raise ValueError(f"Invalid previous sensor channel specifier: {token}")
        indices.append(idx)

    if not indices:
        raise ValueError("prev_sensor_channels resolved to an empty list")

    return sorted(set(indices))


def extract_joint_and_target(filepath, use_vel, vel_window):
    data = load_file(filepath)
    if use_vel:
        return extract_features_v(data, use_vel=True, vel_window=vel_window)
    return extract_features(data, use_vel=False)


def align_with_previous_sensor(X, Y, use_prev_sensor, prev_sensor_indices, prev_sensor_steps):
    if not use_prev_sensor:
        return X, Y

    if prev_sensor_steps < 1:
        raise ValueError("prev_sensor_steps must be >= 1 when using previous sensor inputs")
    if len(X) <= prev_sensor_steps:
        raise ValueError(
            f"Need at least {prev_sensor_steps + 1} samples per file when using "
            f"{prev_sensor_steps} previous sensor steps"
        )

    prev_sensor_features = []
    for lag in range(1, prev_sensor_steps + 1):
        start = prev_sensor_steps - lag
        end = -lag
        prev_sensor_features.append(Y[start:end, prev_sensor_indices])

    X_curr = X[prev_sensor_steps:]
    Y_curr = Y[prev_sensor_steps:]
    X_aug = np.concatenate([X_curr] + prev_sensor_features, axis=1)
    return X_aug, Y_curr


def load_multiple_files_with_prev(
    filepaths,
    use_vel,
    vel_window,
    use_prev_sensor,
    prev_sensor_indices,
    prev_sensor_steps,
):
    X_list = []
    Y_list = []

    for filepath in filepaths:
        if not os.path.exists(filepath):
            print(f"[WARN] File not found: {filepath}")
            continue

        X, Y = extract_joint_and_target(filepath, use_vel=use_vel, vel_window=vel_window)
        X_aligned, Y_aligned = align_with_previous_sensor(
            X,
            Y,
            use_prev_sensor=use_prev_sensor,
            prev_sensor_indices=prev_sensor_indices,
            prev_sensor_steps=prev_sensor_steps,
        )
        X_list.append(X_aligned)
        Y_list.append(Y_aligned)

    if not X_list:
        return np.array([]), np.array([])

    return np.concatenate(X_list, axis=0), np.concatenate(Y_list, axis=0)


def describe_input_config(use_vel, use_prev_sensor, prev_sensor_indices, prev_sensor_steps):
    if not use_vel and not use_prev_sensor:
        return "joints only"
    if use_vel and not use_prev_sensor:
        return "joints + velocity"
    selected = [CHANNEL_NAMES[idx] for idx in prev_sensor_indices]
    if len(selected) == len(CHANNEL_NAMES):
        return f"joints + velocity + previous sensors ({prev_sensor_steps}-step)"
    return (
        f"joints + velocity + previous sensors ({prev_sensor_steps}-step, "
        + ", ".join(selected)
        + ")"
    )


def main():
    parser = argparse.ArgumentParser(description="Train MLP with optional previous sensor inputs")
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

    parser.add_argument("--use-vel", type=int, default=int(USE_VEL), help="Use velocity features")
    parser.add_argument(
        "--use-prev-sensor",
        type=int,
        default=int(USE_PREV_SENSOR),
        help="Append previous sensor values to the current joint feature input",
    )
    parser.add_argument(
        "--prev-sensor-channels",
        default=PREV_SENSOR_CHANNELS,
        help="Comma-separated previous sensor channels, e.g. raw1,raw3 or 1,3. Empty = all",
    )
    parser.add_argument(
        "--prev-sensor-steps",
        type=int,
        default=PREV_SENSOR_STEPS,
        help="Number of previous sensor steps to append (default: 2)",
    )
    parser.add_argument("--vel-window", type=int, default=VEL_WINDOW, help="Velocity smoothing window size")
    parser.add_argument(
        "--raw-threshold",
        type=float,
        default=RAW_ERROR_THRESHOLD,
        help="Raw-scale Huber threshold in original sensor units",
    )
    parser.add_argument(
        "--best-metric-name",
        default=BEST_METRIC_NAME,
        choices=["within_2000", "rmse", "avg_std"],
        help="Validation metric used to select best checkpoint",
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

    args = parser.parse_args()

    data_dir = args.data_dir
    train_files = args.train_files.split(",") if args.train_files else []
    val_files = args.val_files.split(",") if args.val_files else []
    use_vel = bool(args.use_vel)
    use_prev_sensor = bool(args.use_prev_sensor)
    vel_window = args.vel_window
    prev_sensor_indices = parse_prev_sensor_channels(args.prev_sensor_channels)
    prev_sensor_steps = args.prev_sensor_steps

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
    print(f"Input mode: {describe_input_config(use_vel, use_prev_sensor, prev_sensor_indices, prev_sensor_steps)}")
    if use_prev_sensor:
        selected_names = [CHANNEL_NAMES[idx] for idx in prev_sensor_indices]
        print("Previous sensor channels: " + ", ".join(selected_names))
        print(f"Previous sensor steps: {prev_sensor_steps}")

    X_train, Y_train = load_multiple_files_with_prev(
        train_paths,
        use_vel=use_vel,
        vel_window=vel_window,
        use_prev_sensor=use_prev_sensor,
        prev_sensor_indices=prev_sensor_indices,
        prev_sensor_steps=prev_sensor_steps,
    )

    if val_paths:
        X_val, Y_val = load_multiple_files_with_prev(
            val_paths,
            use_vel=use_vel,
            vel_window=vel_window,
            use_prev_sensor=use_prev_sensor,
            prev_sensor_indices=prev_sensor_indices,
            prev_sensor_steps=prev_sensor_steps,
        )
    else:
        X_val = np.array([]).reshape(0, X_train.shape[1])
        Y_val = np.array([]).reshape(0, Y_train.shape[1])

    if X_train.size == 0 or Y_train.size == 0:
        raise RuntimeError("No training samples available after alignment. Check file paths and flags.")

    print(f"Train Input Shape: {X_train.shape}")
    print(f"Train Target Shape: {Y_train.shape}")
    if use_prev_sensor:
        print(
            f"Alignment note: first {prev_sensor_steps} samples of each file are dropped "
            f"because previous sensor history is unavailable."
        )

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

    # criterion = nn.SmoothL1Loss()
    # criterion = ChannelwiseHuberLoss(delta_norm).to(device)
    criterion = WeightedChannelwiseHuberLoss(delta_norm, channel_loss_weights).to(device)

    try:
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=25, verbose=True)
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
        "val_avg_std": [],
        "val_rmse": [],
        "val_within_2000": [],
    }

    run_config = vars(args).copy()
    run_config["use_vel"] = use_vel
    run_config["use_prev_sensor"] = use_prev_sensor
    run_config["prev_sensor_indices"] = prev_sensor_indices
    run_config["prev_sensor_channel_names"] = [CHANNEL_NAMES[idx] for idx in prev_sensor_indices]
    run_config["prev_sensor_steps"] = prev_sensor_steps
    run_config["input_mode"] = describe_input_config(use_vel, use_prev_sensor, prev_sensor_indices, prev_sensor_steps)
    run_config["normalized_threshold_per_channel"] = delta_norm.tolist()
    run_config["channel_loss_weights"] = channel_loss_weights.tolist()
    run_config["best_metric_name"] = args.best_metric_name
    save_json(run_config, str(Path(run_dir) / "config.json"))

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

        if scheduler and "rmse" in metrics:
            scheduler.step(metrics["rmse"])

        history["train_loss"].append(loss)
        if metrics:
            history["val_avg_std"].append(metrics["avg_std"])
            history["val_rmse"].append(metrics["rmse"])
            history["val_within_2000"].append(metrics["within_2000"])

        if metrics:
            selection_value = metrics[args.best_metric_name]
            if is_better_metric(selection_value, best_metric_value, args.best_metric_name):
                best_metric_value = selection_value
                best_epoch = epoch
                norm_params = train_ds.get_norm_params()
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_metrics": metrics,
                        "normalization": norm_params,
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
                    f" | Val std: {metrics['avg_std']:.2f}"
                    f" | Val rmse: {metrics['rmse']:.2f}"
                    f" | W2000: {metrics['within_2000']:.3f}"
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

    print(
        f"Finished training. Best {args.best_metric_name}: "
        f"{best_metric_value if best_metric_value is not None else 'n/a'}"
    )
    if best_rmse_for_early_stop is not None:
        print(f"Best RMSE for early stopping: {best_rmse_for_early_stop:.2f}")
    if best_epoch is not None:
        print(f"Best epoch: {best_epoch}")
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
