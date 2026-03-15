#!/usr/bin/env python3
"""
Inference and plotting for the base + bounded-correction model.

Plots:
    - raw target
    - base prediction
    - final prediction
    - compensated signal (right axis)
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from self_detection_raw.data.loader import load_file
from self_detection_raw.data.loader_v import smooth_data
from self_detection_raw.models.mlp_b_v import ModelBV

# ---------------------------------------------------------------------------
# Editable defaults
# ---------------------------------------------------------------------------
DEFAULT_DATA_DIR = "/home/song/rb10_Proximity/src/self_detection_raw/dataset"
DEFAULT_INPUT_FILE = "8dataset_100_50.txt"
# ---------------------------------------------------------------------------


def find_latest_model():
    script_dir = Path(__file__).parent
    model_dir = script_dir / "model"
    if not model_dir.exists():
        return None
    files = list(model_dir.glob("mlp_vel_base_corr_*/model.pt"))
    if not files:
        files = list(model_dir.glob("**/model.pt"))
    if not files:
        return None
    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return str(files[0])


def find_latest_data(base_dir=None):
    dataset_dir = Path(base_dir) if base_dir and os.path.exists(base_dir) else Path(__file__).parent.parent / "dataset"
    if not dataset_dir.exists():
        dataset_dir = Path("dataset")
    if not dataset_dir.exists():
        return None
    files = list(dataset_dir.glob("*.txt"))
    if not files:
        return None
    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return str(files[0])


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


def load_norm_params(args, checkpoint):
    if args.norm_path and os.path.exists(args.norm_path):
        with open(args.norm_path, "r", encoding="utf-8") as f:
            return json.load(f)
    if "normalization" in checkpoint:
        return checkpoint["normalization"]
    default_norm = os.path.join(os.path.dirname(args.model), "norm_params.json")
    if os.path.exists(default_norm):
        with open(default_norm, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def main():
    parser = argparse.ArgumentParser(description="Inference and plot for base + correction self-detection model")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--norm-path", type=str, default=None)
    parser.add_argument("--output-plot", type=str, default=None)
    parser.add_argument("--vel-window", type=int, default=10)
    parser.add_argument("--use-hw-baseline", dest="use_hardware_baseline", action="store_true")
    parser.add_argument("--use-mean-baseline", dest="use_hardware_baseline", action="store_false")
    parser.set_defaults(use_hardware_baseline=True)
    args = parser.parse_args()

    print("Starting base-correction inference script...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model is None:
        args.model = find_latest_model()
        if args.model is None:
            print("Error: No model found in scripts/model/ and --model not specified.")
            return
        print(f"Auto-selected latest model: {args.model}")

    if not os.path.exists(args.model):
        print(f"Error: Model file not found {args.model}")
        return

    print(f"Loading model from {args.model}...")
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    train_args = checkpoint.get("args", {})
    use_vel = bool(train_args.get("use_vel", True))
    vel_window = int(train_args.get("vel_window", args.vel_window))
    base_in_dim = int(train_args["base_in_dim"])
    corr_in_dim = int(train_args["corr_in_dim"])
    correction_limit_norm = np.asarray(train_args["correction_limit_norm_per_channel"], dtype=np.float32)

    if args.data is None:
        if DEFAULT_INPUT_FILE:
            candidate = DEFAULT_INPUT_FILE if os.path.isabs(DEFAULT_INPUT_FILE) else os.path.join(DEFAULT_DATA_DIR, DEFAULT_INPUT_FILE)
            if os.path.exists(candidate):
                args.data = candidate
        if args.data is None:
            args.data = find_latest_data(DEFAULT_DATA_DIR)
        if args.data is None:
            print("Error: No data file found in dataset/ and --data not specified.")
            return
        print(f"Selected data: {args.data}")

    norm_params = load_norm_params(args, checkpoint)
    if norm_params is None:
        print("Error: Normalization params could not be loaded.")
        return

    x_mean = np.asarray(norm_params["X_mean"], dtype=np.float32)
    x_std = np.asarray(norm_params["X_std"], dtype=np.float32)
    y_mean = np.asarray(norm_params["Y_mean"], dtype=np.float32)
    y_std = np.asarray(norm_params["Y_std"], dtype=np.float32)

    print(f"Using velocity input: {use_vel}")
    print(f"Using Velocity Window: {vel_window}")
    print("Correction history: q_(t-1), q_(t-2), dq_t, dq_(t-1)")
    print(f"Loading data from {args.data}...")

    raw_data = load_file(args.data)
    x_raw, y_raw, _, _ = extract_base_correction_features(
        raw_data,
        use_vel=use_vel,
        vel_window=vel_window,
    )
    if x_raw.size == 0:
        print("Error: Not enough samples for 2-step history alignment.")
        return

    x_norm = (x_raw - x_mean) / x_std
    x_tensor = torch.FloatTensor(x_norm).to(device)

    model = BaseCorrectionModel(
        base_in_dim=base_in_dim,
        corr_in_dim=corr_in_dim,
        out_dim=y_raw.shape[1],
        trunk_hidden=int(train_args.get("hidden", 128)),
        head_hidden=int(train_args.get("head_hidden", 64)),
        dropout=float(train_args.get("dropout", 0.1)),
        corr_hidden=int(train_args.get("corr_hidden", 32)),
        corr_dropout=float(train_args.get("corr_dropout", 0.05)),
        correction_limit_norm=correction_limit_norm,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Running inference...")
    with torch.no_grad():
        pred_norm, base_pred_norm, correction_norm = model(x_tensor)
        pred_norm = pred_norm.cpu().numpy()
        base_pred_norm = base_pred_norm.cpu().numpy()
        correction_norm = correction_norm.cpu().numpy()

    pred = pred_norm * y_std + y_mean
    base_pred = base_pred_norm * y_std + y_mean
    correction_raw = correction_norm * y_std
    residual = y_raw - pred

    hardware_baseline = 4.0e7
    baseline = np.full(y_raw.shape[1], hardware_baseline, dtype=np.float32) if args.use_hardware_baseline else y_mean
    compensated = residual + baseline

    print("Plotting results...")
    fig, axes = plt.subplots(4, 2, figsize=(15, 10), sharex=True)
    axes = axes.flatten()
    t = np.arange(len(y_raw))

    for i in range(8):
        ax = axes[i]
        ax_comp = ax.twinx()

        raw_line = ax.plot(t, y_raw[:, i], label="raw", color="C0", alpha=0.28, zorder=1)
        base_line = ax.plot(t, base_pred[:, i], label="base", color="C3", alpha=0.5, zorder=2)
        pred_line = ax.plot(t, pred[:, i], label="pred", color="C1", alpha=0.7, zorder=3)
        comp_line = ax_comp.plot(t, compensated[:, i], label="compensated", color="C2", alpha=0.4, zorder=4)

        ax_comp.tick_params(axis="y", colors="C2")
        ax_comp.spines["right"].set_color("C2")

        resid_std = np.std(residual[:, i])
        corr_abs = np.mean(np.abs(correction_raw[:, i]))
        ax.set_title(
            f"raw{i+1} | ResStd: {resid_std:.0f} | Mean|corr|: {corr_abs:.0f}",
            fontsize=10,
        )
        if i == 0:
            lines = raw_line + base_line + pred_line + comp_line
            labels = [line.get_label() for line in lines]
            ax.legend(lines, labels, loc="upper right")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Base + correction inference on {os.path.basename(args.data)}")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_path = args.output_plot or os.path.join(os.path.dirname(args.model), "inference_plot_v_base_corr.png")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving plot to {out_path}...")
    plt.savefig(out_path)
    plt.close()

    if os.path.exists(out_path):
        print("\n" + "=" * 60)
        print("SUCCESS: Plot generated successfully!")
        print(f"Location: {os.path.abspath(out_path)}")
        print("=" * 60 + "\n")
    else:
        print(f"Error: Failed to save plot to {out_path}")

    print("Per-channel correction stats (mean abs, max abs):")
    for i in range(8):
        print(
            f" raw{i+1}: "
            f"{np.mean(np.abs(correction_raw[:, i])):.3f}, "
            f"{np.max(np.abs(correction_raw[:, i])):.3f}"
        )


if __name__ == "__main__":
    main()
