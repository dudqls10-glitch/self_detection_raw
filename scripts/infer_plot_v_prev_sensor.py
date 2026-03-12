#!/usr/bin/env python3
"""
Inference and plotting script for models trained with optional previous-sensor input.

Input structure at inference time:
    input_t = [joint_features_t, sensor_{t-1}, sensor_{t-2}, ...]
    target_t = sensor_t

This matches train_mlp_v_prev_sensor.py exactly:
- if previous sensor input is enabled, the first N samples are dropped
- previous sensor channels can be all channels or a selected subset
- previous sensor steps are configurable
- normalization uses the checkpoint's X/Y normalization parameters

Usage:
    python3 scripts/infer_plot_v_prev_sensor.py
    python3 scripts/infer_plot_v_prev_sensor.py --model scripts/model/<run>/model.pt --data dataset/8dataset_100_50.txt
"""

import argparse
import os
import sys
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from self_detection_raw.data.loader import load_file, extract_features
from self_detection_raw.data.loader_v import extract_features_v
from self_detection_raw.models.mlp_b_v import ModelBV

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_DATA_DIR = "/home/song/rb10_Proximity/src/self_detection_raw/dataset"
DEFAULT_INPUT_FILE = "8dataset_100_50.txt"
CHANNEL_NAMES = [f"raw{i}" for i in range(1, 9)]
HARDWARE_BASELINE = 4.0e+07
# ---------------------------------------------------------------------------


def find_latest_model():
    script_dir = Path(__file__).parent
    model_dir = script_dir / "model"
    if not model_dir.exists():
        return None

    files = list(model_dir.glob("**/model.pt"))
    if not files:
        return None

    files.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return str(files[0])


def find_latest_data(base_dir=None):
    if base_dir and os.path.exists(base_dir):
        dataset_dir = Path(base_dir)
    else:
        dataset_dir = Path(__file__).parent.parent / "dataset"

    if not dataset_dir.exists():
        return None

    files = list(dataset_dir.glob("*.txt"))
    if not files:
        return None

    files.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return str(files[0])


def parse_prev_sensor_indices(train_args):
    indices = train_args.get("prev_sensor_indices")
    if indices is not None:
        return [int(idx) for idx in indices]

    channel_spec = train_args.get("prev_sensor_channels", "")
    if not channel_spec:
        return list(range(len(CHANNEL_NAMES)))

    parsed = []
    for item in str(channel_spec).split(","):
        token = item.strip()
        if not token:
            continue
        if token.startswith("raw"):
            idx = int(token[3:]) - 1
        else:
            idx = int(token)
            if idx >= 1:
                idx -= 1
        parsed.append(idx)
    return sorted(set(parsed))


def load_norm_params(checkpoint, norm_path):
    if norm_path and os.path.exists(norm_path):
        with open(norm_path, "r", encoding="utf-8") as f:
            return json.load(f)

    if "normalization" in checkpoint:
        return checkpoint["normalization"]

    default_path = Path(norm_path) if norm_path else None
    if default_path and default_path.exists():
        with open(default_path, "r", encoding="utf-8") as f:
            return json.load(f)

    raise ValueError("Normalization params not found in checkpoint or --norm-path")


def extract_joint_and_target(data, use_vel, vel_window):
    if use_vel:
        return extract_features_v(data, use_vel=True, vel_window=vel_window)
    return extract_features(data, use_vel=False)


def build_inference_inputs(X, Y, use_prev_sensor, prev_sensor_indices, prev_sensor_steps):
    if not use_prev_sensor:
        return X, Y, np.arange(len(Y))

    if prev_sensor_steps < 1:
        raise ValueError("prev_sensor_steps must be >= 1 when using previous sensor inputs")
    if len(X) <= prev_sensor_steps:
        raise ValueError(
            f"Need at least {prev_sensor_steps + 1} samples to use "
            f"{prev_sensor_steps} previous sensor steps"
        )

    prev_sensor_features = []
    for lag in range(1, prev_sensor_steps + 1):
        start = prev_sensor_steps - lag
        end = -lag
        prev_sensor_features.append(Y[start:end, prev_sensor_indices])

    X_curr = X[prev_sensor_steps:]
    Y_curr = Y[prev_sensor_steps:]
    aligned_indices = np.arange(prev_sensor_steps, len(Y))
    X_aug = np.concatenate([X_curr] + prev_sensor_features, axis=1)
    return X_aug, Y_curr, aligned_indices


def resolve_data_path(data_arg):
    if data_arg is None:
        if DEFAULT_INPUT_FILE:
            candidate = (
                DEFAULT_INPUT_FILE
                if os.path.isabs(DEFAULT_INPUT_FILE)
                else os.path.join(DEFAULT_DATA_DIR, DEFAULT_INPUT_FILE)
            )
            if os.path.exists(candidate):
                return candidate
        return find_latest_data(DEFAULT_DATA_DIR)

    if os.path.exists(data_arg):
        return data_arg

    alt_path = os.path.join(DEFAULT_DATA_DIR, data_arg)
    if os.path.exists(alt_path):
        return alt_path

    return data_arg


def main():
    parser = argparse.ArgumentParser(description="Inference and plot for previous-sensor model")
    parser.add_argument("--model", type=str, default=None, help="Path to model.pt (default: latest)")
    parser.add_argument("--data", type=str, default=None, help="Path to input .txt data file")
    parser.add_argument("--norm-path", type=str, default=None, help="Optional norm json path")
    parser.add_argument("--output-plot", type=str, default=None, help="Optional output plot path")
    parser.add_argument("--vel-window", type=int, default=10, help="Fallback velocity window")
    parser.add_argument("--use-hw-baseline", dest="use_hardware_baseline", action="store_true")
    parser.add_argument("--use-mean-baseline", dest="use_hardware_baseline", action="store_false")
    parser.set_defaults(use_hardware_baseline=True)
    args = parser.parse_args()

    print("Starting previous-sensor inference script...")

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

    args.data = resolve_data_path(args.data)
    if args.data is None or not os.path.exists(args.data):
        print("Error: No data file found and --data not specified.")
        return
    print(f"Selected data: {args.data}")

    use_vel = bool(train_args.get("use_vel", True))
    use_prev_sensor = bool(train_args.get("use_prev_sensor", False))
    prev_sensor_indices = parse_prev_sensor_indices(train_args)
    prev_sensor_steps = int(train_args.get("prev_sensor_steps", 1))
    vel_window = int(train_args.get("vel_window", args.vel_window))
    print(f"Using velocity input: {use_vel}")
    print(f"Using previous sensor input: {use_prev_sensor}")
    if use_prev_sensor:
        selected_names = [CHANNEL_NAMES[idx] for idx in prev_sensor_indices]
        print("Previous sensor channels: " + ", ".join(selected_names))
        print(f"Previous sensor steps: {prev_sensor_steps}")
    print(f"Using Velocity Window: {vel_window}")

    norm_params = load_norm_params(checkpoint, args.norm_path)
    X_mean = np.asarray(norm_params["X_mean"], dtype=np.float32)
    X_std = np.asarray(norm_params["X_std"], dtype=np.float32)
    Y_mean = np.asarray(norm_params["Y_mean"], dtype=np.float32)
    Y_std = np.asarray(norm_params["Y_std"], dtype=np.float32)

    print(f"Loading data from {args.data}...")
    raw_data = load_file(args.data)
    X_joint, Y_full = extract_joint_and_target(raw_data, use_vel=use_vel, vel_window=vel_window)
    X_raw, Y_raw, aligned_indices = build_inference_inputs(
        X_joint,
        Y_full,
        use_prev_sensor=use_prev_sensor,
        prev_sensor_indices=prev_sensor_indices,
        prev_sensor_steps=prev_sensor_steps,
    )

    if len(X_raw) == 0:
        print("Error: No samples available after previous-sensor alignment.")
        return

    X_norm = (X_raw - X_mean) / X_std
    X_tensor = torch.from_numpy(X_norm.astype(np.float32)).to(device)

    model = ModelBV(
        in_dim=X_raw.shape[1],
        trunk_hidden=int(train_args.get("hidden", 128)),
        head_hidden=int(train_args.get("head_hidden", 64)),
        out_dim=Y_raw.shape[1],
        dropout=float(train_args.get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Running inference...")
    with torch.no_grad():
        pred_norm = model(X_tensor).cpu().numpy()

    pred = pred_norm * Y_std + Y_mean
    residual = Y_raw - pred

    if args.use_hardware_baseline:
        baseline = np.full(Y_raw.shape[1], HARDWARE_BASELINE, dtype=np.float32)
    else:
        baseline = Y_mean
    compensated = residual + baseline

    print("Plotting results...")
    fig, axes = plt.subplots(4, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()

    t = aligned_indices
    for i in range(8):
        ax = axes[i]
        ax_comp = ax.twinx()

        raw_line = ax.plot(t, Y_raw[:, i], label="raw", color="C0", alpha=0.3, zorder=1)
        pred_line = ax.plot(t, pred[:, i], label="pred", color="C1", alpha=0.5, zorder=2)
        comp_line = ax_comp.plot(t, compensated[:, i], label="compensated", color="C2", alpha=0.4, zorder=3)
        ax_comp.tick_params(axis="y", colors="C2")
        ax_comp.spines["right"].set_color("C2")

        resid_std = np.std(residual[:, i])
        comp_std = np.std(compensated[:, i])
        title = f"raw{i+1} | ResStd: {resid_std:.0f} | CompStd: {comp_std:.0f}"
        if use_prev_sensor:
            title += f" | uses prev x{prev_sensor_steps}"
        ax.set_title(title, fontsize=10)
        if i == 0:
            lines = raw_line + pred_line + comp_line
            labels = [line.get_label() for line in lines]
            ax.legend(lines, labels, loc="upper right")
        ax.grid(True, alpha=0.3)

    file_note = os.path.basename(args.data)
    if use_prev_sensor:
        file_note += f" (first {prev_sensor_steps} samples dropped for previous sensor history)"
    fig.suptitle(f"Inference results on {file_note}")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    if args.output_plot:
        out_path = args.output_plot
    else:
        out_path = os.path.join(os.path.dirname(args.model), "inference_plot_v_prev_sensor.png")

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

    print("Compensated stats (mean, std) for each channel:")
    for i in range(8):
        print(f" raw{i+1}: {compensated[:, i].mean():.3f}, {compensated[:, i].std():.3f}")


if __name__ == "__main__":
    main()
