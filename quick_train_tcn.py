#!/usr/bin/env python3
"""Quick launcher for Method B (MLP main + TCN residual) training."""

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def main():
    script_dir = Path(__file__).parent.absolute()
    package_root = script_dir

    default_data_dir = "/home/son_rb/rb_ws/src/robotory_rb10_ros2/scripts"
    data_dir = sys.argv[1] if len(sys.argv) > 1 else default_data_dir

    if not os.path.exists(data_dir):
        print(f"[ERROR] Data directory not found: {data_dir}")
        print(f"Usage: {sys.argv[0]} [data_dir]")
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"train/outputs/run_{timestamp}"

    cmd = [
        sys.executable,
        "-m",
        "self_detection_raw.train.train_tcn",
        "--data_dir", data_dir,
        "--glob", "robot_data_*.txt",
        "--out_dir", out_dir,
        "--val_split", "file",
        "--val_ratio", "0.2",
        "--use_vel", "0",
        "--seq_len", "32",
        "--stride", "1",
        "--x_norm", "1",
        "--y_norm", "1",
        "--std_floor", "1e-2",
        "--epochs", "300",
        "--batch", "256",
        "--lr", "1e-3",
        "--wd", "1e-4",
        "--stage", "finetune",
        "--lambda_res", "1e-3",
        "--hidden", "128",
        "--head_hidden", "64",
        "--tcn_hidden", "64",
        "--tcn_kernel", "3",
        "--tcn_dilations", "1,2,4,8",
        "--dropout", "0.1",
        "--seed", "42",
        "--num_workers", "4",
    ]

    print("=" * 60)
    print("Self Detection Raw - Quick Training (MLP + TCN Residual)")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {out_dir}")
    print("Command:")
    print(" ".join(cmd))
    print("=" * 60)

    try:
        subprocess.run(cmd, check=True, cwd=package_root)
        print(f"[OK] Training completed. Model: {out_dir}/model.pt")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Training failed: exit {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("[INFO] Training interrupted")
        sys.exit(1)


if __name__ == "__main__":
    main()
