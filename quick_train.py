#!/usr/bin/env python3
"""
빠른 학습 실행 스크립트
기본 설정으로 학습을 시작합니다.

사용법:
    python quick_train.py [data_dir]
    
예시:
    python quick_train.py
    python quick_train.py /home/son_rb/rb_ws/src/robotory_rb10_ros2/scripts
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    # 패키지 루트 디렉토리
    script_dir = Path(__file__).parent.absolute()
    package_root = script_dir
    
    # 기본 데이터 디렉토리
    default_data_dir = "/home/son_rb/rb_ws/src/robotory_rb10_ros2/scripts"
    
    # 데이터 디렉토리 확인
    data_dir = sys.argv[1] if len(sys.argv) > 1 else default_data_dir
    
    if not os.path.exists(data_dir):
        print(f"[ERROR] Data directory not found: {data_dir}")
        print(f"Usage: {sys.argv[0]} [data_dir]")
        print(f"Default: {default_data_dir}")
        sys.exit(1)
    
    # 출력 디렉토리 (타임스탬프 포함)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"train/outputs/run_{timestamp}"
    
    print("=" * 60)
    print("Self Detection Raw - Quick Training")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {out_dir}")
    print("=" * 60)
    print()
    
    # 학습 명령어 구성
    cmd = [
        sys.executable, "-m", "self_detection_raw.train.train",
        "--data_dir", data_dir,
        "--glob", "robot_data_*.txt",
        "--out_dir", out_dir,
        "--val_split", "file",
        "--val_ratio", "0.2",
        "--use_vel", "0",  # joint velocities removed
        "--x_norm", "1",
        "--y_norm", "1",
        "--std_floor", "1e-2",
        "--epochs", "300",
        "--batch", "256",
        "--lr", "1e-3",
        "--wd", "1e-4",
        "--hidden", "128",
        "--head_hidden", "64",
        "--dropout", "0.1",
        "--seed", "42",
        "--num_workers", "4"
    ]
    
    print("실행 명령어:")
    print(" ".join(cmd))
    print()
    
    # 학습 실행
    try:
        subprocess.run(cmd, check=True, cwd=package_root)
        print()
        print("=" * 60)
        print("Training completed successfully!")
        print(f"Model saved to: {out_dir}/model.pt")
        print("=" * 60)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Training failed with exit code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user")
        sys.exit(1)


if __name__ == "__main__":
    main()

