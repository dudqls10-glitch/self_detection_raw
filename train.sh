#!/bin/bash
# 학습 실행 스크립트
# Usage: ./train.sh [data_dir]

set -e

# 스크립트 위치에서 패키지 루트로 이동
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 가상환경 활성화 (venv1이 있는 경우)
if [ -d "venv1" ]; then
    source venv1/bin/activate
    echo "[INFO] Activated venv1"
fi

# 데이터 디렉토리 설정
if [ -n "$1" ]; then
    DATA_DIR="$1"
else
    # 기본 데이터 디렉토리
    DATA_DIR="/home/son_rb/rb_ws/src/robotory_rb10_ros2/scripts"
fi

echo "=" * 60
echo "Self Detection Raw - Training Script"
echo "=" * 60
echo "Data directory: $DATA_DIR"
echo ""

# 데이터 파일 확인
if [ ! -d "$DATA_DIR" ]; then
    echo "[ERROR] Data directory not found: $DATA_DIR"
    echo "Usage: $0 [data_dir]"
    exit 1
fi

# 학습 실행
python -m self_detection_raw.train.train \
    --data_dir "$DATA_DIR" \
    --glob "robot_data_*.txt" \
    --out_dir train/outputs/run_$(date +%Y%m%d_%H%M%S) \
    --val_split file \
    --val_ratio 0.2 \
    --use_vel 0 \
    --x_norm 1 \
    --y_norm 1 \
    --std_floor 1e-2 \
    --epochs 300 \
    --batch 256 \
    --lr 1e-3 \
    --wd 1e-4 \
    --hidden 128 \
    --head_hidden 64 \
    --dropout 0.1 \
    --seed 42 \
    --num_workers 4

echo ""
echo "=" * 60
echo "Training completed!"
echo "=" * 60






