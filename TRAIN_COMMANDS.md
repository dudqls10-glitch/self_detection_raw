# 학습 실행 명령어

## 기본 실행 방법

### 1. 가상환경 활성화
```bash
cd /home/son_rb/rb_ws/src/self_detection_raw
source venv1/bin/activate
```

### 2. 학습 실행

#### 방법 1: 빠른 실행 스크립트 (추천)
```bash
# 기본 데이터 디렉토리 사용
python quick_train.py

# 데이터 디렉토리 지정
python quick_train.py /home/son_rb/rb_ws/src/robotory_rb10_ros2/scripts
```

#### 방법 2: 직접 명령어 실행
```bash
python -m self_detection_raw.train.train \
    --data_dir /home/son_rb/rb_ws/src/robotory_rb10_ros2/scripts \
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
```

#### 방법 3: Bash 스크립트 사용
```bash
./train.sh
# 또는
./train.sh /home/son_rb/rb_ws/src/robotory_rb10_ros2/scripts
```

## 주요 파라미터 설명

- `--data_dir`: 데이터 파일이 있는 디렉토리
- `--glob`: 파일 패턴 (기본: "robot_data_*.txt")
- `--out_dir`: 모델 저장 디렉토리
- `--epochs`: 학습 에포크 수 (기본: 300)
- `--batch`: 배치 크기 (기본: 256)
- `--lr`: 학습률 (기본: 1e-3)
- `--hidden`: 모델 hidden 차원 (기본: 128)
- `--head_hidden`: Head hidden 차원 (기본: 64)
- `--dropout`: Dropout 비율 (기본: 0.1)

## 입력/출력

- **입력**: sin/cos(j1..j6) = 12차원 (joint velocity 제거됨)
- **출력**: raw1..raw8 = 8차원

## 결과 저장 위치

학습 완료 후 다음 파일들이 생성됩니다:
- `train/outputs/run_YYYYMMDD_HHMMSS/model.pt`: 학습된 모델
- `train/outputs/run_YYYYMMDD_HHMMSS/norm_params.json`: 정규화 파라미터
- `train/outputs/run_YYYYMMDD_HHMMSS/config.json`: 학습 설정
- `train/outputs/run_YYYYMMDD_HHMMSS/report.json`: 평가 결과
- `train/outputs/run_YYYYMMDD_HHMMSS/history.csv`: 학습 히스토리

## Method B (MLP main + TCN residual)

### 빠른 실행
```bash
python quick_train_tcn.py
```

### 직접 실행 (권장 기본값)
```bash
python -m self_detection_raw.train.train_tcn \
    --data_dir /home/son_rb/rb_ws/src/robotory_rb10_ros2/scripts \
    --glob "robot_data_*.txt" \
    --out_dir train/outputs/run_$(date +%Y%m%d_%H%M%S) \
    --seq_len 32 \
    --stage finetune \
    --lambda_res 1e-3 \
    --tcn_hidden 64 \
    --tcn_kernel 3 \
    --tcn_dilations 1,2,4,8
```

### 2-stage 예시
```bash
# Stage1: main only
python -m self_detection_raw.train.train_tcn --seq_len 32 --stage main_only

# Stage2: residual only (Loss3)
python -m self_detection_raw.train.train_tcn --seq_len 32 --stage res_only --lambda_res 1e-3
```





