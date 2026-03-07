# Self-Detection Raw Baseline Compensation (Model B)

손/물체가 없는 데이터로 정전용량 센서의 `raw1~raw8` baseline을 학습하여, 실시간에서 `residual = raw - b_hat(q, qdot)` 로 접근(손) 신호를 강화합니다.

## 설치

### 1. 가상환경 생성 및 패키지 설치

```bash
cd /home/son_rb/rb_ws/src/self_detection_raw

# 가상환경 생성 및 자동 설치
./setup_venv.sh

# 가상환경 활성화
source venv1/bin/activate

# 패키지 설치 (개발 모드)
pip install -e .
```

또는 수동으로:

```bash
# 가상환경 생성
python3 -m venv venv1
source venv1/bin/activate

# 패키지 설치
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### 2. 가상환경 사용

**활성화:**
```bash
cd /home/son_rb/rb_ws/src/self_detection_raw
source venv1/bin/activate
```

**비활성화:**
```bash
deactivate
```

## 데이터 파일 위치

데이터 파일은 **어떤 경로든** 둘 수 있습니다. `--data_dir`로 지정한 디렉토리와 **모든 하위 디렉토리**를 재귀적으로 검색합니다.

**예시:**
```bash
# 예시 1: 특정 디렉토리
--data_dir /home/son_rb/rb_ws/src/robotory_rb10_ros2/scripts

# 예시 2: 여러 하위 디렉토리 포함
--data_dir /path/to/data  # 하위 폴더의 모든 robot_data_*.txt 파일도 자동 검색
```

자세한 내용은 [DATA_LOCATION.md](DATA_LOCATION.md)를 참고하세요.

## 사용법

### 학습

```bash
python -m self_detection_raw.train.train \
  --data_dir /path/to/logs \
  --glob "robot_data_*.txt" \
  --out_dir outputs/run_001 \
  --val_split file --val_ratio 0.2 \
  --use_vel 1 \
  --x_norm 1 --y_norm 1 \
  --std_floor 1e-2 \
  --epochs 300 --batch 256 \
  --lr 1e-3 --wd 1e-4 \
  --hidden 128 --head_hidden 64 --dropout 0.1 \
  --seed 42
```

**실제 사용 예시 (현재 워크스페이스):**
```bash
python -m self_detection_raw.train.train \
  --data_dir /home/son_rb/rb_ws/src/robotory_rb10_ros2/scripts \
  --glob "robot_data_*.txt" \
  --out_dir outputs/run_001 \
  --epochs 300 --batch 256
```

학습 후 생성되는 파일:
- `outputs/run_001/model.pt`: 모델 체크포인트
- `outputs/run_001/norm_params.json`: 정규화 파라미터
- `outputs/run_001/config.json`: 학습 설정
- `outputs/run_001/report.json`: 평가 리포트
- `outputs/run_001/history.csv`: 학습 히스토리

### 평가

```bash
python -m self_detection_raw.train.eval \
  --model outputs/run_001/model.pt \
  --norm outputs/run_001/norm_params.json \
  --data_dir /path/to/logs \
  --glob "robot_data_*.txt" \
  --split file --val_ratio 0.2
```

### 추론 (오프라인 residual 생성)

```bash
python -m self_detection_raw.infer.infer \
  --model outputs/run_001/model.pt \
  --norm outputs/run_001/norm_params.json \
  --input robot_data_20260130_173052.txt \
  --output residual_20260130_173052.csv
```

출력 CSV 컬럼:
- `timestamp`: 타임스탬프
- `raw1..raw8`: 원본 센서 값
- `pred_raw1..pred_raw8`: 모델 예측 값
- `residual1..residual8`: 잔차 (raw - pred_raw)
- `j1..j6`: 관절 위치 (degrees)
- `jv1..jv6`: 관절 속도 (deg/s)

### 추론 (시각화 포함)

```bash
python -m self_detection_raw.infer.infer_visualize \
  --model outputs/run_001/model.pt \
  --norm outputs/run_001/norm_params.json \
  --input robot_data_20260130_173052.txt \
  --out_dir outputs/run_001/inference
```

생성되는 시각화:
- `inference_time_series.png`: 시간에 따른 센서 값 변화 (실제값, 예측값, 잔차)
- `inference_std_comparison.png`: 채널별 STD 비교 (보정 전/후), 개선율, 목표 달성 여부
- `inference_residual_dist.png`: 잔차 분포 히스토그램
- `inference_scatter.png`: 실제값 vs 예측값 scatter plot

### 실시간 추론 (ROS2 노드)

**사전 요구사항:**
- ROS2 설치 및 워크스페이스 설정
- `rclpy` 패키지 설치: `pip install rclpy` (venv1에 설치)

**Launch 파일 사용:**
```bash
# 기본 사용 (최신 모델 자동 탐지)
ros2 launch self_detection_raw realtime_infer.launch.py

# 모델 파일 지정
ros2 launch self_detection_raw realtime_infer.launch.py \
  model_file:=outputs/run_001/model.pt

# 모든 파라미터 지정
ros2 launch self_detection_raw realtime_infer.launch.py \
  model_file:=outputs/run_001/model.pt \
  norm_file:=outputs/run_001/norm_params.json \
  use_vel:=true \
  use_hardware_baseline:=true \
  log_rate:=100.0
```

**직접 실행:**
```bash
python -m self_detection_raw.infer.realtime_infer \
  --ros-args \
  -p model_file:=outputs/run_001/model.pt \
  -p norm_file:=outputs/run_001/norm_params.json \
  -p use_vel:=true \
  -p use_hardware_baseline:=true \
  -p log_rate:=100.0
```

**토픽:**
- **구독:**
  - `/raw_distance{i+1}` (Range): 원본 센서 값 (i=0..7)
  - `/joint_states` (JointState): 관절 상태 (position, velocity)
- **발행:**
  - `/compensated_raw_distance{i+1}` (Range): 보상된 센서 값 (i=0..7)

**보상 계산:**
- `compensated = actual - predicted + baseline`
- `baseline`: 4e+7 (하드웨어 baseline, `main.c` 참조) 또는 학습 데이터 평균

**로그 파일:**
- 위치: `~/rb10_Proximity/logs/compensated_raw_{model_name}_{timestamp}.txt`
- 형식: `timestamp j1..j6 jv1..jv6 raw1..raw8 comp1..comp8 pred1..pred8`

### Method B (MLP main + TCN residual)

학습:
```bash
python -m self_detection_raw.train.train_tcn \
  --data_dir /home/son_rb/rb_ws/src/robotory_rb10_ros2/scripts \
  --glob "robot_data_*.txt" \
  --seq_len 32 \
  --stage finetune \
  --lambda_res 1e-3
```

오프라인 추론:
```bash
python -m self_detection_raw.infer.infer_tcn \
  --model train/outputs/run_YYYYMMDD_HHMMSS/model.pt \
  --input robot_data_YYYYMMDD_HHMMSS.txt \
  --seq_len 32
```

실시간 추론:
```bash
ros2 launch self_detection_raw realtime_infer_tcn.launch.py \
  model_file:=train/outputs/run_YYYYMMDD_HHMMSS/model.pt \
  seq_len:=32
```

## 모델 구조

- **입력**: sin/cos(j1..j6) + jv1..jv6 (18D)
  - 관절 각도는 deg → rad 변환 후 sin/cos 적용
- **출력**: raw1..raw8 (8D)
- **아키텍처**: Shared trunk + Per-channel heads (Model B)
  - Trunk: Linear(18, 128) → ReLU → Dropout(0.1) → Linear(128, 128) → ReLU → Dropout(0.1)
  - Heads: 8개 독립 head, 각각 Linear(128, 64) → ReLU → Linear(64, 1)

## 데이터 포맷

입력 파일: `robot_data_YYYYMMDD_HHMMSS.txt`

예상 컬럼 (37개):
1. `timestamp`
2. `j1..j6`: joint position (degrees)
3. `jv1..jv6`: joint velocity (deg/s) - 없으면 0으로 채움
4. `prox1..prox8`: proximity 센서
5. `raw1..raw8`: 정전용량 센서 raw 값 (학습 타겟)
6. `tof1..tof8`: ToF 센서

주석 라인 (`#`로 시작)은 자동으로 스킵됩니다.

## 특징

- ✅ Robust parser: 깨진 라인 자동 처리, 토큰 클린
- ✅ 정규화: Y 필수, X 권장 (std_floor 포함)
- ✅ File-level split: 시간 순서 유지
- ✅ Welford 알고리즘: 온라인 통계 계산
- ✅ SmoothL1 Loss: Outlier에 강건
- ✅ Gradient clipping: 학습 안정화
