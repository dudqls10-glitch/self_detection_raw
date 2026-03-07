# 데이터 파일 위치 가이드

## 데이터 파일을 어디에 둘 수 있나요?

**어떤 경로든 가능합니다!** `--data_dir` 인자로 지정한 디렉토리와 그 **모든 하위 디렉토리**를 재귀적으로 검색합니다.

## 검색 방식

코드는 `find_files_by_pattern` 함수를 사용하여:
- 지정한 `--data_dir` 디렉토리부터 시작
- `**` 패턴으로 모든 하위 디렉토리를 재귀적으로 검색
- `--glob` 패턴(기본값: `"robot_data_*.txt"`)과 일치하는 파일을 찾음

## 사용 예시

### 예시 1: 특정 디렉토리에 파일들이 있는 경우

```bash
# 데이터 파일 위치: /home/son_rb/rb_ws/src/robotory_rb10_ros2/scripts/
python -m self_detection_raw.train.train \
  --data_dir /home/son_rb/rb_ws/src/robotory_rb10_ros2/scripts \
  --glob "robot_data_*.txt" \
  --out_dir outputs/run_001
```

### 예시 2: 여러 하위 디렉토리에 파일들이 있는 경우

```bash
# 데이터 구조:
# /path/to/data/
#   ├── session1/
#   │   ├── robot_data_20260130_173052.txt
#   │   └── robot_data_20260130_173843.txt
#   └── session2/
#       └── robot_data_20260130_180447.txt

python -m self_detection_raw.train.train \
  --data_dir /path/to/data \
  --glob "robot_data_*.txt" \
  --out_dir outputs/run_001
```

이 경우 모든 하위 디렉토리의 파일들이 자동으로 찾아집니다.

### 예시 3: 현재 워크스페이스의 실제 경로

```bash
# robotory_rb10_ros2/scripts/ 에 있는 파일들 사용
python -m self_detection_raw.train.train \
  --data_dir /home/son_rb/rb_ws/src/robotory_rb10_ros2/scripts \
  --glob "robot_data_*.txt" \
  --out_dir outputs/run_001 \
  --epochs 300 --batch 256
```

## 현재 워크스페이스에서 찾은 데이터 파일들

다음 위치에 `robot_data_*.txt` 파일들이 있습니다:

- `/home/son_rb/rb_ws/src/robotory_rb10_ros2/scripts/robot_data_*.txt` (여러 파일)
- `/home/son_rb/rb_ws/src/self_detection/robot_data_20260127_121928.txt`

## 권장 데이터 구조

```
your_data_directory/
├── robot_data_20260130_173052.txt
├── robot_data_20260130_173843.txt
├── robot_data_20260130_174634.txt
└── robot_data_20260130_180447.txt
```

또는

```
your_data_directory/
├── session1/
│   └── robot_data_*.txt
├── session2/
│   └── robot_data_*.txt
└── session3/
    └── robot_data_*.txt
```

두 경우 모두 `--data_dir your_data_directory`로 지정하면 모든 파일을 찾습니다.

## 파일 패턴 변경

기본값은 `"robot_data_*.txt"`이지만, 다른 패턴도 사용 가능합니다:

```bash
# 모든 .txt 파일
--glob "*.txt"

# 특정 날짜의 파일만
--glob "robot_data_20260130_*.txt"

# 다른 이름 패턴
--glob "sensor_data_*.txt"
```







