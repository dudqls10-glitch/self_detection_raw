#!/bin/bash
# 가상환경 설정 스크립트

set -e  # 에러 발생 시 스크립트 중단

VENV_NAME="venv1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "Self-Detection Raw 가상환경 설정"
echo "=========================================="
echo ""

# 가상환경 생성
if [ -d "$SCRIPT_DIR/$VENV_NAME" ]; then
    echo "⚠️  가상환경 '$VENV_NAME'이 이미 존재합니다."
    read -p "기존 가상환경을 삭제하고 새로 만들까요? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "기존 가상환경 삭제 중..."
        rm -rf "$SCRIPT_DIR/$VENV_NAME"
    else
        echo "기존 가상환경을 사용합니다."
        source "$SCRIPT_DIR/$VENV_NAME/bin/activate"
        echo ""
        echo "✅ 가상환경 활성화 완료!"
        echo ""
        echo "다음 명령어로 가상환경을 활성화하세요:"
        echo "  source $SCRIPT_DIR/$VENV_NAME/bin/activate"
        exit 0
    fi
fi

echo "가상환경 생성 중: $VENV_NAME"
python3 -m venv "$SCRIPT_DIR/$VENV_NAME"

# 가상환경 활성화
echo "가상환경 활성화 중..."
source "$SCRIPT_DIR/$VENV_NAME/bin/activate"

# pip 업그레이드
echo "pip 업그레이드 중..."
pip install --upgrade pip

# requirements.txt 설치
if [ -f "$SCRIPT_DIR/requirements.txt" ]; then
    echo ""
    echo "필요한 패키지 설치 중..."
    echo "이 작업은 몇 분이 걸릴 수 있습니다..."
    pip install -r "$SCRIPT_DIR/requirements.txt"
else
    echo "⚠️  requirements.txt를 찾을 수 없습니다."
    echo "기본 패키지만 설치합니다..."
    pip install numpy torch tqdm pandas matplotlib
fi

echo ""
echo "=========================================="
echo "✅ 가상환경 설정 완료!"
echo "=========================================="
echo ""
echo "가상환경 활성화 방법:"
echo "  cd $SCRIPT_DIR"
echo "  source $VENV_NAME/bin/activate"
echo ""
echo "가상환경 비활성화:"
echo "  deactivate"
echo ""







