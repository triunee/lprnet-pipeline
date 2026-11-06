#!/usr/bin/env bash
# GPU 서버(RunPod 등)에서 한 번만 실행: 패키지 설치 + 데이터·환경 점검 + base 재측정
set -euo pipefail

WORK=${WORK:-/workspace}
REPO=${REPO:-$WORK/lprnet-pipeline}
LPRNET_SRC=${LPRNET_SRC:-$WORK/LPRNet-master2}
DATA_ROOT=${DATA_ROOT:-$WORK/ALPRData}

echo "[1/4] 패키지 설치 (torch는 이미지에 포함된 것 사용)"
# lightning 버전은 05-11 Colab 학습과 같게 맞춤 (체크포인트의 pytorch-lightning_version)
pip install -q "lightning==2.6.1" imutils opencv-python-headless pyyaml tqdm rich onnxruntime

echo "[2/4] GPU 확인"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch, lightning; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| lightning', lightning.__version__)"
echo "vCPU: $(nproc)  (PARALLEL × 4 data worker가 이 수를 크게 넘지 않게)"

echo "[3/4] 데이터 점검"
for s in train val test; do
  echo "  dataset3/$s: $(ls "$DATA_ROOT/dataset3/$s" | wc -l)장"
done
python - "$DATA_ROOT/dataset3" <<'EOF'
import os, sys, unicodedata
root = sys.argv[1]
nfd = [f for s in ("train", "val", "test") for f in os.listdir(os.path.join(root, s))
       if f != unicodedata.normalize("NFC", f)]
print(f"  NFC가 아닌 파일명: {len(nfd)}개" + ("  ← 0이어야 합니다. 복사 중 이름이 바뀌었는지 확인하세요" if nfd else ""))
EOF
test -f "$LPRNET_SRC/weights/lprnet_kor.pt" && test -f "$LPRNET_SRC/lprnet/lprnet.py"

echo "[4/4] base 재측정 (맥 CPU 기준 37.47%, 550/1468 근처면 환경 정상)"
export PYTHONPATH="$LPRNET_SRC:$REPO/training/finetune"
python "$REPO/training/eval/eval_offline.py" \
  --test_dir "$DATA_ROOT/dataset3/test" \
  --model "base=$LPRNET_SRC/weights/lprnet_kor.pt" \
  --out_dir "$WORK/runs/setup_check"
