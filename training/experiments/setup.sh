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

echo "[3/4] 데이터 점검 (1차: dataset3, 2차: dataset3_raw — 올린 것만)"
DATASETS=""
for d in dataset3 dataset3_raw; do [ -d "$DATA_ROOT/$d" ] && DATASETS="$DATASETS $d"; done
[ -n "$DATASETS" ] || { echo "  $DATA_ROOT 에 dataset3 / dataset3_raw 가 없습니다"; exit 1; }
for d in $DATASETS; do
  echo "  $d: train $(ls "$DATA_ROOT/$d/train" | wc -l) / val $(ls "$DATA_ROOT/$d/val" | wc -l) / test $(ls "$DATA_ROOT/$d/test" | wc -l)장 (6714 / 1474 / 1468)"
done
python - "$DATA_ROOT" $DATASETS <<'EOF'
import csv, os, sys, unicodedata
root, datasets = sys.argv[1], sys.argv[2:]
for d in datasets:
    files = {(s, f) for s in ("train", "val", "test") for f in os.listdir(os.path.join(root, d, s))}
    nfd = [f for _, f in files if f != unicodedata.normalize("NFC", f)]
    print(f"  {d}: NFC가 아닌 파일명 {len(nfd)}개" + ("  ← 0이어야 합니다. 복사 중 이름이 바뀌었는지 확인하세요" if nfd else ""))
    if d.endswith("_raw"):
        rows = list(csv.DictReader(open(os.path.join(root, d, "manifest.csv"), encoding="utf-8")))
        missing = files - {(r["split"], r["file"]) for r in rows}
        two = sum(r["layout"] == "two_line" for r in rows)
        print(f"  {d}/manifest.csv: {len(rows)}줄, 두 줄 번호판 {two}장 (1541), 배치 정보 없는 파일 {len(missing)}개 (0)")
EOF
test -f "$LPRNET_SRC/weights/lprnet_kor.pt" && test -f "$LPRNET_SRC/lprnet/lprnet.py"

echo "[4/4] base 재측정 (맥 CPU 기준 dataset3 37.47% = 550/1468, dataset3_raw 38.08% = 559/1468 근처면 환경 정상)"
export PYTHONPATH="$LPRNET_SRC:$REPO/training/finetune"
# 두 값이 조금 다른 건 dataset3가 100x50으로 줄인 뒤 JPEG로 다시 저장한 이미지라서다
for d in $DATASETS; do
  python "$REPO/training/eval/eval_offline.py" \
    --test_dir "$DATA_ROOT/$d/test" \
    --model "base=$LPRNET_SRC/weights/lprnet_kor.pt" \
    --out_dir "$WORK/runs/setup_check/$d"
done
