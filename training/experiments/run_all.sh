#!/usr/bin/env bash
# 실험 전체 실행: 학습(실험 × seed, 병렬) → batch=1 평가 → 결과 표
# 이미 끝난 run(DONE 파일 있음)은 건너뛰므로, 중간에 끊겨도 다시 실행하면 이어서 돈다.
#
#   SMOKE=1 bash run_all.sh      # 2 epoch짜리로 파이프라인만 확인
#   bash run_all.sh              # 본 실험
set -uo pipefail

WORK=${WORK:-/workspace}
REPO=${REPO:-$WORK/lprnet-pipeline}
LPRNET_SRC=${LPRNET_SRC:-$WORK/LPRNet-master2}
DATA_ROOT=${DATA_ROOT:-$WORK/ALPRData}
OUT=${OUT:-$WORK/runs}
SEEDS=${SEEDS:-"0 1 2"}
# run 하나가 GPU 메모리를 6~7GB 쓴다 (batch 256, AMP). 24GB(RTX 4090)에서 5개는 OOM, 3개까지
PARALLEL=${PARALLEL:-3}
SMOKE=${SMOKE:-0}

export PYTHONPATH="$LPRNET_SRC:$REPO/training/finetune"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
PRETRAINED=$LPRNET_SRC/weights/lprnet_kor.pt

# 이름|train_exp.py 옵션. 각 실험은 A_orig에서 한 가지씩만 바꾼다 (E는 B+C 조합)
EXPERIMENTS=(
  "A_orig|--norm batch --sched orig"
  "B_sample_norm|--norm sample --sched orig"
  "C_cosine|--norm batch --sched cosine"
  "D_cosine_lr3e-4|--norm batch --sched cosine --lr 3e-4"
  "E_sample_cosine|--norm sample --sched cosine"
)

EXTRA=""
if [ "$SMOKE" = 1 ]; then
  EXTRA="--max_epochs 2 --warmup_epochs 1"
  SEEDS=0
  OUT=$OUT-smoke
fi

run_one() {
  local name=$1 seed=$2; shift 2
  local dir=$OUT/$name/s$seed
  if [ -f "$dir/DONE" ]; then
    echo "[skip] $name s$seed (이미 완료)"
    return
  fi
  mkdir -p "$dir"
  echo "[start] $name s$seed  $(date '+%H:%M')"
  if python "$REPO/training/finetune/train_exp.py" --name "$name" --seed "$seed" \
      --data_root "$DATA_ROOT" --pretrained "$PRETRAINED" --out_dir "$OUT" "$@" $EXTRA \
      > "$dir/train.log" 2>&1; then
    echo "[done]  $name s$seed  $(date '+%H:%M')"
  else
    echo "[FAIL]  $name s$seed  → $dir/train.log"
  fi
}

echo "== 학습: 실험 ${#EXPERIMENTS[@]}개 × seed [$SEEDS], 동시에 $PARALLEL개"
for exp in "${EXPERIMENTS[@]}"; do
  name=${exp%%|*}
  flags=${exp#*|}
  for seed in $SEEDS; do
    while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do wait -n; done
    # shellcheck disable=SC2086
    run_one "$name" "$seed" $flags &
  done
done
wait

echo "== 평가 (batch=1)"
EVAL=$OUT/eval
mkdir -p "$EVAL"
models=(--model "base=$PRETRAINED")
for ckdir in "$OUT"/*/s*/ckpt; do
  [ -d "$ckdir" ] || continue
  seed=$(basename "$(dirname "$ckdir")")
  name=$(basename "$(dirname "$(dirname "$ckdir")")")
  for sel in b256 b1; do
    ck=$(ls "$ckdir"/best-$sel-*.ckpt 2>/dev/null | head -1)
    [ -n "$ck" ] && models+=(--model "${name}__${seed}__${sel}=$ck")
  done
  [ -f "$ckdir/last.ckpt" ] && models+=(--model "${name}__${seed}__last=$ckdir/last.ckpt")
done
python "$REPO/training/eval/eval_offline.py" \
  --test_dir "$DATA_ROOT/dataset3/test" --train_dir "$DATA_ROOT/dataset3/train" \
  --out_dir "$EVAL" "${models[@]}"

python "$REPO/training/eval/aggregate.py" "$EVAL" | tee "$OUT/results.md"
echo "== 끝. 결과 표: $OUT/results.md  (Pod를 멈추기 전에 $OUT 을 내려받으세요)"
