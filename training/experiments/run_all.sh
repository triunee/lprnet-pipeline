#!/usr/bin/env bash
# 실험 전체 실행: 학습(실험 × seed, 병렬) → batch=1 평가 → 결과 표
# 실험 목록은 rounds/<ROUND>.sh 에 있다. 이미 끝난 run(DONE 파일 있음)은 건너뛰므로,
# 중간에 끊겨도 다시 실행하면 이어서 돈다.
#
#   ROUND=round2 SMOKE=1 bash run_all.sh   # 2 epoch짜리로 파이프라인만 확인
#   ROUND=round2 bash run_all.sh           # 본 실험
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)

ROUND=${ROUND:-round2}
WORK=${WORK:-/workspace}
REPO=${REPO:-$WORK/lprnet-pipeline}
LPRNET_SRC=${LPRNET_SRC:-$WORK/LPRNet-master2}
DATA_ROOT=${DATA_ROOT:-$WORK/ALPRData}
OUT=${OUT:-$WORK/runs_$ROUND}
SEEDS=${SEEDS:-"0 1 2"}
# run 하나가 GPU 메모리를 6~7GB 쓴다 (batch 256, AMP). 24GB(RTX 4090)에서 5개는 OOM, 3개까지
PARALLEL=${PARALLEL:-3}
SMOKE=${SMOKE:-0}

# shellcheck source=rounds/round2.sh
source "$HERE/rounds/$ROUND.sh"   # DATASET, CKPT_SELECTIONS, EXPERIMENTS

export PYTHONPATH="$LPRNET_SRC:$REPO/training/finetune"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
PRETRAINED=$LPRNET_SRC/weights/lprnet_kor.pt
TEST_DIR=$DATA_ROOT/$DATASET/test

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
      --data_root "$DATA_ROOT" --dataset "$DATASET" --pretrained "$PRETRAINED" --out_dir "$OUT" "$@" $EXTRA \
      > "$dir/train.log" 2>&1; then
    echo "[done]  $name s$seed  $(date '+%H:%M')"
  else
    echo "[FAIL]  $name s$seed  → $dir/train.log"
  fi
}

echo "== $ROUND 학습: 실험 ${#EXPERIMENTS[@]}개 × seed [$SEEDS], 동시에 $PARALLEL개, 데이터 $DATASET"
for exp in "${EXPERIMENTS[@]}"; do
  IFS='|' read -r name flags _ <<< "$exp"
  for seed in $SEEDS; do
    while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do wait -n; done
    # shellcheck disable=SC2086
    run_one "$name" "$seed" $flags &
  done
done
wait

echo "== 평가 (batch=1, $TEST_DIR)"
EVAL=$OUT/eval
mkdir -p "$EVAL"
evaluate() {  # evaluate <--unfold 값> <--model ...>
  local unfold=$1; shift
  python "$REPO/training/eval/eval_offline.py" --test_dir "$TEST_DIR" \
    --train_dir "$DATA_ROOT/$DATASET/train" --out_dir "$EVAL" --unfold "$unfold" "$@"
}
evaluate none --model "base=$PRETRAINED"
for exp in "${EXPERIMENTS[@]}"; do
  IFS='|' read -r name _ unfolds <<< "$exp"
  for unfold in $unfolds; do
    # select는 같은 체크포인트를 배치 정보 없이 평가한 것이라 이름 뒤에 _select를 붙인다
    suffix=""; [ "$unfold" = select ] && suffix="_select"
    models=()
    for seed in $SEEDS; do
      ckdir=$OUT/$name/s$seed/ckpt
      [ -d "$ckdir" ] || continue
      for sel in $CKPT_SELECTIONS; do
        if [ "$sel" = last ]; then ck=$ckdir/last.ckpt; else ck=$(ls "$ckdir"/best-$sel-*.ckpt 2>/dev/null | head -1); fi
        [ -n "$ck" ] && [ -f "$ck" ] && models+=(--model "${name}__s${seed}__${sel}${suffix}=$ck")
      done
    done
    [ ${#models[@]} -gt 0 ] && evaluate "$unfold" "${models[@]}"
  done
done

python "$REPO/training/eval/aggregate.py" "$EVAL" | tee "$OUT/results.md"
echo "== 끝. 결과 표: $OUT/results.md  (Pod를 멈추기 전에 $OUT 을 내려받으세요)"
