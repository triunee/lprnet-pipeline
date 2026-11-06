# 1차 (2026-11-06): 누수 없는 dataset3에서 정규화·LR 스케줄을 하나씩 바꿔 본다
# 이름|train_exp.py 옵션|평가 방식(eval_offline.py --unfold)
DATASET=dataset3
CKPT_SELECTIONS="b256 b1 last"
EXPERIMENTS=(
  "A_orig|--norm batch --sched orig|none"
  "B_sample_norm|--norm sample --sched orig|none"
  "C_cosine|--norm batch --sched cosine|none"
  "D_cosine_lr3e-4|--norm batch --sched cosine --lr 3e-4|none"
  "E_sample_cosine|--norm sample --sched cosine|none"
)
