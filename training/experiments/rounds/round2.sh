# 2차: 1차 최고 설정(E: 샘플별 정규화 + cosine) 위에서 남은 오답(두 줄 지역 화물차 윗줄 숫자)을 겨냥한다.
# 원본 크기 데이터셋(dataset3_raw, 분할·파일명은 dataset3와 같음)을 써야 펴기·증강을 줄이기 전에 할 수 있다.
# 이름|train_exp.py 옵션|평가 방식(eval_offline.py --unfold)
DATASET=dataset3_raw
CKPT_SELECTIONS="b1 last"
E="--norm sample --sched cosine"
EXPERIMENTS=(
  "E_raw|$E|none"                                                      # 대조군: 데이터 형식만 바뀐 E
  "F_aug|$E --aug|none"                                                # + 온라인 증강
  "G_unfold|$E --unfold oracle|oracle select"                          # + 두 줄 펴기
  "H_unfold_aug|$E --unfold oracle --aug|oracle select"                # + 펴기 + 증강
  "I_unfold_aug_os|$E --unfold oracle --aug --oversample_two_line 2|oracle select"  # + 두 줄 2배 샘플링
)
