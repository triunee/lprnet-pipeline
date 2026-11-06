# 재학습·비교 실험 (RunPod 등 GPU 서버)

05-11 배포 모델은 기본 설정으로 한 번 학습했고, test 셋에 train과 같은 번호판이 섞여 있었습니다. 여기 있는 스크립트는 **누수 없는 `dataset3`** 에서 설정을 하나씩 바꿔 가며 다시 학습하고, 모든 결과를 **batch=1(서빙과 같은 조건)** 로 비교합니다.

## 실험

| 이름 | 바꾸는 것 | 확인하려는 것 |
|---|---|---|
| `base` | 학습 없음 (`lprnet_kor.pt`) | 기준값 |
| `A_orig` | 없음. 05-11과 같은 설정을 `dataset3`으로 | 누수 없는 파인튜닝 수치 |
| `B_sample_norm` | global context를 샘플마다 정규화 | 배치 의존 문제를 고치면 나아지는지 |
| `C_cosine` | LR 스케줄: 5 epoch warmup + epoch 단위 cosine | 원본의 step 단위 warm restart를 고치면 나아지는지 |
| `D_cosine_lr3e-4` | C + 시작 lr 3e-4 | 파인튜닝에 맞게 lr을 낮추면 나아지는지 |
| `E_sample_cosine` | B + C | 두 수정을 합친 효과 |

- 실험마다 seed 0, 1, 2로 세 번 돌립니다. 결과는 평균 ± 표준편차로 냅니다.
- 한 번 학습할 때 체크포인트를 세 가지 기준으로 남깁니다.
  - `b256`: 원본과 같은 기준(배치 256으로 잰 val-acc)
  - `b1`: 서빙 조건(한 장씩 잰 val-acc)
  - `last`: 마지막 epoch
  - 그래서 `A_orig`의 `b256`과 `b1`을 비교하면 "체크포인트를 고르는 기준만 바꾼 효과"를 학습 추가 없이 볼 수 있습니다.
- 학습 로그(`metrics.csv`)에는 epoch마다 `val-acc`와 `val-acc-b1`이 함께 남습니다.

## 파일

| 파일 | 역할 |
|---|---|
| `../finetune/lprnet_ext.py` | kwon-evan LPRNet에 실험 옵션(정규화, 스케줄러, batch=1 검증)을 얹은 클래스 |
| `../finetune/train_exp.py` | 실험용 학습. `train.py`(05-11 기록)는 그대로 둡니다 |
| `../eval/eval_offline.py` | test 셋을 한 장씩 평가하고 번호판 유형별 정확도를 JSON·CSV로 저장 |
| `../eval/aggregate.py` | 평가 결과를 실험·선택 기준별 표(markdown)로 묶음 |
| `setup.sh` | 패키지 설치, GPU·데이터 점검, base 재측정 |
| `run_all.sh` | 학습(병렬) → 평가 → 표까지 한 번에. 끝난 run은 건너뛰어서 다시 실행하면 이어서 돕니다 |

## 서버에 올릴 것

서버의 `/workspace` 아래에 이렇게 놓습니다(경로를 바꾸려면 `WORK`, `REPO`, `LPRNET_SRC`, `DATA_ROOT` 환경변수).

```
/workspace/
├── lprnet-pipeline/          이 레포
├── LPRNet-master2/
│   ├── lprnet/               kwon-evan 모델 코드
│   └── weights/lprnet_kor.pt base 가중치
└── ALPRData/
    └── dataset3/{train,val,test}   prepare_lprnet.py로 만든 누수 없는 분할 (약 72MB)
```

`dataset3`은 파일명이 NFC로 저장돼 있어서 Linux로 옮겨도 한글 라벨이 바뀌지 않습니다. 맥 Finder에서 zip으로 묶으면 이름이 깨질 수 있어서, 아래처럼 rsync로 보내는 편이 안전합니다.

```bash
POD="root@<IP>"; PORT=<포트>
rsync -avz -e "ssh -p $PORT" ~/dev/lprnet-pipeline/ $POD:/workspace/lprnet-pipeline/ --exclude .git --exclude terraform --exclude k8s
rsync -avz -e "ssh -p $PORT" ~/Documents/LPRNet-master2/lprnet ~/Documents/LPRNet-master2/weights $POD:/workspace/LPRNet-master2/
rsync -avz -e "ssh -p $PORT" ~/Documents/ALPRData-main_4/dataset3 $POD:/workspace/ALPRData/
```

## 실행 순서

1. **Pod 만들기.** `runpod_lifecycle_script`의 SSH 접속 모드로 만듭니다.
   - 모델이 작아서(파라미터 110만 개) H100은 필요 없습니다. RTX 4090, A4000급 한 장이면 충분합니다.
   - 이미지는 `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`(torch 포함)를 씁니다.
   - 결과가 Pod를 멈춘 뒤에도 남도록 Volume을 `/workspace`에 마운트합니다.
2. **파일 올리기.** 위 rsync 세 줄을 실행합니다.
3. **환경 점검.**
   ```bash
   cd /workspace/lprnet-pipeline/training/experiments
   bash setup.sh
   ```
   마지막 줄의 base 정확도가 37.47%(550/1468) 근처면 환경이 맞는 것입니다. GPU 연산 차이로 한두 장은 다를 수 있습니다.
4. **짧게 돌려 보기.** 2 epoch만 돌려서 학습 → 평가 → 표가 끝까지 도는지 확인합니다.
   ```bash
   SMOKE=1 bash run_all.sh
   ```
5. **본 실험.** SSH가 끊겨도 계속 돌도록 `nohup`으로 실행합니다.
   ```bash
   nohup bash run_all.sh > /workspace/run_all.log 2>&1 &
   tail -f /workspace/run_all.log
   ```
   - 기본값은 동시에 5개(`PARALLEL=5`)입니다. run 하나가 data worker 4개를 쓰므로, `setup.sh`가 보여 준 vCPU 수에 맞춰 `PARALLEL=3 bash run_all.sh`처럼 줄이세요.
   - 실험 5개 × seed 3개 = 학습 15번입니다. 중간에 끊기면 같은 명령을 다시 실행하면 됩니다.
6. **결과 내려받기.**
   ```bash
   rsync -avz -e "ssh -p $PORT" --exclude '*.ckpt' $POD:/workspace/runs/ ./runs/
   ```
   체크포인트까지 필요하면 `--exclude`를 빼세요. run 하나에 13MB짜리가 3~4개 생깁니다.
7. **Pod 멈추기.** 내려받은 뒤 바로 `stop` 또는 `terminate` 합니다. 켜 두는 동안 계속 과금됩니다.

## 결과물

- `runs/results.md`: 실험 × 선택 기준별 전체·유형별 정확도 표 (batch=1, 평균 ± 표준편차)
- `runs/eval/*.csv`: 이미지별 예측. 오답 분석용
- `runs/<실험>/s<seed>/logs/metrics.csv`: epoch별 train/val loss, `val-acc`, `val-acc-b1`, lr
- `runs/<실험>/s<seed>/run_config.json`: 그 run의 전체 설정

비교할 때는 `val-acc`가 아니라 `results.md`의 batch=1 수치만 씁니다. 학습 중 `val-acc`는 배치 256으로 재서, 원본 정규화(`A`, `C`, `D`)에서는 서빙보다 3~4%p 높게 나옵니다.
