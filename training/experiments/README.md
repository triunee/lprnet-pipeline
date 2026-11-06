# 재학습·비교 실험 (RunPod 등 GPU 서버)

05-11 배포 모델은 기본 설정으로 한 번 학습했고, test 셋에 train과 같은 번호판이 섞여 있었습니다. 여기 있는 스크립트는 **누수 없는 분할**에서 설정을 하나씩 바꿔 가며 다시 학습하고, 모든 결과를 **batch=1(서빙과 같은 조건)** 로 비교합니다.

실험 목록은 `rounds/<차수>.sh`에 있고, `run_all.sh`가 그 목록을 읽어 학습 → 평가 → 표까지 돌립니다.

## 1차 (`rounds/round1.sh`, 데이터 `dataset3`)

| 이름 | 바꾸는 것 | 확인하려는 것 |
|---|---|---|
| `base` | 학습 없음 (`lprnet_kor.pt`) | 기준값 |
| `A_orig` | 없음. 05-11과 같은 설정을 `dataset3`으로 | 누수 없는 파인튜닝 수치 |
| `B_sample_norm` | global context를 샘플마다 정규화 | 배치 의존 문제를 고치면 나아지는지 |
| `C_cosine` | LR 스케줄: 5 epoch warmup + epoch 단위 cosine | 원본의 step 단위 warm restart를 고치면 나아지는지 |
| `D_cosine_lr3e-4` | C + 시작 lr 3e-4 | 파인튜닝에 맞게 lr을 낮추면 나아지는지 |
| `E_sample_cosine` | B + C | 두 수정을 합친 효과 |

결과(test 1468장, seed 3개 평균): base 37.47% → A 89.2 ± 4.6% → B 97.0 ± 0.3% / C 92.1 ± 0.9% / D 93.7 ± 1.5% / **E 97.7 ± 1.2%**.
B·E의 남은 오답은 거의 전부 **두 줄 지역 화물차의 윗줄 숫자**(`경기92아 → 경기91아`)였고, 형식은 맞는 번호판이라 형식 제약 디코딩으로는 고칠 수 없었습니다.

> 1차의 "마지막 epoch" 줄은 실제 마지막 epoch가 아닙니다. monitor가 붙은 체크포인트 콜백에 `save_last`를 걸어서 `last.ckpt`가 b256 best와 같은 파일이 됐습니다. 지금 `train_exp.py`는 고쳐져 있습니다.

## 2차 (`rounds/round2.sh`, 데이터 `dataset3_raw`)

두 줄 번호판(335×170)을 100×50으로 바로 줄이면 윗줄 글자 높이가 약 17px까지 작아집니다. 2차는 1차 최고 설정(E) 위에서 이 부분을 겨냥합니다.

| 이름 | E에 더하는 것 | 확인하려는 것 |
|---|---|---|
| `E_raw` | 없음 (데이터 형식만 `dataset3_raw`) | 대조군 |
| `F_aug` | 학습 중 온라인 증강 | 외우기(train-acc 100%)를 줄이면 나아지는지 |
| `G_unfold` | 두 줄 번호판 펴기 | 윗줄 숫자를 크게 보여 주면 나아지는지 |
| `H_unfold_aug` | 펴기 + 증강 | 둘을 합친 효과 |
| `I_unfold_aug_os` | H + 두 줄 번호판 2배 자주 뽑기 | 오답이 몰린 유형을 더 보여 주면 나아지는지 |

- **두 줄 펴기**: 윗줄(지역+숫자 2개)과 아랫줄(한글+숫자 4개)을 잘라 같은 높이로 맞춘 뒤 `[윗줄 | 아랫줄]`로 이어 붙이고 100×50으로 줄입니다. 한 줄 번호판과 비슷한 모양이 됩니다. 원본 크기 이미지에서 잘라야 해서 `dataset3_raw`가 필요합니다.
- **펴기 평가는 두 가지**로 냅니다.
  - `배치 앎(상한)`: manifest의 배치(two_line)를 보고 두 줄 번호판만 폅니다. 번호판 배치를 알 때의 상한입니다.
  - `자동 선택`: 모든 이미지를 원본·펴기 두 번 읽고, 펴서 읽은 결과가 지역 화물차 형식이면서 신뢰도(프레임별 최대 확률 평균)가 더 높을 때만 그쪽을 씁니다. 배치 정보가 없는 서빙에서 그대로 쓸 수 있는 방식입니다.
- **증강**(`../finetune/plate_preproc.py`의 `augment`): 약한 회전·크기·이동, 원근, 밝기·대비, 흐림, 노이즈, JPEG 압축. 오프라인 증강(result6: 좌우 회전·전단)과 겹치지 않게 골랐습니다.
- 체크포인트는 `b1`(서빙 조건 val-acc로 선택)과 진짜 `last`만 평가합니다. 1차에서 b1과 b256의 차이는 0.1%p 이하였습니다.
- `dataset3_raw`는 `dataset3`와 분할·파일명이 같고, 이미지만 줄이기 전 원본입니다. dataset3는 줄인 뒤 JPEG로 다시 저장해서 같은 모델도 0.1~0.6%p 다르게 나옵니다(base 37.47% vs 38.08%). 그래서 2차는 대조군 `E_raw`와 비교합니다.

## 파일

| 파일 | 역할 |
|---|---|
| `../data_gen/prepare_lprnet.py` | 번호판 단위 분할. `--raw`면 같은 분할로 원본 크기를 `dataset3_raw`에 복사. 둘 다 `manifest.csv`(파일별 원본·크기·배치)를 남김 |
| `../finetune/lprnet_ext.py` | kwon-evan LPRNet에 실험 옵션(정규화, 스케줄러, batch=1 검증)을 얹은 클래스 |
| `../finetune/plate_preproc.py` | 두 줄 펴기, 증강, 모델 입력 변환, 신뢰도가 붙은 greedy 디코딩. 학습·평가가 같이 씀 |
| `../finetune/datamodule_ext.py` | `dataset3_raw`용 DataModule (펴기, 증강, 두 줄 오버샘플링) |
| `../finetune/train_exp.py` | 실험용 학습. `train.py`(05-11 기록)는 그대로 둡니다 |
| `../eval/eval_offline.py` | test 셋을 한 장씩 평가(`--unfold none/oracle/select`)하고 유형·배치별 정확도를 JSON·CSV로 저장 |
| `../eval/aggregate.py` | 평가 결과를 실험·체크포인트·펴기 방식별 표(markdown)로 묶음 |
| `rounds/round1.sh`, `rounds/round2.sh` | 차수별 데이터셋과 실험 목록 |
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
    ├── dataset3/{train,val,test}                    1차용 (약 72MB)
    └── dataset3_raw/{train,val,test,manifest.csv}   2차용 (약 244MB)
```

두 데이터셋 모두 맥에서 `prepare_lprnet.py`로 만듭니다(`ALPRData-main_4`에서 `python prepare_lprnet.py`, `python prepare_lprnet.py --raw`). 파일명이 NFC로 저장돼 있어서 Linux로 옮겨도 한글 라벨이 바뀌지 않습니다. 맥 Finder에서 zip으로 묶으면 이름이 깨질 수 있어서, 아래처럼 rsync로 보내는 편이 안전합니다. 필요한 차수의 데이터셋만 올려도 됩니다.

```bash
POD="root@<IP>"; PORT=<포트>
rsync -avz -e "ssh -p $PORT" ~/dev/lprnet-pipeline/ $POD:/workspace/lprnet-pipeline/ --exclude .git --exclude terraform --exclude k8s
rsync -avz -e "ssh -p $PORT" ~/Documents/LPRNet-master2/lprnet ~/Documents/LPRNet-master2/weights $POD:/workspace/LPRNet-master2/
rsync -avz -e "ssh -p $PORT" ~/Documents/ALPRData-main_4/dataset3_raw $POD:/workspace/ALPRData/
```

## 실행 순서

1. **Pod 만들기.** `runpod_lifecycle_script`의 SSH 접속 모드로 만듭니다.
   - 모델이 작아서(파라미터 110만 개) H100은 필요 없습니다. RTX 4090, A4000급 한 장이면 충분합니다.
   - 이미지는 `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`(torch 포함)를 씁니다.
   - Volume 없이 돌리면 Pod를 Terminate할 때 결과가 지워집니다. 끄기 전에 꼭 내려받습니다.
2. **파일 올리기.** 위 rsync 세 줄을 실행합니다.
3. **환경 점검.**
   ```bash
   cd /workspace/lprnet-pipeline/training/experiments
   bash setup.sh
   ```
   마지막 줄의 base 정확도가 dataset3 37.47%(550/1468), dataset3_raw 38.08%(559/1468) 근처면 환경이 맞는 것입니다. GPU 연산 차이로 한두 장은 다를 수 있습니다.
4. **짧게 돌려 보기.** 2 epoch만 돌려서 학습 → 평가 → 표가 끝까지 도는지 확인합니다.
   ```bash
   ROUND=round2 SMOKE=1 bash run_all.sh
   ```
5. **본 실험.** SSH가 끊겨도 계속 돌도록 `nohup`으로 실행합니다.
   ```bash
   ROUND=round2 nohup bash run_all.sh > /workspace/run_round2.log 2>&1 &
   tail -f /workspace/run_round2.log
   ```
   - 기본값은 동시에 3개(`PARALLEL=3`)입니다. run 하나가 GPU 메모리 6~7GB와 data worker 4개를 씁니다. RTX 4090(24GB)에서 5개를 돌리면 메모리가 부족해 실패합니다(OOM). GPU 메모리가 더 큰 서버라면 vCPU 수도 보고 늘리세요. Pod 안의 `nproc`는 서버 전체 코어 수를 보여 줄 수 있어서, RunPod 화면에 나온 vCPU 수를 기준으로 잡습니다.
   - 실험 5개 × seed 3개 = 학습 15번입니다. 1차는 RTX 4090에서 약 3시간 10분(early stopping으로 run당 15~55분) 걸렸습니다. 중간에 끊기면 같은 명령을 다시 실행하면 됩니다.
6. **결과 내려받기.** 결과는 `/workspace/runs_<차수>`에 쌓입니다.
   ```bash
   rsync -avz -e "ssh -p $PORT" --exclude '*.ckpt' $POD:/workspace/runs_round2/ ./runs_round2/
   ```
   체크포인트까지 필요하면 `--exclude`를 빼세요. run 하나에 13MB짜리가 3개 생깁니다.
7. **Pod 멈추기.** 내려받은 뒤 바로 `stop` 또는 `terminate` 합니다. 켜 두는 동안 계속 과금됩니다.

## 결과물

- `runs_<차수>/results.md`: 실험 × 체크포인트 × 펴기 방식별 전체·유형별(2차는 두 줄 번호판 포함) 정확도 표 (batch=1, 평균 ± 표준편차)
- `runs_<차수>/eval/*.csv`: 이미지별 예측(2차는 배치와 원본·펴기 중 어느 쪽을 썼는지 포함). 오답 분석용
- `runs_<차수>/eval/*.json`의 `routing`: 자동 선택이 두 줄 번호판을 펴서 읽은 비율, 한 줄 번호판을 잘못 편 수
- `runs_<차수>/<실험>/s<seed>/logs/metrics.csv`: epoch별 train/val loss, `val-acc`, `val-acc-b1`, lr
- `runs_<차수>/<실험>/s<seed>/run_config.json`: 그 run의 전체 설정

비교할 때는 `val-acc`가 아니라 `results.md`의 batch=1 수치만 씁니다. 학습 중 `val-acc`는 배치 256으로 재서, 원본 정규화(1차 `A`, `C`, `D`)에서는 서빙보다 3~4%p 높게 나옵니다.
