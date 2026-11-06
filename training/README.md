# Training Pipeline

이 폴더는 배포용 서빙 코드(`app.py`, `evaluate.py` 등)와 분리된 **학습 파이프라인** 모음입니다.
서빙 의존성(`requirements.txt`)과 섞이지 않도록 별도로 관리합니다.

## 구조

- `data_gen/` — 합성 번호판 이미지/라벨 생성 코드. 출처: [yakhyo/korean-license-plate-generator](https://github.com/yakhyo/korean-license-plate-generator), [qjadud1994/Korean-license-plate-Generator](https://github.com/qjadud1994/Korean-license-plate-Generator) (MIT, [THIRD_PARTY_LICENSES.md](../THIRD_PARTY_LICENSES.md) 참고)
  - ⚠️ `data_gen/kor_config.yaml`은 데이터 생성 전용 설정입니다. 루트의 [`config/kor_config.yaml`](../config/kor_config.yaml)(학습/서빙용)과 이름은 같지만 `train_dir`/`test_dir` 등 내용이 다르니 혼동하지 마세요.
- `finetune/` — 파인튜닝 실행 스크립트(`train.py`, `export_onnx.py`), 실험용 학습(`train_exp.py`, `lprnet_ext.py`, `plate_preproc.py`, `datamodule_ext.py`)
- `eval/` — test 셋을 batch=1(서빙과 같은 조건)로 평가하는 스크립트(`eval_offline.py`, `aggregate.py`)
- `experiments/` — GPU 서버(RunPod 등)에서 누수 없는 분할(`dataset3`, 원본 크기 `dataset3_raw`)로 재학습·비교 실험을 돌리는 스크립트와 안내. 차수별 실험 목록은 `experiments/rounds/`
- `notebooks/` — 실제 파인튜닝 ~ ONNX 변환 ~ 평가에 사용한 Colab 노트북 (`lprnet_kor_finetune_onnx_eval.ipynb`)

## 실행 순서 (data_gen)

자세한 내용은 [`../docs/data_gen_notes.md`](../docs/data_gen_notes.md) 참고.

```bash
cd training/data_gen
python gen_car.py
python gen_car_military.py
python gen_truck_8digit.py
python gen_truck.py
python augment.py
python prepare_lprnet.py
```

베이스 모델: [kwon-evan/LPRNet](https://github.com/kwon-evan/LPRNet) (Apache-2.0)
