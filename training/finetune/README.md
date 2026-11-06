# Fine-tuning

- `train.py` — 파인튜닝 실행 스크립트 (PyTorch Lightning, `config/kor_config.yaml` 사용, `max_epochs=300`, `patience=30`)
- `export_onnx.py` — 학습된 체크포인트(.ckpt/.pt) → ONNX 변환 스크립트
- `train_exp.py`, `lprnet_ext.py` — 정규화·LR 스케줄·seed를 바꿔 가며 돌리는 실험용 학습. 사용법은 [`../experiments/README.md`](../experiments/README.md)
- `plate_preproc.py`, `datamodule_ext.py` — 원본 크기 데이터셋(`dataset3_raw`)용 전처리와 DataModule: 두 줄 번호판 펴기, 학습 중 증강, 두 줄 번호판 오버샘플링

실행 예시:

```bash
python train.py
python export_onnx.py --ckpt <checkpoint>.ckpt --output lprnet_kor.onnx
```

실제 실행 로그와 ONNX 변환·평가 과정은 [`../notebooks/lprnet_kor_finetune_onnx_eval.ipynb`](../notebooks/lprnet_kor_finetune_onnx_eval.ipynb) 참고.
