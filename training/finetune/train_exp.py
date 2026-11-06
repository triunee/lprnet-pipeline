"""실험용 파인튜닝. train.py(05-11 배포 모델 학습 기록)는 그대로 두고, 옵션만 바꿔 돌린다.

python train_exp.py --name A_orig --seed 0 --data_root /workspace/ALPRData
"""
import argparse
import json
import os
import warnings
from argparse import Namespace
from pathlib import Path

import yaml
import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger

from lprnet import DataModule
from lprnet_ext import LPRNetExp

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[2]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="실험 이름 (예: A_orig)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--config", default=str(REPO / "config/kor_config.yaml"))
    p.add_argument("--data_root", required=True, help="dataset3/{train,val,test}가 있는 폴더")
    p.add_argument("--dataset", default="dataset3")
    p.add_argument("--pretrained", required=True, help="base 가중치 lprnet_kor.pt")
    p.add_argument("--out_dir", default="runs")
    p.add_argument("--norm", choices=["batch", "sample"], default="batch")
    p.add_argument("--sched", choices=["orig", "cosine"], default="orig")
    p.add_argument("--lr", type=float, default=None, help="비우면 config 값(0.001)")
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--max_epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=30, help="val-acc 기준, 0이면 끔")
    p.add_argument("--batch_size", type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    opt = parse_args()
    with open(opt.config, encoding="utf-8") as f:
        args = Namespace(**yaml.load(f, Loader=yaml.FullLoader))

    data = Path(opt.data_root) / opt.dataset
    args.train_dir = str(data / "train")
    args.valid_dir = str(data / "val")
    args.test_dir = str(data / "test")
    args.pretrained = opt.pretrained
    args.norm, args.sched = opt.norm, opt.sched
    args.warmup_epochs, args.max_epochs = opt.warmup_epochs, opt.max_epochs
    args.seed = opt.seed
    if opt.lr is not None:
        args.lr = opt.lr
    if opt.batch_size is not None:
        args.batch_size = opt.batch_size

    run_dir = Path(opt.out_dir) / opt.name / f"s{opt.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_dump = {k: v for k, v in vars(args).items() if k != "chars"}
    (run_dir / "run_config.json").write_text(json.dumps(cfg_dump, ensure_ascii=False, indent=2, default=str))
    print(json.dumps(cfg_dump, ensure_ascii=False, default=str))

    L.seed_everything(opt.seed, workers=True)

    model = LPRNetExp(args)
    model.load_state_dict(torch.load(args.pretrained, map_location="cpu", weights_only=True))

    callbacks = [
        # E0: 원본과 같은 기준(배치 단위 val-acc)으로 고른 체크포인트
        ModelCheckpoint(dirpath=run_dir / "ckpt", monitor="val-acc", mode="max", save_top_k=1,
                        filename="best-b256-{epoch:03d}", save_last=True),
        # E1: 서빙 조건(batch=1) val-acc로 고른 체크포인트
        ModelCheckpoint(dirpath=run_dir / "ckpt", monitor="val-acc-b1", mode="max", save_top_k=1,
                        filename="best-b1-{epoch:03d}"),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    if opt.patience > 0:
        callbacks.append(EarlyStopping(monitor="val-acc", mode="max", patience=opt.patience))

    trainer = L.Trainer(
        max_epochs=opt.max_epochs,
        callbacks=callbacks,
        precision="16-mixed",
        accelerator="auto",
        devices=1,
        logger=CSVLogger(run_dir, name="logs", version=""),
        enable_progress_bar=False,
        log_every_n_steps=10,
    )
    trainer.fit(model=model, datamodule=DataModule(args))
    (run_dir / "DONE").write_text(f"epochs={trainer.current_epoch}\n")
