"""원본 크기 데이터셋(dataset3_raw)용 DataModule: 두 줄 번호판 펴기, 온라인 증강, 두 줄 번호판 오버샘플링.

kwon-evan DataModule은 이미 100x50으로 줄인 이미지(dataset3)를 읽어서, 줄이기 전에 해야 하는
펴기·증강을 넣을 수 없다. 그래서 dataset3_raw/{train,val}과 manifest.csv(파일별 배치)를 읽는다.
옵션을 모두 끄면 입력은 app.py 전처리(원본 → 100x50 INTER_CUBIC)와 같다.

- unfold="oracle": manifest에서 two_line인 이미지만 펴서 넣는다 (학습·검증 모두)
- aug=True: train에만 plate_preproc.augment 적용
- oversample_two_line=k: train에서 두 줄 번호판을 k배 자주 뽑는다 (epoch 길이는 그대로)
"""
import csv
import os
import random
import unicodedata
from pathlib import Path

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from lprnet.datamodule import collate_fn
from lprnet.utils import encode
from plate_preproc import augment, read_image, to_input

NFC = lambda s: unicodedata.normalize("NFC", s)


def load_layouts(manifest_path) -> dict:
    with open(manifest_path, encoding="utf-8") as f:
        return {(r["split"], NFC(r["file"])): r["layout"] for r in csv.DictReader(f)}


def label_of(filename: str) -> str:
    # kwon-evan datamodule과 같은 규칙
    return NFC(os.path.splitext(filename)[0].split("-")[0].split("_")[0].upper())


class PlateDataset(Dataset):
    def __init__(self, img_dir, split, layouts, chars, unfold="none", aug=False):
        self.img_dir = Path(img_dir)
        self.files = sorted(f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png", ".jpeg")))
        self.layouts = [layouts[(split, NFC(f))] for f in self.files]
        self.labels = [encode(label_of(f), chars) for f in self.files]
        self.unfold, self.aug = unfold, aug

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = read_image(self.img_dir / self.files[i])
        if self.aug:
            img = augment(img, random)  # worker별 seed는 L.seed_everything(workers=True)가 맞춘다
        x = to_input(img, unfold=self.unfold == "oracle" and self.layouts[i] == "two_line")
        return x, self.labels[i], len(self.labels[i])


class DataModuleExp(L.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        root = Path(args.train_dir).parent
        self.layouts = load_layouts(root / "manifest.csv")

    def setup(self, stage: str):
        a = self.args
        self.train = PlateDataset(a.train_dir, "train", self.layouts, a.chars, a.unfold, a.aug)
        self.val = PlateDataset(a.valid_dir, "val", self.layouts, a.chars, a.unfold, aug=False)
        two = sum(l == "two_line" for l in self.train.layouts)
        print(f"train {len(self.train)} (두 줄 {two}) / val {len(self.val)} | "
              f"unfold={a.unfold} aug={a.aug} oversample_two_line={a.oversample_two_line}")

    def train_dataloader(self):
        k = self.args.oversample_two_line
        common = dict(batch_size=self.args.batch_size, num_workers=4, collate_fn=collate_fn,
                      persistent_workers=True)
        if k == 1:
            return DataLoader(self.train, shuffle=True, **common)
        weights = torch.tensor([k if l == "two_line" else 1.0 for l in self.train.layouts], dtype=torch.double)
        sampler = WeightedRandomSampler(weights, num_samples=len(self.train), replacement=True)
        return DataLoader(self.train, sampler=sampler, **common)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.args.batch_size, shuffle=False, num_workers=4,
                          collate_fn=collate_fn, persistent_workers=True)
