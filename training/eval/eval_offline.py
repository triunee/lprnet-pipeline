"""모델을 test 셋에서 한 장씩(batch=1, 서빙과 같은 조건) 평가한다.

LPRNet은 global context 정규화가 배치 평균에 의존해서, 배치로 묶어 재면
서빙(app.py)보다 정확도가 높게 나온다. 실험 비교는 이 스크립트 수치로만 한다.

python eval_offline.py --test_dir /workspace/ALPRData/dataset3/test \
    --model base=/workspace/LPRNet-master2/weights/lprnet_kor.pt \
    --model A_orig_s0_b1=runs/A_orig/s0/ckpt/best-b1-epoch=250.ckpt \
    --out_dir runs/eval
"""
import argparse
import csv
import json
import os
import re
import unicodedata
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
NFC = lambda s: unicodedata.normalize("NFC", s)


def plate_type(label: str) -> str:
    if re.search(r"[공국해합육]", label):
        return "military"
    if re.match(r"^[가-힣]{2,}\d", label):
        return "region_truck"
    if re.match(r"^[89]\d\d[가-힣]", label):
        return "truck_8digit"
    return "car"


def label_of(filename: str) -> str:
    return NFC(re.sub(r"_\d+$", "", Path(filename).stem))


def preprocess(path: Path) -> np.ndarray:
    # app.py와 동일
    img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (100, 50), interpolation=cv2.INTER_CUBIC).astype("float32")
    img -= 127.5
    img *= 0.0078125
    return np.transpose(img, (2, 0, 1))[np.newaxis]


def decode(pred: np.ndarray, chars) -> str:
    # app.py와 동일한 greedy CTC
    blank, result, pre = len(chars) - 1, [], -1
    for j in range(pred.shape[1]):
        c = int(np.argmax(pred[:, j]))
        if c != pre and c != blank:
            result.append(chars[c])
        pre = c
    return "".join(result)


def make_predictor(path: str, cfg: dict, device: str):
    if path.endswith(".onnx"):
        import onnxruntime as ort

        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        return lambda x: sess.run(None, {"input": x})[0][0]

    import torch
    from lprnet_ext import LPRNetExp

    if path.endswith(".ckpt"):
        model = LPRNetExp.load_from_checkpoint(path, map_location="cpu")
    else:  # 원본 lprnet_kor.pt (flat state_dict)
        model = LPRNetExp(Namespace(**cfg))
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model = model.to(device).eval()

    def run(x):
        with torch.no_grad():
            return model(torch.from_numpy(x).to(device)).float().cpu().numpy()[0]

    return run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test_dir", required=True)
    p.add_argument("--model", action="append", required=True, help="NAME=PATH (.pt / .ckpt / .onnx)")
    p.add_argument("--train_dir", help="주면 train에 같은 번호가 있는 test 이미지 수를 함께 기록")
    p.add_argument("--config", default=str(REPO / "config/kor_config.yaml"))
    p.add_argument("--out_dir", required=True)
    p.add_argument("--device", default="auto", help="auto / cuda / cpu (.onnx는 항상 CPU)")
    args = p.parse_args()

    if args.device == "auto":
        try:
            import torch

            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            args.device = "cpu"

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.full_load(f)
    chars = cfg["chars"]

    test_dir = Path(args.test_dir)
    files = sorted(f for f in os.listdir(test_dir) if f.lower().endswith((".jpg", ".png", ".jpeg")))
    labels = [label_of(f) for f in files]
    types = [plate_type(l) for l in labels]
    xs = [preprocess(test_dir / f) for f in files]
    train_labels = {label_of(f) for f in os.listdir(args.train_dir)} if args.train_dir else set()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for spec in args.model:
        name, path = spec.split("=", 1)
        predict = make_predictor(path, cfg, args.device)
        preds = [NFC(decode(predict(x), chars)) for x in xs]

        by_type = defaultdict(lambda: [0, 0])
        for t, l, pr in zip(types, labels, preds):
            by_type[t][0] += int(l == pr)
            by_type[t][1] += 1
        correct = sum(l == pr for l, pr in zip(labels, preds))
        summary = {
            "name": name,
            "path": path,
            "test_dir": str(test_dir),
            "batch_size": 1,
            "total": len(files),
            "correct": correct,
            "accuracy": correct / len(files),
            "by_type": {t: {"correct": c, "total": n, "accuracy": c / n} for t, (c, n) in sorted(by_type.items())},
            "test_labels_in_train": sum(l in train_labels for l in labels) if args.train_dir else None,
        }
        (out / f"{name}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        with open(out / f"{name}.csv", "w", newline="", encoding="utf-8") as fp:
            w = csv.writer(fp)
            w.writerow(["file", "label", "type", "pred", "correct"])
            for f, l, t, pr in zip(files, labels, types, preds):
                w.writerow([NFC(f), l, t, pr, int(l == pr)])
        types_txt = " ".join(f"{t}={v['accuracy']*100:.1f}%" for t, v in summary["by_type"].items())
        print(f"{name}: {summary['accuracy']*100:.2f}% ({correct}/{len(files)}) | {types_txt}")


if __name__ == "__main__":
    main()
