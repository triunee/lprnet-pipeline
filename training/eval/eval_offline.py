"""모델을 test 셋에서 한 장씩(batch=1, 서빙과 같은 조건) 평가한다.

LPRNet은 global context 정규화가 배치 평균에 의존해서, 배치로 묶어 재면
서빙(app.py)보다 정확도가 높게 나온다. 실험 비교는 이 스크립트 수치로만 한다.

python eval_offline.py --test_dir /workspace/ALPRData/dataset3/test \
    --model base=/workspace/LPRNet-master2/weights/lprnet_kor.pt \
    --model A_orig_s0_b1=runs/A_orig/s0/ckpt/best-b1-epoch=250.ckpt \
    --out_dir runs/eval

두 줄 번호판 펴기 실험은 원본 크기 test 셋(dataset3_raw/test)과 --unfold로 평가한다.
  --unfold none    : 모두 원본 그대로 (기본, app.py와 같은 전처리)
  --unfold oracle  : manifest.csv에서 two_line인 이미지만 편다 (배치를 안다고 가정한 상한)
  --unfold select  : 모든 이미지를 원본·펴기 두 번 읽고, 펴서 읽은 결과가 지역 화물차 형식이면서
                     신뢰도가 더 높을 때만 그쪽을 쓴다 (배치 정보 없이 서빙에서 쓸 수 있는 방식)
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

import sys

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "training/finetune"))
from plate_preproc import greedy_decode, read_image, to_input  # noqa: E402

NFC = lambda s: unicodedata.normalize("NFC", s)
REGION_TRUCK = re.compile(r"^[가-힣]{2}\d{2}[가-힣]\d{4}$")


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


def load_layouts(test_dir: Path) -> dict:
    manifest = test_dir.parent / "manifest.csv"
    if not manifest.exists():
        return {}
    with open(manifest, encoding="utf-8") as f:
        return {NFC(r["file"]): r["layout"] for r in csv.DictReader(f) if r["split"] == test_dir.name}


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
    p.add_argument("--unfold", choices=["none", "oracle", "select"], default="none",
                   help="두 줄 번호판 펴기 방식 (위 설명). oracle은 test 셋 옆 manifest.csv가 필요")
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
    layouts_map = load_layouts(test_dir)
    layouts = [layouts_map.get(NFC(f), "-") for f in files]
    if args.unfold == "oracle" and "-" in layouts:
        p.error(f"--unfold oracle 은 {test_dir.parent / 'manifest.csv'} 에 모든 test 파일의 배치가 있어야 합니다")
    imgs = [read_image(test_dir / f) for f in files]
    xs_plain = [to_input(im)[np.newaxis] for im in imgs]
    xs_unfold = [to_input(im, unfold=True)[np.newaxis] for im in imgs] if args.unfold != "none" else None
    train_labels = {label_of(f) for f in os.listdir(args.train_dir)} if args.train_dir else set()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for spec in args.model:
        name, path = spec.split("=", 1)
        predict = make_predictor(path, cfg, args.device)
        preds, views = [], []
        for i in range(len(files)):
            if args.unfold == "none" or (args.unfold == "oracle" and layouts[i] != "two_line"):
                pr, view = greedy_decode(predict(xs_plain[i]), chars)[0], "plain"
            elif args.unfold == "oracle":
                pr, view = greedy_decode(predict(xs_unfold[i]), chars)[0], "unfold"
            else:  # select
                (p0, c0), (p1, c1) = (greedy_decode(predict(x[i]), chars) for x in (xs_plain, xs_unfold))
                pr, view = (p1, "unfold") if REGION_TRUCK.match(NFC(p1)) and c1 > c0 else (p0, "plain")
            preds.append(NFC(pr))
            views.append(view)

        by_type = defaultdict(lambda: [0, 0])
        for t, l, pr in zip(types, labels, preds):
            by_type[t][0] += int(l == pr)
            by_type[t][1] += 1
        by_layout = defaultdict(lambda: [0, 0])
        for lay, l, pr in zip(layouts, labels, preds):
            by_layout[lay][0] += int(l == pr)
            by_layout[lay][1] += 1
        # 펴서 읽은 이미지가 실제로 두 줄 번호판이었는지 (select가 배치를 맞게 골랐는지)
        routing = defaultdict(int)
        for lay, v in zip(layouts, views):
            routing[f"{lay}->{v}"] += 1
        correct = sum(l == pr for l, pr in zip(labels, preds))
        summary = {
            "name": name,
            "path": path,
            "test_dir": str(test_dir),
            "batch_size": 1,
            "unfold": args.unfold,
            "total": len(files),
            "correct": correct,
            "accuracy": correct / len(files),
            "by_type": {t: {"correct": c, "total": n, "accuracy": c / n} for t, (c, n) in sorted(by_type.items())},
            "by_layout": {k: {"correct": c, "total": n, "accuracy": c / n} for k, (c, n) in sorted(by_layout.items())},
            "routing": dict(sorted(routing.items())),
            "test_labels_in_train": sum(l in train_labels for l in labels) if args.train_dir else None,
        }
        (out / f"{name}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        with open(out / f"{name}.csv", "w", newline="", encoding="utf-8") as fp:
            w = csv.writer(fp)
            w.writerow(["file", "label", "type", "layout", "view", "pred", "correct"])
            for f, l, t, lay, v, pr in zip(files, labels, types, layouts, views, preds):
                w.writerow([NFC(f), l, t, lay, v, pr, int(l == pr)])
        types_txt = " ".join(f"{t}={v['accuracy']*100:.1f}%" for t, v in summary["by_type"].items())
        layout_txt = " ".join(f"{k}={v['accuracy']*100:.1f}%" for k, v in summary["by_layout"].items() if k != "-")
        print(f"{name}: {summary['accuracy']*100:.2f}% ({correct}/{len(files)}) | {types_txt}"
              + (f" | {layout_txt} | {summary['routing']}" if layout_txt else ""))


if __name__ == "__main__":
    main()
