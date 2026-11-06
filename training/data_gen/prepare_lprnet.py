import os
import re
import sys
import random
import unicodedata
import yaml
import cv2
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

SRC_DIRS    = ["result2", "result3", "result32", "result33", "result6"]
AUG_DIR     = "result6"  # 다른 폴더 이미지를 증강한 폴더 — 원본과 같은 split으로 보낸다
TRAIN_DIR   = "./dataset3/train"
VAL_DIR     = "./dataset3/val"
TEST_DIR    = "./dataset3/test"
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# 나머지 0.15가 test
RANDOM_SEED = 42
IMG_SIZE    = (100, 50)  # (width, height) for cv2.resize

print("[1/5] kor_config.yaml 로드 중...")
with open("./kor_config.yaml", encoding="utf-8") as f:
    cfg = yaml.full_load(f)
CHARS = cfg["chars"]
print(f"      chars 로드 완료: {len(CHARS)}개 토큰\n")


def is_valid_label(label: str) -> bool:
    i = 0
    while i < len(label):
        j = len(label)
        matched = False
        while i < j:
            if label[i:j] in CHARS:
                i = j
                matched = True
                break
            j -= 1
        if not matched:
            return False
    return True


def extract_label(filepath) -> str:
    stem = Path(filepath).stem
    stem = re.sub(r'_(lp|rp|l|r)$', '', stem)  # 증강 접미사 제거 (_l, _r, _lp, _rp)
    stem = re.sub(r'_\d+$', '', stem)           # 중복 접미사 제거 (_1, _2 등)
    return unicodedata.normalize("NFC", stem)   # macOS 파일명(NFD) → NFC, 다른 OS로 옮겨도 라벨이 같게


def process_image(src_path: str, dst_path: str) -> bool:
    img = cv2.imread(src_path)
    if img is None:
        return False
    img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(dst_path, img)
    return True


def save_split(file_list, dst_dir, desc):
    counter = {}
    for src_path, label in tqdm(file_list, desc=desc):
        counter[label] = counter.get(label, 0) + 1
        dst_name = f"{label}_{counter[label]}.jpg"
        dst_path = os.path.join(dst_dir, dst_name)
        if not process_image(str(src_path), dst_path):
            print(f"[ERROR] 읽기 실패: {src_path}")


for d in (TRAIN_DIR, VAL_DIR, TEST_DIR):
    if os.path.isdir(d) and os.listdir(d):
        sys.exit(f"[ERROR] {d} 가 비어 있지 않습니다. 이전 결과와 섞이지 않도록 지우거나 경로를 바꾼 뒤 다시 실행하세요.")

random.seed(RANDOM_SEED)

print("[2/5] 소스 폴더에서 이미지 수집 중...")
all_files = []
for src in SRC_DIRS:
    before = len(all_files)
    for ext in ("*.jpg", "*.png", "*.jpeg"):
        all_files.extend(Path(src, "images").glob(ext))
    print(f"      {src}/: {len(all_files) - before}장")
# glob 순서는 파일시스템마다 달라서, 정렬해야 다른 기기에서도 같은 분할이 나온다
all_files.sort(key=lambda p: unicodedata.normalize("NFC", str(p)))
print(f"      합계: {len(all_files)}장\n")

print("[3/5] 라벨 유효성 검사 중...")
# 같은 번호판(라벨)의 이미지를 한 그룹으로 묶는다 (원본 + 증강본 + 중복 생성분)
groups = defaultdict(list)
skip_count = 0
for f in all_files:
    label = extract_label(f)
    if is_valid_label(label):
        groups[label].append((f, label))
    else:
        print(f"[SKIP] {f.name}  ←  미등록 문자 포함: '{label}'")
        skip_count += 1

total_valid = sum(len(v) for v in groups.values())
print(f"      유효: {total_valid}장 / 스킵: {skip_count}장 / 번호판 {len(groups)}개\n")

print("[4/5] 번호판 단위 그룹 분할(원본 폴더 기준 stratified) train/val/test 중...")
# 그룹의 원본 폴더 = 증강 폴더가 아닌 첫 소스. 증강본만 있으면 증강 폴더로 둔다
per_src = {src: [] for src in SRC_DIRS}
for label, files in groups.items():
    origins = [f.parts[0] for f, _ in files if f.parts[0] != AUG_DIR]
    per_src[origins[0] if origins else AUG_DIR].append(label)

train_files, val_files, test_files = [], [], []
for src, labels in per_src.items():
    random.shuffle(labels)
    t_end = int(len(labels) * TRAIN_RATIO)
    v_end = int(len(labels) * (TRAIN_RATIO + VAL_RATIO))
    parts = [labels[:t_end], labels[t_end:v_end], labels[v_end:]]
    for dst, part in zip((train_files, val_files, test_files), parts):
        for label in part:
            dst += groups[label]
    counts = " / ".join(f"{name} {len(p)}개({sum(len(groups[l]) for l in p)}장)"
                        for name, p in zip(("train", "val", "test"), parts))
    print(f"      {src}/: {counts}")

# 누수 검사: 한 번호판이 두 split에 들어가면 안 된다
split_labels = [{l for _, l in s} for s in (train_files, val_files, test_files)]
assert not (split_labels[0] & split_labels[1] or split_labels[0] & split_labels[2]
            or split_labels[1] & split_labels[2]), "split 간 번호판 중복"

# 폴더 순서 편향 제거를 위해 각 split 내부도 셔플
random.shuffle(train_files)
random.shuffle(val_files)
random.shuffle(test_files)
print(f"      합계  : train {len(train_files)} / val {len(val_files)} / test {len(test_files)}\n")

print("[5/5] 이미지 저장 중...")
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR,   exist_ok=True)
os.makedirs(TEST_DIR,  exist_ok=True)

save_split(train_files, TRAIN_DIR, "train")
save_split(val_files,   VAL_DIR,   "val  ")
save_split(test_files,  TEST_DIR,  "test ")

print(f"\n완료! (split 간 번호판 중복 0건)")
print(f"  train : {len(train_files)}장  →  {TRAIN_DIR}")
print(f"  val   : {len(val_files)}장  →  {VAL_DIR}")
print(f"  test  : {len(test_files)}장  →  {TEST_DIR}")
