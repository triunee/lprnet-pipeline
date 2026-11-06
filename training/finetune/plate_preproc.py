"""번호판 이미지 전처리: 두 줄 번호판 펴기, 학습용 증강, 모델 입력 변환.

학습(datamodule_ext.py)과 평가(eval_offline.py)가 같은 함수를 쓰도록 한곳에 둔다.
torch 없이 numpy·cv2만 쓴다 (ONNX 평가·서빙에서도 그대로 쓸 수 있게).

두 줄 번호판(지역 화물차, 335x170)을 100x50으로 바로 줄이면 윗줄 "지역+숫자 2개"의
글자 높이가 약 17px까지 작아진다. 남은 오답이 거의 다 이 윗줄 숫자라서,
윗줄과 아랫줄을 잘라 가로로 이어 붙여 한 줄 번호판처럼 만든 뒤 줄인다.
"""
import random

import cv2
import numpy as np

IMG_SIZE = (100, 50)  # (너비, 높이) — app.py와 같음

# gen_truck.py Type_C 기준 비율. 윗줄 y 8~68, 아랫줄 y 70~162 (높이 170).
# 회전·전단 증강본(result6)도 읽히도록 여유를 둔다.
TOP_ROWS = (0.0, 0.44)
BOTTOM_ROWS = (0.38, 1.0)
TOP_COLS = (0.12, 0.88)  # 윗줄은 가운데에만 글자가 있다


def unfold_two_line(img: np.ndarray) -> np.ndarray:
    """두 줄 번호판의 윗줄과 아랫줄을 같은 높이로 맞춰 [윗줄 | 아랫줄]로 잇는다."""
    h, w = img.shape[:2]
    top = img[int(h * TOP_ROWS[0]) : int(h * TOP_ROWS[1]), int(w * TOP_COLS[0]) : int(w * TOP_COLS[1])]
    bottom = img[int(h * BOTTOM_ROWS[0]) : int(h * BOTTOM_ROWS[1])]
    bh = bottom.shape[0]
    top = cv2.resize(top, (max(1, round(top.shape[1] * bh / top.shape[0])), bh), interpolation=cv2.INTER_CUBIC)
    return np.hstack([top, bottom])


def to_input(img: np.ndarray, unfold: bool = False) -> np.ndarray:
    """BGR uint8 이미지 → 모델 입력 (3, 50, 100) float32. unfold=False면 app.py 전처리와 같다."""
    if unfold:
        img = unfold_two_line(img)
    img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_CUBIC).astype("float32")
    img -= 127.5
    img *= 0.0078125
    return np.transpose(img, (2, 0, 1))


def read_image(path) -> np.ndarray:
    # 한글 경로도 읽히게 imdecode 사용
    img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"이미지를 읽지 못했습니다: {path}")
    return img


def augment(img: np.ndarray, rng: random.Random = random) -> np.ndarray:
    """학습용 온라인 증강. 원본 크기 이미지에 적용한 뒤 펴기·축소를 한다.

    오프라인 증강(result6: 좌우 회전·전단)과 겹치지 않게, 실제 카메라에서 흔한
    약한 기하 변형·밝기·흐림·노이즈·JPEG 압축을 무작위로 섞는다.
    """
    h, w = img.shape[:2]
    if rng.random() < 0.5:  # 회전·크기·이동
        angle = rng.uniform(-4, 4)
        scale = rng.uniform(0.92, 1.04)
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        m[0, 2] += rng.uniform(-0.03, 0.03) * w
        m[1, 2] += rng.uniform(-0.04, 0.04) * h
        img = cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if rng.random() < 0.3:  # 원근
        d = 0.04
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        jitter = np.float32([[rng.uniform(-d, d) * w, rng.uniform(-d, d) * h] for _ in range(4)])
        img = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, src + jitter), (w, h),
                                  borderMode=cv2.BORDER_REPLICATE)
    if rng.random() < 0.5:  # 밝기·대비
        img = cv2.convertScaleAbs(img, alpha=rng.uniform(0.7, 1.3), beta=rng.uniform(-25, 25))
    if rng.random() < 0.2:  # 흐림 — 100px 너비로 줄인 뒤에도 0.5~1.2px 정도 번지게 원본 크기에 맞춘다
        img = cv2.GaussianBlur(img, (0, 0), sigmaX=rng.uniform(0.5, 1.2) * w / IMG_SIZE[0])
    if rng.random() < 0.2:  # 노이즈
        noise = np.random.default_rng(rng.getrandbits(32)).normal(0, rng.uniform(3, 10), img.shape)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if rng.random() < 0.3:  # JPEG 압축
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(30, 80)])
        if ok:
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img


def greedy_decode(logits: np.ndarray, chars) -> tuple:
    """(클래스 수, T) 출력 → (문자열, 신뢰도). 디코딩은 app.py와 같은 greedy CTC.

    신뢰도는 프레임마다 가장 높은 softmax 확률의 평균. 펴기/원본 두 결과 중 하나를 고를 때 쓴다.
    """
    z = logits - logits.max(axis=0, keepdims=True)
    prob = np.exp(z) / np.exp(z).sum(axis=0, keepdims=True)
    best = prob.argmax(axis=0)
    blank, out, pre = len(chars) - 1, [], -1
    for c in best:
        c = int(c)
        if c != pre and c != blank:
            out.append(chars[c])
        pre = c
    return "".join(out), float(prob.max(axis=0).mean())
