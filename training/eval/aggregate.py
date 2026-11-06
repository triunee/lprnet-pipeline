"""eval_offline.py 결과(JSON)를 실험·체크포인트 선택 기준별로 묶어 평균±표준편차 표를 만든다.

파일 이름 규칙: <실험>__s<seed>__<선택기준>[_select].json (예: A_orig__s0__b1.json), base는 base.json
두 줄 펴기 방식은 JSON의 "unfold" 값(없으면 none)으로 나눈다.
python aggregate.py runs/eval > runs/results.md
"""
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

TYPES = ["car", "truck_8digit", "military", "region_truck"]
SELECTION = {"b256": "val-acc(배치 256)로 선택", "b1": "val-acc-b1(서빙 조건)로 선택", "last": "마지막 epoch"}
UNFOLD = {"none": "안 함", "oracle": "배치 앎(상한)", "select": "자동 선택"}


def fmt(values):
    values = [v * 100 for v in values]
    if len(values) == 1:
        return f"{values[0]:.2f}"
    return f"{statistics.mean(values):.2f} ± {statistics.stdev(values):.2f}"


def main(eval_dir):
    rows = defaultdict(list)
    for p in sorted(Path(eval_dir).glob("*.json")):
        s = json.loads(p.read_text())
        m = re.match(r"^(.+)__s(\d+)__([a-z0-9]+)(?:_select)?$", p.stem)
        key = (m.group(1), m.group(3), s.get("unfold", "none")) if m else (p.stem, "-", s.get("unfold", "none"))
        rows[key].append(s)

    has_layout = any("two_line" in s.get("by_layout", {}) for ss in rows.values() for s in ss)
    extra = " 두 줄 번호판 |" if has_layout else ""
    print("| 실험 | 체크포인트 선택 | 두 줄 펴기 | seed 수 | 전체 (%) | " + " | ".join(TYPES) + " |" + extra)
    print("|---|---|---|---|---|" + "---|" * len(TYPES) + ("---|" if has_layout else ""))
    order = {"none": 0, "oracle": 1, "select": 2}
    for (exp, sel, unfold), ss in sorted(rows.items(), key=lambda kv: (kv[0][0] != "base", kv[0][0], kv[0][1], order.get(kv[0][2], 9))):
        cells = [fmt([s["accuracy"] for s in ss])]
        for t in TYPES + (["two_line"] if has_layout else []):
            src = "by_layout" if t == "two_line" else "by_type"
            vals = [s[src][t]["accuracy"] for s in ss if t in s.get(src, {})]
            cells.append(fmt(vals) if vals else "-")
        print(f"| {exp} | {SELECTION.get(sel, sel)} | {UNFOLD.get(unfold, unfold)} | {len(ss)} | " + " | ".join(cells) + " |")
    total = next(iter(rows.values()))[0]["total"]
    print(f"\ntest {total}장, batch=1(서빙과 같은 조건)로 측정. 값은 seed 간 평균 ± 표본표준편차.")


if __name__ == "__main__":
    main(sys.argv[1])
