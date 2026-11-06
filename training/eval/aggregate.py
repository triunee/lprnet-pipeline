"""eval_offline.py 결과(JSON)를 실험·체크포인트 선택 기준별로 묶어 평균±표준편차 표를 만든다.

파일 이름 규칙: <실험>__s<seed>__<선택기준>.json (예: A_orig__s0__b1.json), base는 base.json
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


def fmt(values):
    values = [v * 100 for v in values]
    if len(values) == 1:
        return f"{values[0]:.2f}"
    return f"{statistics.mean(values):.2f} ± {statistics.stdev(values):.2f}"


def main(eval_dir):
    rows = defaultdict(list)
    for p in sorted(Path(eval_dir).glob("*.json")):
        s = json.loads(p.read_text())
        m = re.match(r"^(.+)__s(\d+)__(\w+)$", p.stem)
        key = (m.group(1), m.group(3)) if m else (p.stem, "-")
        rows[key].append(s)

    print("| 실험 | 체크포인트 선택 | seed 수 | 전체 (%) | " + " | ".join(TYPES) + " |")
    print("|---|---|---|---|" + "---|" * len(TYPES))
    for (exp, sel), ss in sorted(rows.items(), key=lambda kv: (kv[0][0] != "base", kv[0])):
        cells = [fmt([s["accuracy"] for s in ss])]
        for t in TYPES:
            vals = [s["by_type"][t]["accuracy"] for s in ss if t in s["by_type"]]
            cells.append(fmt(vals) if vals else "-")
        print(f"| {exp} | {SELECTION.get(sel, sel)} | {len(ss)} | " + " | ".join(cells) + " |")
    total = next(iter(rows.values()))[0]["total"]
    print(f"\ntest {total}장, batch=1(서빙과 같은 조건)로 측정. 값은 seed 간 평균 ± 표본표준편차.")


if __name__ == "__main__":
    main(sys.argv[1])
