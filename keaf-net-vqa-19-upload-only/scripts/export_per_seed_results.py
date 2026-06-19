from __future__ import annotations

import json, csv
from pathlib import Path

runs = sorted(Path("runs").glob("okvqa_seed*/eval.json"))
rows = []
for f in runs:
    obj = json.load(open(f, "r", encoding="utf-8"))
    rows.append({
        "run_dir": str(f.parent),
        "seed": f.parent.name.replace("okvqa_seed", ""),
        "okvqa_accuracy": obj.get("okvqa_accuracy", "TO_BE_FILLED_BY_EVALUATOR"),
        "aokvqa_accuracy": obj.get("aokvqa_accuracy", "TO_BE_FILLED_BY_EVALUATOR"),
        "vqa_v2_accuracy": obj.get("vqa_v2_accuracy", "TO_BE_FILLED_BY_EVALUATOR"),
        "gqa_accuracy": obj.get("gqa_accuracy", "TO_BE_FILLED_BY_EVALUATOR"),
    })
with open("per_seed_results.csv", "w", newline="", encoding="utf-8") as out:
    writer = csv.DictWriter(out, fieldnames=["run_dir", "seed", "okvqa_accuracy", "aokvqa_accuracy", "vqa_v2_accuracy", "gqa_accuracy"])
    writer.writeheader()
    writer.writerows(rows)
print("Wrote per_seed_results.csv")
