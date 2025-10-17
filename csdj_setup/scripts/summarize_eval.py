import os, sys, json, glob
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path: sys.path.insert(0, ROOT)
import config

base = os.path.join(config.EVAL_OUTPUT_DIR, config.STRATEGY_NAME, config.OBJECT_MODEL)
files = glob.glob(os.path.join(base, "*.json"))
summary = {}
for fp in files:
    data = json.load(open(fp, "r", encoding="utf-8"))
    flagged = sum(1 for x in data if x.get("flagged") is True)
    total = len(data)
    summary[os.path.basename(fp)] = {"total": total, "flagged": flagged}
print(json.dumps({"eval_dir": base, "files": summary}, ensure_ascii=False, indent=2))
