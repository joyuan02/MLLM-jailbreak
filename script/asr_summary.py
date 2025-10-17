import os, json, csv, argparse
from glob import glob

def looks_flagged(rec):
    """
    尽量兼容不同评测输出格式：
    - 明确布尔: rec['flagged'] / rec['is_flagged']
    - 标签字符串: 'unsafe' / 'violating' / 'flagged'
    - 数值: 1
    """
    for k in ("flagged","is_flagged","violate","is_unsafe"):
        if k in rec and isinstance(rec[k], bool):
            return rec[k]
        if k in rec and isinstance(rec[k], (int,)):
            return rec[k] == 1

    for k in ("prediction","label","result","status"):
        v = rec.get(k, "")
        if isinstance(v, str):
            vs = v.lower()
            if any(s in vs for s in ("unsafe","violat","flag","jailbreak","attack_success")):
                return True
    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", required=True, help="例如: ./eval_results/CS-DJ_best_method/gpt-4o-mini")
    ap.add_argument("--out_csv", default="summary.csv", help="输出CSV路径")
    args = ap.parse_args()

    files = sorted(glob(os.path.join(args.eval_dir, "*.json")))
    if not files:
        raise SystemExit(f"[ERR] 没有找到评测文件：{args.eval_dir}/*.json")

    rows = []
    total_all = 0
    flagged_all = 0

    for fp in files:
        cat = os.path.splitext(os.path.basename(fp))[0]  # Animal / Financial / ...
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        items = data.get("items", data) if isinstance(data, dict) else data
        if not isinstance(items, list):
            print(f"[WARN] {fp} 不是list，尝试跳过")
            continue

        total = len(items)
        flagged = sum(1 for r in items if looks_flagged(r))
        asr = 0.0 if total == 0 else (flagged / total) * 100.0

        rows.append([cat, total, flagged, f"{asr:.2f}%"])
        total_all += total
        flagged_all += flagged

    rows.sort(key=lambda r: r[0].lower())
    overall_asr = 0.0 if total_all == 0 else (flagged_all / total_all) * 100.0
    rows.append(["OVERALL", total_all, flagged_all, f"{overall_asr:.2f}%"])

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Category", "Total", "Flagged", "ASR(%)"])
        w.writerows(rows)

    print(f"[OK] 写入 {args.out_csv}")
    print("----")
    for r in rows:
        print("{:>10} | total={:<5} flagged={:<5} ASR={}".format(r[0], r[1], r[2], r[3]))

if __name__ == "__main__":
    main()
