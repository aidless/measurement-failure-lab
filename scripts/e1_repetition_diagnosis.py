"""E1 口径冲突离线诊断：量化编辑前后贪心输出的重复循环程度。
verifier: rep_ratio = 1 - unique_tokens/total_tokens; max_run = 单个 token 最长连续重复次数。
判据：若 after 的 rep_ratio/max_run 系统性高于 before，则 logit↑ 与生成↓ 的解耦由解码层重复循环放大解释。
"""
import json, re, sys
from collections import Counter

TOK = re.compile(r"\S+")

def metrics(text):
    toks = TOK.findall(text)
    n = len(toks)
    if n == 0:
        return dict(n=0, rep_ratio=0.0, max_run=0, dup_top=0)
    rep_ratio = 1 - len(set(toks)) / n
    max_run, cur, prev = 1, 1, None
    for t in toks:
        if t == prev:
            cur += 1
        else:
            cur = 1
        max_run = max(max_run, cur)
        prev = t
    dup_top = Counter(toks).most_common(1)[0][1]
    return dict(n=n, rep_ratio=round(rep_ratio, 3), max_run=max_run, dup_top=dup_top)

def main(path):
    d = json.load(open(path))
    bm = {r["id"]: r for r in d["before_rows"]}
    am = {r["id"]: r for r in d["after_rows"]}
    rows = []
    for k in sorted(bm):
        b, a = metrics(bm[k]["output"]), metrics(am[k]["output"])
        rows.append((k, b, a, bm[k]["ok"], am[k]["ok"]))
    print("%-10s %20s %20s %6s %6s" % ("id", "before(r/max/dup)", "after(r/max/dup)", "okB", "okA"))
    for k, b, a, ob, oa in rows:
        print("%-10s %6.3f/%2d/%3d %12.3f/%2d/%3d %6s %6s" % (
            k, b["rep_ratio"], b["max_run"], b["dup_top"],
            a["rep_ratio"], a["max_run"], a["dup_top"], ob, oa))
    rb = [b["rep_ratio"] for _, b, _, _, _ in rows]
    ra = [a["rep_ratio"] for _, _, a, _, _ in rows]
    mb = [b["max_run"] for _, b, _, _, _ in rows]
    ma = [a["max_run"] for _, _, a, _, _ in rows]
    cyc_b = sum(1 for x in mb if x >= 8)
    cyc_a = sum(1 for x in ma if x >= 8)
    print("\n=== 汇总（配对 30 题）===")
    print("mean rep_ratio: before %.3f -> after %.3f (delta %+.3f)" % (
        sum(rb)/len(rb), sum(ra)/len(ra), sum(ra)/len(ra)-sum(rb)/len(rb)))
    print("mean max_run  : before %.2f -> after %.2f (delta %+.2f)" % (
        sum(mb)/len(mb), sum(ma)/len(ma), sum(ma)/len(ma)-sum(mb)/len(mb)))
    print("出现 >=8 连续重复循环: before %d/30 -> after %d/30" % (cyc_b, cyc_a))
    up = sum(1 for _, b, a, _, _ in rows if a["rep_ratio"] > b["rep_ratio"] + 0.05)
    down = sum(1 for _, b, a, _, _ in rows if b["rep_ratio"] > a["rep_ratio"] + 0.05)
    print("重复度上升>0.05: %d/30; 下降>0.05: %d/30" % (up, down))

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "reports/cloud_e1/a_fs_u3_judgefix.jsonl.summary.json")
