#!/usr/bin/env python3
"""exclude_constant_recompute.py — 把恒定任务移出计分分母后重算 L1/L2/L3。

背景：evalkit 有 18 条恒 1.0（adv10+audit8），这些题在所有配置下同分，
不提供任何区分度，却在计分分母里。本脚本复用 task_spectra 的读取口径，
把恒定任务移出后重算，避免手算 46−18。

方向：const==1（天花板饱和，太简单）、const==0（地板塌陷，太难）、both。
默认 const==1（evalkit 场景）。

用法：
  python3 scripts/exclude_constant_recompute.py --selftest
  python3 scripts/exclude_constant_recompute.py --results agent-evalkit/results \
      --direction const1 --label evalkit_drop18 --out reports/evalkit_drop18_recompute.json
"""
import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import task_spectra as ts


def constant_ids(configs, direction):
    """找 power>=POWER_MIN 且得分恒定的任务 id。direction: const1/const0/both。"""
    task_cfg = collections.defaultdict(dict)
    for i, c in enumerate(configs):
        for t, v in c["scores"].items():
            task_cfg[t][i] = v
    out = {}
    for t, d in task_cfg.items():
        if len(d) < ts.POWER_MIN:
            continue
        vals = set(d.values())
        if len(vals) != 1:
            continue
        v = next(iter(vals))
        if (direction == "const1" and v == 1.0) or (direction == "const0" and v == 0.0) or direction == "both":
            out[t] = v
    return dict(sorted(out.items()))


def recompute(configs, exclude):
    """移出 exclude 集合里的任务后重算口径（复用 task_spectra.POWER_MIN）。"""
    cc = []
    for c in configs:
        d = {k: v for k, v in c["scores"].items() if k not in exclude}
        if len(d) >= 2:
            cc.append(d)
    task_cfg = collections.defaultdict(set)
    for i, scores in enumerate(cc):
        for t in scores:
            task_cfg[t].add(i)
    L1 = len(task_cfg)
    L2 = [t for t in task_cfg if len(task_cfg[t]) >= ts.POWER_MIN]
    const = [t for t in L2 if len({cc[i][t] for i in task_cfg[t]}) == 1]
    L3 = [t for t in L2 if t not in const]
    # power 分布
    power_dist = collections.Counter(len(v) for v in task_cfg.values())
    return {
        "L1_after": L1,
        "L2_after": len(L2),
        "L3_after": len(L3),
        "const_after": len(const),
        "n_configs": len(cc),
        "inflation_after": round(L1 / len(L2), 3) if L2 else None,
        "power_distribution": dict(sorted(power_dist.items())),
        "L2_ids": sorted(L2),
        "const_ids_after": sorted(const),
    }


def analyze(results_dir, direction):
    configs, skipped = ts.load_configs(results_dir)
    const1 = constant_ids(configs, "const1")
    const0 = constant_ids(configs, "const0")
    # 原始口径
    orig = recompute(configs, set())
    # 按方向排除
    exclude = {"const1": const1, "const0": const0, "both": {**const1, **const0}}[direction]
    pur = recompute(configs, set(exclude))
    return {
        "results_dir": os.path.abspath(results_dir),
        "direction": direction,
        "n_configs_raw": len(configs),
        "skipped": skipped[:20],
        "constant_found": {"const1": const1, "const0": const0},
        "excluded_ids": sorted(exclude),
        "n_excluded": len(exclude),
        "before": {k: v for k, v in orig.items() if k not in ("L2_ids", "const_ids_after")},
        "after": pur,
    }


def _selftest():
    fails = []

    def check(name, cond, detail=""):
        if cond:
            print(f"  ✔ {name}")
        else:
            print(f"  ✘ {name}  {detail}")
            fails.append(name)

    # 反证 1：注入一条恒 1.0 任务，const1 必须抓到它（方向正确）
    c = [{"scores": {"T1": 1.0, "T2": 1.0, "T3": 0.0}, "meta": {"config": "a"}},
         {"scores": {"T1": 1.0, "T2": 0.0, "T3": 1.0}, "meta": {"config": "b"}},
         {"scores": {"T1": 1.0, "T2": 0.0, "T3": 0.0}, "meta": {"config": "c"}}]
    c1 = constant_ids(c, "const1")
    c0 = constant_ids(c, "const0")
    check("T1 恒1.0 被 const1 抓到 (red)", "T1" in c1, f"const1={c1}")
    check("T1 不被 const0 误抓 (red)", "T1" not in c0, f"const0={c0}")

    # 反证 2：排除恒 1.0 后 const_after 必须为 0（真排掉了）
    pur = recompute(c, set(c1))
    check("排除后 const_after=0 (red)", pur["const_after"] == 0, f"got {pur['const_after']}")
    check("排除后 L1=2 (red)", pur["L1_after"] == 2, f"got {pur['L1_after']}")
    check("排除后 L3=2 (T2/T3 仍有区分度)", pur["L3_after"] == 2, f"got {pur['L3_after']}")

    # 反证 3：const0 方向能抓全 0 恒定列（地板塌陷）
    c2 = [{"scores": {"T1": 0.0, "T2": 0.0}, "meta": {"config": "a"}},
          {"scores": {"T1": 0.0, "T2": 1.0}, "meta": {"config": "b"}},
          {"scores": {"T1": 0.0, "T2": 0.0}, "meta": {"config": "c"}}]
    c0b = constant_ids(c2, "const0")
    check("T1 恒0 被 const0 抓到 (red)", "T1" in c0b, f"const0={c0b}")

    print()
    if fails:
        print("FAIL:", fails)
        return 1
    print("ALL PASS")
    return 0


def main():
    ap = argparse.ArgumentParser(description="移出恒定任务后重算覆盖率口径")
    ap.add_argument("--results", help="results 目录")
    ap.add_argument("--direction", choices=["const1", "const0", "both"], default="const1")
    ap.add_argument("--label", default="recompute", help="报告标签")
    ap.add_argument("--out", help="输出 JSON 路径")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(_selftest())
    if not args.results:
        ap.error("需要 --results 或 --selftest")

    r = analyze(args.results, args.direction)
    r["label"] = args.label
    text = json.dumps(r, ensure_ascii=False, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"written: {args.out}")
    print(text)


if __name__ == "__main__":
    main()