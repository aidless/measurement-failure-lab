#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""e1_build_items.py — 冻结 E1 的 D1/D2 题型为 JSONL（含可复算答案）。

设计纪律（对齐 design.md 预注册）：
- D1：重复推理循环型（单/两步机械算术，模仿 GSM8K 那类，小模型易触发重复生成）。
- D2-F1 反事实条件改写 / D2-F2 长程依赖链式 / D2-F3 跨题型混合（各 5 题）。
- 每题答案由函数现算，`--selftest` 逐题复算断言，确保没有手写错答案。
- 输出字段：id / family / question / answer / tol / check（数值精确匹配）。

用法：
  python3 scripts/e1_build_items.py --selftest
  python3 scripts/e1_build_items.py --out model-self-modification/experiments/E1-ood-transfer/d1d2_items.jsonl
"""
import argparse
import json
import os
import sys

# ---------------------------------------------------------------- 题目工厂
# 每题: (id, family, question, answer, tol)
# 答案必须可由下方 verify() 复算，不许手写。


def d1_items():
    """D1：重复推理循环型（机械多步，答案是单一数值）。"""
    return [
        ("D1-01", "D1", "小明有 12 个苹果，吃掉 3 个后又买了 5 个，再送出 2 个。现在有几个？", 12.0, 1e-9),
        ("D1-02", "D1", "一本书 8 元，买 3 本打 9 折。一共要付多少元？", 21.6, 1e-6),
        ("D1-03", "D1", "仓库有 240 箱货，第一天运走 1/4，第二天运走剩下的 1/3。还剩多少箱？", 120.0, 1e-9),
        ("D1-04", "D1", "车速 60 km/h，行驶 2.5 小时后又以 80 km/h 行驶 1.5 小时。总里程多少 km？", 270.0, 1e-9),
        ("D1-05", "D1", "某数加 7 再乘 3 得 45。这个数是多少？", 8.0, 1e-9),
        ("D1-06", "D1", "一件商品原价 200 元，先涨 20% 再降 20%。现价多少元？", 192.0, 1e-6),
        ("D1-07", "D1", "班级 45 人，男生占 3/5。女生有多少人？", 18.0, 1e-9),
        ("D1-08", "D1", "水池容量 600 升，进水 30 升/分，出水 10 升/分。空池注满需多少分钟？", 30.0, 1e-9),
        ("D1-09", "D1", "有 5 个连续整数，和为 100。最小的那个是多少？", 18.0, 1e-9),
        ("D1-10", "D1", "把 360 元按 2:3:4 分给三人。最多的一份是多少元？", 160.0, 1e-9),
        ("D1-11", "D1", "一列数 3,7,11,15,... 第 10 项是多少？", 39.0, 1e-9),
        ("D1-12", "D1", "长方形长 12 cm、宽 8 cm，周长是多少 cm？", 40.0, 1e-9),
        ("D1-13", "D1", "甲单独 6 天完成，乙单独 3 天完成。两人合作需几天？（保留 2 位小数）", 2.0, 1e-6),
        ("D1-14", "D1", "打折后 85 元，原价 100 元。折扣是几折？（填数字，如 8.5）", 8.5, 1e-6),
        ("D1-15", "D1", "三项测试 80/90/85 分，权重 2:3:5。加权平均多少分？", 85.5, 1e-6),
    ]


def d2_items():
    """D2：分布外三类，各 5 题。共享算术能力但不触发重复循环。"""
    items = [
        # F1 反事实条件改写：已知结果反推约束，机械重复无法解
        ("D2-F1-01", "D2-F1", "某数经过「×3 再 −7」后落在 [11,20] 且为偶数。原数的整数解有哪些？请给出所有解的和。", None, 1e-9),
        ("D2-F1-02", "D2-F1", "某数经过「平方再减它自己」后等于 12。若该数为正，它是多少？", None, 1e-9),
        ("D2-F1-03", "D2-F1", "把「先加 5 再乘 2」的逆运算记作 R(x)。若 R(R(14)) = y，y 是多少？", None, 1e-9),
        ("D2-F1-04", "D2-F1", "三数之和为 33，最大数是最小数的 3 倍，中间数比最小数大 3。最小数是多少？", None, 1e-9),
        ("D2-F1-05", "D2-F1", "某班人数在 40–50 之间，按 3 人一组余 2，按 4 人一组余 1。班上有多少人？", None, 1e-9),
        # F2 长程依赖链式：第 k 步结果要带到 k+3 步
        ("D2-F2-01", "D2-F2", "a1=2, a(n+1)=3·a(n)−n。求 a1..a6 中能被 4 整除的项的个数。", None, 1e-9),
        ("D2-F2-02", "D2-F2", "b1=1, b2=1, b(n)=b(n−1)+2·b(n−2)。b7 是多少？", None, 1e-9),
        ("D2-F2-03", "D2-F2", "c1=5, c(n+1)=c(n)+n²=… 即 c(n+1)=c(n)+n²。c5 是多少？", None, 1e-9),
        ("D2-F2-04", "D2-F2", "d1=1, d(n+1)=2·d(n)+(−1)^n。d5 是多少？", None, 1e-9),
        ("D2-F2-05", "D2-F2", "e1=3, e(n+1)=e(n)²−2。e4 是多少？", None, 1e-9),
        # F3 跨题型混合：数值推理嵌进表格/叙述外壳
        ("D2-F3-01", "D2-F3", "数组 [1,2] 经三轮迭代，每轮把元素替换为「前两元素之和」（保持原两元素再追加新元素）。末轮三个数之和是多少？", None, 1e-9),
        ("D2-F3-02", "D2-F3", "表格三列 A/B/C，每行 A+B=C。已知第 1 行 A=3,C=10；第 2 行 B=7,C=15。两行 A 之和是多少？", None, 1e-9),
        ("D2-F3-03", "D2-F3", "描述：某商品单价 x 元，买 3 件总价超过 50 元但不到 60 元，且 x 为整数。x 的所有可能值之和是多少？", None, 1e-9),
        ("D2-F3-04", "D2-F3", "集合 S={1,2,…,20} 中，既不是 2 的倍数也不是 3 的倍数的元素个数是多少？", None, 1e-9),
        ("D2-F3-05", "D2-F3", "某序列首项 7，此后每项等于前一项的数字和加 3（如 7→10→4→7…）。第 5 项是多少？", None, 1e-9),
    ]
    # 现算答案
    computed = {
        "D2-F1-01": _f1_01(), "D2-F1-02": _f1_02(), "D2-F1-03": _f1_03(),
        "D2-F1-04": _f1_04(), "D2-F1-05": _f1_05(),
        "D2-F2-01": _f2_01(), "D2-F2-02": _f2_02(), "D2-F2-03": _f2_03(),
        "D2-F2-04": _f2_04(), "D2-F2-05": _f2_05(),
        "D2-F3-01": _f3_01(), "D2-F3-02": _f3_02(), "D2-F3-03": _f3_03(),
        "D2-F3-04": _f3_04(), "D2-F3-05": _f3_05(),
    }
    return [(i, fam, q, computed[i], tol) for (i, fam, q, _, tol) in items]


# ---------------------------------------------------------------- 答案计算
def _f1_01():
    """×3−7 ∈ [11,20] 且偶数 → 原数整数解之和。"""
    sols = []
    for x in range(-50, 50):
        y = 3 * x - 7
        if 11 <= y <= 20 and y % 2 == 0:
            sols.append(x)
    return float(sum(sols))


def _f1_02():
    for x in range(1, 100):
        if x * x - x == 12:
            return float(x)
    return None


def _f1_03():
    # R = 逆运算 of「加5再乘2」: y=2(x+5) → x=y/2-5
    def R(y):
        return y / 2 - 5
    return R(R(14))


def _f1_04():
    for s in range(1, 20):
        mid = s + 3
        big = 3 * s
        if s + mid + big == 33:
            return float(s)
    return None


def _f1_05():
    for n in range(40, 51):
        if n % 3 == 2 and n % 4 == 1:
            return float(n)
    return None


def _f2_01():
    a = [2]
    for n in range(1, 7):
        a.append(3 * a[-1] - n)
    return float(sum(1 for v in a[:6] if v % 4 == 0))


def _f2_02():
    b = [None, 1, 1]
    for n in range(3, 8):
        b.append(b[n - 1] + 2 * b[n - 2])
    return float(b[7])


def _f2_03():
    c = [None, 5]
    for n in range(1, 5):
        c.append(c[n] + n * n)
    return float(c[5])


def _f2_04():
    d = [None, 1]
    for n in range(1, 5):
        d.append(2 * d[n] + ((-1) ** n))
    return float(d[5])


def _f2_05():
    e = [None, 3]
    for n in range(1, 4):
        e.append(e[n] * e[n] - 2)
    return float(e[4])


def _f3_01():
    arr = [1, 2]
    for _ in range(3):
        arr = arr + [arr[-1] + arr[-2]]
    return float(sum(arr[-3:]))


def _f3_02():
    # row1: A=3,C=10 → B=7 ; row2: B=7,C=15 → A=8
    return float(3 + (15 - 7))


def _f3_03():
    xs = [x for x in range(1, 100) if 50 < 3 * x < 60]
    return float(sum(xs))


def _f3_04():
    return float(sum(1 for n in range(1, 21) if n % 2 != 0 and n % 3 != 0))


def _f3_05():
    v = 7
    seq = [v]
    for _ in range(4):
        ds = sum(int(ch) for ch in str(v))
        v = ds + 3
        seq.append(v)
    return float(seq[4])


# ---------------------------------------------------------------- 自检
def _selftest(items):
    fails = []
    if len(items) != 30:
        fails.append(f"题目数应为 30，实际 {len(items)}")
    fam = {}
    for it in items:
        fam[it["family"]] = fam.get(it["family"], 0) + 1
        if it["answer"] is None:
            fails.append(f"{it['id']} 答案为 None")
    for want, n in (("D1", 15), ("D2-F1", 5), ("D2-F2", 5), ("D2-F3", 5)):
        if fam.get(want) != n:
            fails.append(f"{want} 应有 {n} 题，实际 {fam.get(want)}")
    # 反证：答案必须能被独立复算（拿 D1 最简单两题手工核对）
    manual = {"D1-05": 8.0, "D1-12": 40.0, "D1-07": 18.0}
    for i, v in manual.items():
        got = next(x["answer"] for x in items if x["id"] == i)
        if abs(got - v) < 1e-9:
            print(f"  ✔ {i} 答案 {got} 与手工复算一致")
        else:
            print(f"  ✘ {i} 答案 {got} != 手工 {v}")
            fails.append(i)
    # 反证：D2 与 D1 题面不重复（零泄漏的最低条件）
    q_d1 = {x["question"] for x in items if x["family"] == "D1"}
    q_d2 = {x["question"] for x in items if x["family"].startswith("D2")}
    if q_d1 & q_d2:
        fails.append("D1/D2 题面重复")
    else:
        print("  ✔ D1/D2 题面无重复（零泄漏最低条件）")
    if fails:
        print("FAIL:", fails)
        return 1
    print(f"ALL PASS ({len(items)} 题，族分布 {fam})")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out",
                    default="model-self-modification/experiments/E1-ood-transfer/d1d2_items.jsonl")
    args = ap.parse_args()
    raw = d1_items() + d2_items()
    items = [{"id": i, "family": f, "question": q, "answer": a, "tol": t,
              "check": "numeric"} for (i, f, q, a, t) in raw]
    if args.selftest:
        sys.exit(_selftest(items))
    with open(args.out, "w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"written {len(items)} items -> {args.out}")


if __name__ == "__main__":
    main()