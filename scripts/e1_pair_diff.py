#!/usr/bin/env python3
"""E1 配对差分析：把各臂 summary.json 折成同一张表。

口径约定（写死在代码里，避免报告与脚本各说一套）：
- **配对基线 = 每臂自己的 `before_rows`**（同一进程、同一 max_new、同一 D2 档位），
  不做跨进程绝对比较——实测两次独立 baseline 的通过题集就不同（见 `--repeat-check`）。
- **G1 = D1 通过数变化，G2 = D2（三族合计）通过数变化**，均为 `after - before`。
- **编辑量以 `edited_positions` 实测值为准，不用标称 `frac*candidate`**：
  A 臂按 `int(numel*frac/len(d1))` 分题取 top-k 再 `unique` 去重，实测落位只有标称的约 1/3，
  因此 A/R 必须按 `edited_positions` 而不是 `frac` 对齐（否则是把"编辑量差 3 倍"读成"选择策略差异"）。

用法：
  python3 scripts/e1_pair_diff.py reports/cloud_e1/*.summary.json
  python3 scripts/e1_pair_diff.py --repeat-check a.json b.json   # 比两次无编辑基线的通过题集
"""
import json
import sys
import argparse


def rows(d, which):
    return d.get(which + "_rows") or []


def pass_ids(d, which):
    return [r["id"] for r in rows(d, which) if r["ok"]]


def fam_pass(d, which, prefix):
    return sum(1 for r in rows(d, which) if r["ok"] and r["family"].startswith(prefix))


SELECT_FROM_NAME = (("_grad_", "grad"), ("_random_", "random"))


def select_of(d, path):
    """臂的 select 只能从产物里读，不能猜。

    当前 `cloud_e1_run.py` 把 `params` 记成参数量（int），没有把 `select` 写进 summary，
    所以已跑完的臂只能按文件名里 `_grad_`/`_random_` 的标识还原，并在输出里标 `from-name`
    以示这是**推断而非读取**。后续脚本已补 `edit_spec` 字段，新臂直接读字段。
    """
    spec = d.get("edit_spec")
    if isinstance(spec, dict) and spec.get("select"):
        return spec["select"], "field"
    for tok, name in SELECT_FROM_NAME:
        if tok in path:
            return name, "from-name"
    return None, "unknown"


def describe(path):
    d = json.load(open(path, encoding="utf-8"))
    mode = d.get("mode", "?")
    b_ids = pass_ids(d, "before")
    a_ids = pass_ids(d, "after") if "after_rows" in d else b_ids
    d1b, d1a = fam_pass(d, "before", "D1"), fam_pass(d, "after", "D1") if "after_rows" in d else fam_pass(d, "before", "D1")
    d2b = fam_pass(d, "before", "D2")
    d2a = fam_pass(d, "after", "D2") if "after_rows" in d else d2b
    lost = sorted(set(b_ids) - set(a_ids))
    gained = sorted(set(a_ids) - set(b_ids))
    return {
        "file": path.split("/")[-1],
        "mode": mode,
        "select": select_of(d, path)[0],
        "select_src": select_of(d, path)[1],
        "ulps": d.get("ulps"),
        "edited": d.get("edited_positions"),
        "frac_pct": d.get("frac_pct"),
        "d1_before": d1b, "d1_after": d1a, "d2_before": d2b, "d2_after": d2a,
        "overall_before": len(b_ids), "overall_after": len(a_ids),
        "G1": d1a - d1b, "G2": d2a - d2b,
        "lost": lost, "gained": gained,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--repeat-check", action="store_true",
                    help="两份 summary 一起给：只比两次无编辑基线的通过题集，判断噪声地板")
    args = ap.parse_args()
    if not args.files:
        ap.error("at least one summary.json")

    if args.repeat_check:
        a, b = (describe(args.files[0]), describe(args.files[1]))
        sa = sorted(set(pass_ids(json.load(open(args.files[0])), "before")))
        sb = sorted(set(pass_ids(json.load(open(args.files[1])), "before")))
        only_a = sorted(set(sa) - set(sb))
        only_b = sorted(set(sb) - set(sa))
        print(f"repeat A: {len(sa)} passed  {sa}")
        print(f"repeat B: {len(sb)} passed  {sb}")
        print(f"only in A: {only_a}")
        print(f"only in B: {only_b}")
        print(f"|symmetric difference| = {len(only_a) + len(only_b)}  -> 单臂差异 ≤ 此值时不可判方向")
        return 0

    print(f"{'file':52s} {'sel':6s} {'ulps':4s} {'edited':>9s} {'D1 before→after':>16s} {'D2 before→after':>16s} {'G1':>3s} {'G2':>3s}")
    for f in args.files:
        r = describe(f)
        sel = f"{r['select']}*" if r["select_src"] == "from-name" else str(r["select"])
        print(f"{r['file']:52s} {sel:6s} {str(r['ulps']):4s} {str(r['edited']):>9s} "
              f"{r['d1_before']:>7d}→{r['d1_after']:<8d} {r['d2_before']:>7d}→{r['d2_after']:<8d} "
              f"{r['G1']:>+3d} {r['G2']:>+3d}")
    print()
    for f in args.files:
        r = describe(f)
        if r["mode"] != "edit":
            continue
        print(f"{r['file']}\n  lost={r['lost']}\n  gained={r['gained']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
