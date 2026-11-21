#!/usr/bin/env python3
"""wse_dev_backfill.py — 用本机 ollama 补测 wse-bench 从未测过的 dev 题。

设计纪律：
- 只补 dev split（hidden/fresh 受 provenance_lock / fresh-only-once 保护，不碰）。
- 评分复用 wse-bench-public/eval_self_evolution.py 的确定性 score_contains（不引入 LLM judge）。
- 输出 task_spectra 兼容的 .jsonl（按 (llm, config) 分组，每个模型一个配置）。
- --dry 只列待补测题 + 评分器反证，不调模型；--run 才真正跑。

用法：
  python3 scripts/wse_dev_backfill.py --selftest
  python3 scripts/wse_dev_backfill.py --plan                 # 列 24 题 + split 分布
  python3 scripts/wse_dev_backfill.py --smoke               # 跑 1 题 1 模型验证通道
  python3 scripts/wse_dev_backfill.py --run --out wse-bench-public/results_dev_backfill
"""
import argparse
import collections
import json
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(BASE, ".."))
BENCH = os.path.join(ROOT, "wse-bench-public")
sys.path.insert(0, BENCH)
sys.path.insert(0, BASE)

import eval_self_evolution as ev
import task_spectra as ts

MANIFEST = os.path.join(BENCH, "assets", "bench_manifest.json")
SPECTRA_REPORT = os.path.join(ROOT, "reports", "task_spectra_wse_20261119.json")
DEFAULT_MODELS = ["qwen2.5:3b", "qwen2.5:1.5b", "llama3.2:3b"]


def dev_never_measured():
    """manifest 里 dev split 且 task_spectra 报告标为 never_measured 的题。"""
    m = json.load(open(MANIFEST, encoding="utf-8"))
    rep = json.load(open(SPECTRA_REPORT, encoding="utf-8"))
    never = {b["id"] for b in rep["backfill_targets"] if b.get("reason") == "never_measured"}
    return [t for t in m["tasks"] if t["id"] in never and t["split"] == "dev"]


def score_sub(task, submission):
    """确定性评分：复用 wse 官方 harness。contains 题必须全片段命中才 is_pass。"""
    sc = ev.score_task(task, submission)
    return sc, ev.is_pass(sc, task)


def call_ollama(prompt, model, base_url="http://localhost:11434/v1", timeout=120):
    from openai import OpenAI
    client = OpenAI(api_key="ollama", base_url=base_url)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        timeout=timeout,
    )
    return resp.choices[0].message.content or ""


def run(models, tasks, out_dir, limit=None, base_url="http://localhost:11434/v1"):
    os.makedirs(out_dir, exist_ok=True)
    todo = tasks[:limit] if limit else tasks
    summary = []
    for model in models:
        cfg = "dev-backfill"
        path = os.path.join(out_dir, f"wse_dev_{model.replace(':', '_')}.jsonl")
        rows = []
        for t in todo:
            try:
                sub = call_ollama(t["prompt"], model, base_url=base_url)
            except Exception as e:
                sub = ""
                print(f"  ! {t['id']} {model}: call failed {type(e).__name__}: {e}", file=sys.stderr)
            sc, ok = score_sub(t, sub)
            rows.append({"task_id": t["id"], "llm": model, "config": cfg,
                         "score": round(sc, 4), "passed": ok, "split": t["split"],
                         "rubric": t["rubric"]["type"], "submission": sub[:500]})
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        n_pass = sum(1 for r in rows if r["passed"])
        summary.append({"model": model, "file": path, "n": len(rows), "n_pass": n_pass,
                        "mean_score": round(sum(r["score"] for r in rows) / len(rows), 4) if rows else None})
        print(f"  {model}: {n_pass}/{len(rows)} pass, file={path}")
    return summary


def _selftest():
    fails = []

    def check(name, cond, detail=""):
        if cond:
            print(f"  ✔ {name}")
        else:
            print(f"  ✘ {name}  {detail}")
            fails.append(name)

    t = {"id": "X", "rubric": {"type": "contains", "required": ["alpha", "beta"]}}
    # 反证 1：全命中必须 pass（评分器是活的，不是恒 0 装饰）
    s1, ok1 = score_sub(t, "The ALPHA and Beta are here.")
    check("全命中 pass (red)", ok1 and abs(s1 - 1.0) < 1e-9, f"score={s1} ok={ok1}")
    # 反证 2：部分命中 (0.5) 必须 fail（contains 阈值=1.0，差一点不算过）
    s2, ok2 = score_sub(t, "only alpha here")
    check("部分命中 0.5 判 fail (red)", abs(s2 - 0.5) < 1e-9 and not ok2, f"score={s2} ok={ok2}")
    # 反证 3：全不命中 fail
    s3, ok3 = score_sub(t, "nothing relevant")
    check("全不命中 fail", s3 == 0.0 and not ok3, f"score={s3} ok={ok3}")
    # 反证 4：norm 只折叠空白+小写，不做同义词——同义替换必须抓不到（证明确定性）
    s4, ok4 = score_sub(t, "The first letter and second letter are here.")
    check("同义替换不算命中 (red)", s4 == 0.0 and not ok4, f"score={s4} ok={ok4}")
    # 一致性：补 dev 集全 contains 且全 dev
    tasks = dev_never_measured()
    check("dev 待补测=24 条", len(tasks) == 24, f"got {len(tasks)}")
    check("全是 dev", all(t["split"] == "dev" for t in tasks))
    check("全是 contains", all(t["rubric"]["type"] == "contains" for t in tasks))
    check("不碰 hash 失配题 T029/030/031", not ({"T029", "T030", "T031"} & {t["id"] for t in tasks}))

    print()
    if fails:
        print("FAIL:", fails)
        return 1
    print("ALL PASS")
    return 0


def main():
    ap = argparse.ArgumentParser(description="补测 wse-bench 未测 dev 题（本机 ollama）")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--out", default=os.path.join(BENCH, "results_dev_backfill"))
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    if args.selftest:
        sys.exit(_selftest())
    tasks = dev_never_measured()
    if args.plan:
        by_cap = collections.Counter(t.get("capability") for t in tasks)
        print(f"dev 待补测 {len(tasks)} 条；capability 分布: {dict(by_cap)}")
        print("IDs:", [t["id"] for t in tasks])
        return
    if args.smoke:
        print(f"smoke: model={args.models[0]}, task={tasks[0]['id']}")
        sub = call_ollama(tasks[0]["prompt"], args.models[0])
        sc, ok = score_sub(tasks[0], sub)
        print(f"  submission[:200]: {sub[:200]!r}")
        print(f"  score={sc} pass={ok}")
        return
    if args.run:
        print(f"run: {len(args.models)} models × {len(tasks) if not args.limit else args.limit} tasks")
        summary = run(args.models, tasks, args.out, limit=args.limit)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    ap.error("需要 --selftest / --plan / --smoke / --run 之一")


if __name__ == "__main__":
    main()