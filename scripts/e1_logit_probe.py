#!/usr/bin/env python3
"""E1 逐题 logit 探针：用连续量取代 30 题二值通过数。

为什么需要它（来自首轮实测的教训）：
    30 题、D1 15 题的二值通过数分辨率是 1/15 ≈ 6.7 个百分点，Δ=1 就是最小可分辨单位。
    A 臂（梯度 top-k）的 G1=−1 落在随机臂四种子的 [−2,−1] 区间内，因此"位置选择是否有用"
    在这把尺上永远判不开。本探针改为测**目标数值 token 的平均对数似然**，每道题给出一个
    连续读数，把"打中脆弱题"从"通过/不通过"变成"损伤多深"。

口径：
    score(item) = 目标串（" " + 答案）各 token 的 logprob 的**均值**（长度归一，避免长答案天然低分）
    Δ = after − before，负值 = 该题被削弱
    同时报告 D1/D2 分组统计与逐题表；不重跑生成（生成读数见 cloud_e1_run.py）。

用法：
    python3 e1_logit_probe.py --model M --items items.jsonl --mode baseline --out base.json
    python3 e1_logit_probe.py --model M --items items.jsonl --mode edit --select random --frac 5.202e-5 --ulps 3 --seed 1 --out r1.json
    python3 e1_logit_probe.py --model M --items items.jsonl --mode edit --select grad   --frac 1.6e-4  --ulps 3 --out a.json
"""
import argparse
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cloud_e1_run import global_sparse_edit, build_prompt  # noqa: E402


def target_text(it):
    a = float(it["answer"])
    return str(int(a)) if a.is_integer() else str(a)


def answer_logprob(model, tok, it, mode="plain"):
    """目标串 token 的平均 logprob。返回 (mean_lp, n_tokens)。"""
    prompt = build_prompt(tok, it["question"], mode)
    pids = tok(prompt, return_tensors="pt").to(model.device)
    tids = tok(" " + target_text(it), return_tensors="pt", add_special_tokens=False).to(model.device)
    full = torch.cat([pids["input_ids"], tids["input_ids"]], dim=1)
    labels = full.clone()
    labels[:, : pids["input_ids"].shape[1]] = -100
    with torch.inference_mode():
        out = model(input_ids=full, labels=labels)
        # transformers 返回 mean loss（reduction=mean 且已按 -100 剔除），直接转 logprob
        mean_lp = -float(out.loss)
    n = int((labels != -100).sum())
    del pids, tids, full, labels, out
    torch.cuda.empty_cache()
    return mean_lp, n


def probe(model, tok, items, d2_mode="fewshot"):
    rows = []
    for it in items:
        mode = d2_mode if it["family"].startswith("D2") else "plain"
        try:
            lp, n = answer_logprob(model, tok, it, mode)
        except Exception as e:
            print(f"  ! {it['id']}: {type(e).__name__}: {e}", file=sys.stderr)
            lp, n = None, 0
        rows.append({"id": it["id"], "family": it["family"], "mean_lp": lp, "n_tok": n})
    return rows


def stats(rows, before=None):
    out = {}
    for prefix in ("D1", "D2", "_all"):
        sel = [r for r in rows if (prefix == "_all" or r["family"].startswith(prefix))]
        vals = [r["mean_lp"] for r in sel if r["mean_lp"] is not None]
        if not vals:
            continue
        out[prefix] = {"n": len(vals), "mean": round(sum(vals) / len(vals), 6),
                      "min": round(min(vals), 6), "max": round(max(vals), 6)}
    if before is not None:
        b = {r["id"]: r["mean_lp"] for r in before}
        for prefix in ("D1", "D2", "_all"):
            sel = [r for r in rows if (prefix == "_all" or r["family"].startswith(prefix))]
            ds = [r["mean_lp"] - b[r["id"]] for r in sel
                  if r["mean_lp"] is not None and b.get(r["id"]) is not None]
            if not ds:
                continue
            out.setdefault(prefix, {}).update({
                "delta_mean": round(sum(ds) / len(ds), 6),
                "delta_min": round(min(ds), 6),
                "delta_max": round(max(ds), 6),
                "n_weakened": sum(1 for d in ds if d < 0),
                "n_strengthened": sum(1 for d in ds if d > 0),
            })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["baseline", "edit"], default="baseline")
    ap.add_argument("--select", choices=["grad", "random"], default="grad")
    ap.add_argument("--frac", type=float, default=1.6e-4)
    ap.add_argument("--ulps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-len", type=int, default=96)
    ap.add_argument("--d2-mode", default="fewshot")
    args = ap.parse_args()

    os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
    items = [json.loads(l) for l in open(args.items, encoding="utf-8") if l.strip()]
    print(f"[load] {args.model}", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True)
    model.eval()

    before = probe(model, tok, items, args.d2_mode)
    print("[before]", json.dumps(stats(before), ensure_ascii=False))

    payload = {"model": args.model, "mode": args.mode,
               "spec": {"select": args.select, "frac": args.frac, "ulps": args.ulps,
                        "seed": args.seed, "max_len": args.max_len, "d2_mode": args.d2_mode},
               "before": stats(before), "before_rows": before}

    if args.mode == "edit":
        edited, total = global_sparse_edit(model, tok, items, args.frac, args.ulps,
                                           args.max_len, args.select, args.seed)
        model.eval()
        if getattr(model, "config", None) is not None:
            model.config.use_cache = True
        after = probe(model, tok, items, args.d2_mode)
        print("[after]", json.dumps(stats(after, before), ensure_ascii=False))
        payload.update({"edited_positions": edited, "candidate_params": total,
                        "frac_pct": round(100.0 * edited / total, 6),
                        "after": stats(after, before), "after_rows": after})
        with open(args.out, "w", encoding="utf-8") as f:
            for r in after:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    json.dump(payload, open(args.out + ".summary.json", "w"), ensure_ascii=False, indent=2)
    print("[done]", args.out)


if __name__ == "__main__":
    main()
