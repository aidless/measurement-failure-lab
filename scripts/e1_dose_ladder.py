#!/usr/bin/env python3
"""E1 剂量-反应阶梯：一次性算梯度，然后按多个编辑量逐级施加并测目标函数。

为什么需要它：
    首轮实测已给出两个剂量点（ulps=1 与 3），但两点无法区分"线性"与"阈值"。
    更关键的是，这两个点都显示 A 臂使**目标函数本身上升**（D1 答案 token 平均 logprob
    从 −3.2354 降到 −3.4192，即 loss 从 3.2354 升到 3.4192）。既然步进方向取自
    −sign(∇)，它本应使 loss 下降；观测到上升说明步长已越出一阶近似的有效半径。
    本脚本用一次梯度遍历给出 A 臂的完整剂量曲线（多个 frac，同一 ulps），
    回答：在哪个剂量上 loss 才开始真的下降（若有），以及低剂量处是否仍有损伤。

口径：
    objective(item) = 答案 token 的平均负对数似然 = −mean_lp（与 cloud_e1_run 的训练目标同一量，
    已实测两者逐题完全相等：日志 `[grad] D1-05 loss=6.2903` 与探针 `mean_lp=-6.2903` 一致）。
    Δloss > 0 表示该题被削弱。报告 D1/D2 分组均值与逐题值。

用法：
    python3 e1_dose_ladder.py --model M --items items.jsonl --ulps 1 \
        --fracs 4e-5,8e-5,1.2e-4,1.6e-4 --out {OUT_DIR}/ladder_u1.json
"""
import argparse
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cloud_e1_run import build_prompt  # noqa: E402
from e1_logit_probe import answer_logprob, target_text  # noqa: E402


def measure(model, tok, items, d2_mode):
    model.eval()
    if getattr(model, "config", None) is not None:
        model.config.use_cache = True
    rows = []
    for it in items:
        mode = d2_mode if it["family"].startswith("D2") else "plain"
        lp, n = answer_logprob(model, tok, it, mode)
        rows.append({"id": it["id"], "family": it["family"], "mean_lp": lp, "n_tok": n})
    return rows


def summarize(rows, before):
    b = {r["id"]: r["mean_lp"] for r in before}
    out = {}
    for prefix in ("D1", "D2", "_all"):
        sel = [r for r in rows if prefix == "_all" or r["family"].startswith(prefix)]
        ds = [(-r["mean_lp"]) - (-b[r["id"]]) for r in sel
              if r["mean_lp"] is not None and b.get(r["id"]) is not None]
        if not ds:
            continue
        out[prefix] = {
            "n": len(ds),
            "delta_loss_mean": round(sum(ds) / len(ds), 6),
            "delta_loss_min": round(min(ds), 6),
            "delta_loss_max": round(max(ds), 6),
            "n_worsened": sum(1 for d in ds if d > 0),
        }
    return out


def collect_ranking(model, tok, items, frac_max, max_len):
    """一次梯度遍历，返回 {tensor_name: (idx, sign)} 按 |grad| 降序排列。"""
    d1 = [it for it in items if it["family"] == "D1"]
    params = [(n, p) for n, p in model.named_parameters()
              if p.dtype == torch.bfloat16 and p.dim() >= 2
              and "embed_tokens" not in n and "lm_head" not in n]
    print(f"[rank] tensors={len(params)} frac_max={frac_max}", file=sys.stderr)

    model.train()
    if hasattr(model, "activation_checkpointing_enable"):
        model.activation_checkpointing_enable()
    elif hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if model.config is not None:
        model.config.use_cache = False
    model.zero_grad(set_to_none=True)

    per_item = frac_max / len(d1)
    acc = {}  # name -> list of (idx_cpu, val_cpu)

    def make_hook(name):
        def hook(param):
            with torch.no_grad():
                g = param.grad
                if g is None:
                    return
                kk = max(1, int(param.numel() * per_item))
                gf = g.detach().float().flatten()
                topv, idx = torch.topk(gf.abs(), kk)
                vals = gf[idx]
                acc.setdefault(name, []).append((idx.cpu(), vals.cpu()))
                param.grad = None
                del g, gf, topv, vals, idx
        return hook

    hooks = []
    for n, p in params:
        h = p.register_post_accumulate_grad_hook(make_hook(n))
        hooks.append((p, h))

    ctx = max_len
    for it in d1:
        prompt = build_prompt(tok, it["question"], "plain")
        pids = tok(prompt, return_tensors="pt").to(model.device)
        tids = tok(" " + target_text(it), return_tensors="pt",
                   add_special_tokens=False).to(model.device)
        pi_cut = pids["input_ids"][:, -ctx:]
        full = torch.cat([pi_cut, tids["input_ids"]], dim=1)
        labels = full.clone()
        labels[:, : pi_cut.shape[1]] = -100
        loss = model(input_ids=full, labels=labels).loss
        (loss / len(d1)).backward()
        print(f"  [grad] {it['id']} loss={float(loss):.4f}", file=sys.stderr)
        del pids, tids, pi_cut, full, labels, loss
        torch.cuda.empty_cache()

    for p, h in hooks:
        h.remove()

    ranking = {}
    for n, chunks in acc.items():
        idx = torch.cat([c[0] for c in chunks])
        val = torch.cat([c[1] for c in chunks])
        # 同一位置被多题选中：保留 |grad| 最大的那次（符号随它）
        uniq, inv = torch.unique(idx, return_inverse=True)
        best = torch.zeros(uniq.numel(), dtype=torch.float32)
        pos = torch.arange(idx.numel())
        absval = val.abs()
        order = torch.argsort(absval, descending=True)
        seen = {}
        for o in order.tolist():
            u = int(inv[o])
            if u not in seen:
                seen[u] = float(val[o])
        vals = torch.tensor([seen[int(u)] for u in range(uniq.numel())], dtype=torch.float32)
        ranking[n] = (uniq, torch.sign(vals).to(torch.int16))
        del idx, val, uniq, inv, best, pos, absval, order
    torch.cuda.empty_cache()
    print(f"[rank] tensors with grad={len(ranking)}", file=sys.stderr)
    return ranking, params, hooks


def apply_stage(params, ranking, prev_k, new_k, ulps):
    """把第 prev_k..new_k 名之间的位置施加编辑（累计）。"""
    edited = 0
    for n, p in params:
        if n not in ranking:
            continue
        idx, sgn = ranking[n]
        end = min(new_k.get(n, 0), idx.numel())
        if end <= prev_k.get(n, 0):
            continue
        lo = min(prev_k.get(n, 0), end)
        sel_i = idx[lo:end].to(p.device)
        sel_s = sgn[lo:end].to(p.device).to(torch.int16)
        pi = p.view(torch.int16).flatten()
        # 与 cloud_e1_run.py 同一符号约定：step = sign(g)*sign(w)，见该文件注释
        wsign = torch.where(pi[sel_i] < 0, -1, 1).to(torch.int16)
        pi[sel_i] = pi[sel_i] - sel_s * wsign * ulps
        edited += int(sel_i.numel())
        del sel_i, sel_s, wsign, pi
    torch.cuda.empty_cache()
    return edited


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ulps", type=int, default=1)
    ap.add_argument("--fracs", default="4e-5,8e-5,1.2e-4,1.6e-4")
    ap.add_argument("--max-len", type=int, default=96)
    ap.add_argument("--d2-mode", default="fewshot")
    args = ap.parse_args()

    os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
    fracs = [float(x) for x in args.fracs.split(",")]
    frac_max = max(fracs)
    items = [json.loads(l) for l in open(args.items, encoding="utf-8") if l.strip()]

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True)

    before = measure(model, tok, items, args.d2_mode)
    print("[before]", json.dumps(summarize(before, before), ensure_ascii=False))

    ranking, params, _ = collect_ranking(model, tok, items, frac_max, args.max_len)
    total = sum(p.numel() for _, p in params)

    stages, prev = [], {n: 0 for n in ranking}
    cum = 0
    for f in sorted(fracs):
        target = {n: int(p.numel() * f) for n, p in params if n in ranking}
        added = apply_stage(params, ranking, prev, target, args.ulps)
        cum += added
        prev = target
        model.eval()
        rows = measure(model, tok, items, args.d2_mode)
        st = summarize(rows, before)
        stages.append({"frac": f, "frac_pct": round(100.0 * cum / total, 6),
                       "edited_positions": cum, "stats": st})
        print(f"[stage] frac={f:.2e} edited={cum} "
              f"D1_dloss={st['D1']['delta_loss_mean']:+.4f} "
              f"D2_dloss={st['D2']['delta_loss_mean']:+.4f} "
              f"all={st['_all']['delta_loss_mean']:+.4f}", file=sys.stderr)

    payload = {"model": args.model, "ulps": args.ulps, "d2_mode": args.d2_mode,
               "candidate_params": total, "max_len": args.max_len,
               "before": summarize(before, before), "stages": stages,
               "before_rows": before}
    json.dump(payload, open(args.out, "w"), ensure_ascii=False, indent=2)
    print("[done]", args.out)


if __name__ == "__main__":
    main()
