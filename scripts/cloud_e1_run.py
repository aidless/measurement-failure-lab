#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cloud_e1_run.py v2 — AMD 云上跑 E1 的 C 臂基线 / A 臂全局 top-k 稀疏编辑。

A 臂实现（与 Tree-Ising / weight-space programming 同族：稀疏离散编辑，不做梯度训练）：
  1. 对 D1（同分布失败题型）做一次梯度累积，得到每个候选参数的梯度；
  2. 把所有候选参数的 |grad| 拉平后做**全局** top-k 排序，选 k 个位置（k 由 --frac 决定，frac×总参数量）；
  3. 在这 k 个位置把 bf16 位型按 int16 视图做 mantissa LSB 步进，步数由 --ulps 决定，方向取 -sign(grad)；
  4. 编辑后重测 D1（同分布）与 D2（分布外三族），算 G1/G2。

纪律：
- 只改 bf16 mantissa LSB，编辑量精确可数（edited_positions / total_params / frac_pct）。
- 不做任何梯度更新（no optimizer），符合"权重空间编程 = 直接读写权重"的口径。
- 阴性/阳性都必须落盘：before/after 每族分数 + 每条样本的 got/expected/output。

用法（云上）：
  python3 cloud_e1_run.py --mode baseline --model M --items I --out c.jsonl
  python3 cloud_e1_run.py --mode edit --model M --items I --out a.jsonl --frac 0.0002 --ulps 3
"""
import argparse
import json
import os
import re
import sys
import collections

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

NUM_RE = re.compile(r"(-?\d+(?:\.\d+)?(?:e-?\d+)?)", re.I)


def last_number(text):
    nums = NUM_RE.findall(str(text).replace(",", ""))
    if not nums:
        return None
    try:
        return float(nums[-1])
    except ValueError:
        return None


ANS_RE = re.compile(r"答案\s*[：:]\s*(-?\d+(?:\.\d+)?(?:e-?\d+)?)", re.I)


def answer_number(text):
    """优先取「答案：<数值>」标记后的数值（fewshot/cot 明确要求这个格式），
    取不到再回退到全文最后一个数字。

    为什么必须这样：fewshot 提示让模型「最后单独一行写成『答案：<数值>』」，
    但编辑后的模型会在给出正确答案后继续重复整段推理，直到 max_new 截断；
    此时 last_number 抓到的是截断处半句话里的数（如「1. 从第 1 行…」的 1），
    而真正的答案早就写在前面的「答案：11」里 —— 这是纯判分器假负。
    """
    tagged = ANS_RE.findall(str(text).replace(",", ""))
    if tagged:
        try:
            return float(tagged[-1])
        except ValueError:
            pass
    return last_number(text)


def load_items(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


D2_FEWSHOT = ("示例：一家店原价 100 元，先打八折再减 10 元，现价多少？\n"
              "推理：100×0.8=80，80−10=70。答案：70\n\n"
              "现在回答：{q}\n请一步一步推理，最后单独一行写成「答案：<数值>」。")
D2_COT = "{q}\n请一步一步推理，最后单独一行写成「答案：<数值>」。"
D2_PLAIN = "{q}\n请只给出最终数值答案，不要解释。"


def build_prompt(tok, question, mode="plain"):
    """D1 用 plain（96 token，与既有读数同尺）；D2 可用 cot/fewshot 放大预算。

    E1 的 plain 口径把 D2 压成 0/15（15 条全被 96 token 截断），而 D2 天花板探针
    在 fewshot+512 token 下读到 12/15 —— 若不换档，G2 的分母恒为 0，"编辑无效"
    与"任务测不出"就无法区分。故 D2 档位做成显式参数，并把口径写进产物。
    """
    if mode == "fewshot":
        content = D2_FEWSHOT.format(q=question)
    elif mode == "cot":
        content = D2_COT.format(q=question)
    else:
        content = D2_PLAIN.format(q=question)
    msgs = [{"role": "user", "content": content}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def gen(model, tok, prompt, max_new=48):
    ids = tok(prompt, return_tensors="pt").to(model.device)
    try:
        with torch.inference_mode():
            out = model.generate(**ids, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        txt = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        del out
        return txt
    finally:
        del ids
        torch.cuda.empty_cache()


def evaluate(model, tok, items, max_new=48, tag="", d2_mode="plain", d2_max_new=None):
    # 强制回 eval 模式并恢复 KV cache：编辑阶段会把模型置为 train 且 use_cache=False，
    # 若此处不显式恢复，"编辑后评估"就与"编辑前评估"不是同一把尺（这会让 A 臂与 R 臂不可比）。
    model.eval()
    if getattr(model, "config", None) is not None:
        model.config.use_cache = True
    rows = []
    for i, it in enumerate(items, 1):
        is_d2 = it["family"].startswith("D2")
        mode = d2_mode if is_d2 else "plain"
        mn = (d2_max_new or max_new) if is_d2 else max_new
        try:
            txt = gen(model, tok, build_prompt(tok, it["question"], mode), mn)
        except Exception as e:
            txt = ""
            print(f"  ! {it['id']}: {type(e).__name__}: {e}", file=sys.stderr)
        got = answer_number(txt)
        got_raw = last_number(txt)
        ok = got is not None and abs(got - float(it["answer"])) <= float(it["tol"])
        ok_raw = got_raw is not None and abs(got_raw - float(it["answer"])) <= float(it["tol"])
        rows.append({"id": it["id"], "family": it["family"], "ok": ok,
                     "got": got, "got_raw_lastnum": got_raw, "ok_raw_lastnum": ok_raw,
                     "expected": float(it["answer"]), "output": txt})
        if i % 10 == 0:
            print(f"  [{tag}] {i}/{len(items)}", file=sys.stderr)
    torch.cuda.empty_cache()
    return rows


def summarize(rows):
    fam = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        fam[r["family"]][1] += 1
        fam[r["family"]][0] += 1 if r["ok"] else 0
    out = {f: {"pass": v[0], "n": v[1], "rate": round(v[0] / v[1], 3) if v[1] else None}
           for f, v in sorted(fam.items())}
    out["_overall"] = {"pass": sum(1 for r in rows if r["ok"]), "n": len(rows)}
    return out


def global_sparse_edit(model, tok, items, frac, ulps, max_len=96, select="grad", seed=0):
    """全局 top-k 稀疏 mantissa LSB 编辑。返回 (edited_positions, total_params)。

    select="grad"  -> 梯度引导（A 臂：定向编辑）
    select="random" -> 同 k 同 ulps 的随机位置（R 臂：等幅度对照，用于把"扰动本身"与"选择"分离）
    """
    d1 = [it for it in items if it["family"] == "D1"]
    params = [(n, p) for n, p in model.named_parameters()
              if p.dtype == torch.bfloat16 and p.dim() >= 2
              and "embed_tokens" not in n and "lm_head" not in n]
    total = sum(p.numel() for _, p in params)
    k = max(1, int(total * frac))
    print(f"[edit] candidate tensors={len(params)} params={total} target k={k} ({frac*100:.4f}%) ulps={ulps} select={select}")

    if select == "random":
        torch.manual_seed(seed)
        edited = 0
        with torch.no_grad():
            for n, p in params:
                kk = int(p.numel() * frac)
                if kk <= 0:
                    continue
                sel = torch.randint(0, p.numel(), (kk,), device=p.device)
                sign = torch.where(torch.rand(kk, device=p.device) > 0.5, 1, -1).to(torch.int16)
                pi = p.view(torch.int16).flatten()
                pi[sel] = pi[sel] - sign * ulps
                edited += kk
                print(f"  [apply-random] {n}: {kk} positions", file=sys.stderr)
        return edited, total

    model.train()
    if hasattr(model, "activation_checkpointing_enable"):
        try:
            model.activation_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.config.use_cache = False
            print("[edit] activation checkpointing on", file=sys.stderr)
        except TypeError:
            model.activation_checkpointing_enable()
            model.config.use_cache = False
            print("[edit] activation checkpointing on (legacy sig)", file=sys.stderr)
        except Exception as e:
            print(f"[edit] ac unavailable: {e}", file=sys.stderr)

    # 关键：7B 的**全量梯度本身就是 15.2 GiB**（weights 15.2 + grads 15.2 ≈ 30.4 GiB > 空闲 29.7 GiB），
    # 必然 OOM。所以不能"先 backward 完再读 grad"，而是用 post-accumulate hook **在梯度一算出来就取走 top-k 并立即置 None**，
    # 这样任一时刻显存里最多只有一两个张量的梯度（实测从 30.4 GiB 降到 ~16 GiB）。
    # 口径变化（须在报告里写明）：每个 D1 题各自贡献 frac/|d1| 的选中位置，跨题取并集；
    #   总编辑量仍 ≈ frac×候选参数量，但"同一个位置被多题选中"只记录最后一次的符号。
    per_item_frac = frac / max(1, len(d1))
    selected = {}

    def make_hook(name, param):
        def _hook(p):
            g = p.grad
            if g is None:
                return
            kk = max(1, min(int(p.numel() * per_item_frac), p.numel()))
            gf = g.detach().float().cpu().abs().flatten()
            topv, idx = torch.topk(gf, kk)
            sign = torch.sign(g.detach().float().cpu().flatten()[idx]).to(torch.int8)
            selected.setdefault(name, []).append((idx, sign))
            del gf, topv
            p.grad = None  # 立即释放，保证峰值不随张量数累积
        return _hook

    hooks = []
    for n, p in params:
        hooks.append(p.register_post_accumulate_grad_hook(make_hook(n, p)))

    model.zero_grad(set_to_none=True)
    ctx = int(max_len)
    for it in d1:
        prompt = build_prompt(tok, it["question"])
        target = str(int(it["answer"])) if float(it["answer"]).is_integer() else str(it["answer"])
        pids = tok(prompt, return_tensors="pt").to(model.device)
        tids = tok(" " + target, return_tensors="pt", add_special_tokens=False).to(model.device)
        # 只回传 prompt 末尾 ctx 个 token + 答案 token：把激活图截短，信号仍是"该题该给的数值"
        pi_cut = pids["input_ids"][:, -ctx:]
        full = torch.cat([pi_cut, tids["input_ids"]], dim=1)
        labels = full.clone()
        labels[:, :pi_cut.shape[1]] = -100
        loss = model(input_ids=full, labels=labels).loss
        (loss / len(d1)).backward()
        print(f"  [grad] {it['id']} loss={float(loss):.4f}", file=sys.stderr)
        del pids, tids, pi_cut, full, labels, loss
        torch.cuda.empty_cache()

    for h in hooks:
        h.remove()

    edited = 0
    with torch.no_grad():
        for n, p in params:
            if n not in selected:
                continue
            idxs = [t[0] for t in selected[n]]
            signs = [t[1] for t in selected[n]]
            idx = torch.cat(idxs)
            sgn = torch.cat(signs)
            # 去重：同一位置被多题选中时，保留最后一次的符号
            uniq, rev = torch.unique(idx, return_inverse=True)
            sgn_last = torch.zeros(uniq.numel(), dtype=torch.int8)
            sgn_last[rev] = sgn
            dev = p.device
            uidx = uniq.to(dev)
            pi = p.view(torch.int16).flatten()
            # bf16 的 int16 视图里，**幅值增大的方向与符号位无关**（实测四象限）：
            #   w=+1.0  +1ulp → +1.0078（幅值↑）   w=-1.0  +1ulp → -1.0078（幅值↑）
            # 而梯度下降要 w_new = w - lr*g。因此 step 必须是 sign(g)*sign(w)，只乘 sign(g)
            # 会让 |w| 沿 +g 方向走 = 梯度上升。符号在 int16 视图上直接读（bit15=1 即负），
            # 不 cast 回 float32，避免 6790 万元素的 down_proj 多一份 272MB 临时量。
            wsign = torch.where(pi[uidx] < 0, -1, 1).to(torch.int16)
            pi[uidx] = pi[uidx] - sgn_last.to(dev).to(torch.int16) * wsign * ulps
            del wsign
            edited += int(uidx.numel())
            print(f"  [apply] {n}: {uidx.numel()} pos (of {idx.numel()} raw)", file=sys.stderr)
            del idxs, signs, idx, sgn, uniq, rev, sgn_last, uidx, pi
            torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    model.eval()
    return edited, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline", "edit"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frac", type=float, default=2e-4)
    ap.add_argument("--ulps", type=int, default=3)
    ap.add_argument("--select", choices=["grad", "random"], default="grad")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--d2-mode", choices=["plain", "cot", "fewshot"], default="plain")
    ap.add_argument("--d2-max-new", type=int, default=0,
                    help="D2 单独预算；0 表示沿用 --max-new")
    ap.add_argument("--max-len", type=int, default=96)
    args = ap.parse_args()

    os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
    items = load_items(args.items)
    print(f"[load] {args.model}", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True)
    model.eval()
    pcount = sum(p.numel() for p in model.parameters())
    print(f"[load] params={pcount}", file=sys.stderr)

    before = evaluate(model, tok, items, args.max_new, "before",
                      d2_mode=args.d2_mode, d2_max_new=args.d2_max_new or None)
    print("[before]", json.dumps(summarize(before), ensure_ascii=False))

    # 口径必须随产物落盘：评估档位、预算、编辑规格都写进 summary，
    # 否则下游（报告/配对差脚本）只能靠文件名猜"这一臂到底怎么测的"。
    payload = {"model": args.model, "mode": args.mode, "params": pcount,
               "eval_spec": {"d1_max_new": args.max_new, "d2_mode": args.d2_mode,
                             "d2_max_new": args.d2_max_new or args.max_new},
               "before": summarize(before), "before_rows": before}

    if args.mode == "edit":
        edited, total = global_sparse_edit(model, tok, items, args.frac, args.ulps, args.max_len,
                                           args.select, args.seed)
        after = evaluate(model, tok, items, args.max_new, "after",
                         d2_mode=args.d2_mode, d2_max_new=args.d2_max_new or None)
        print("[after]", json.dumps(summarize(after), ensure_ascii=False))
        payload.update({
            "edited_positions": edited, "candidate_params": total,
            "frac_pct": round(100.0 * edited / total, 6),
            "ulps": args.ulps,
            "edit_spec": {"select": args.select, "frac": args.frac, "ulps": args.ulps,
                          "seed": args.seed, "max_len": args.max_len},
            "after": summarize(after), "after_rows": after,
        })
        with open(args.out, "w", encoding="utf-8") as f:
            for r in after:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    json.dump(payload, open(args.out + ".summary.json", "w"), ensure_ascii=False, indent=2)
    print("[done]", args.out)


if __name__ == "__main__":
    main()