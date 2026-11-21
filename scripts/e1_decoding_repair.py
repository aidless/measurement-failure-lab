"""E1 判决性实验：解码层抑制重复后，生成口径能否跟随 logit 口径。

预注册假设 H1-decode:
  A@u3 编辑后 logit 增益 +2.18(D1)/+2.70(D2)、30/30 题目标串似然全升，
  但贪心生成通过数净 0（18/30→18/30）。离线诊断显示编辑后贪心输出重复率
  0.389→0.634、>=8 连续重复循环 0/30→6/30、23/30 题重复度上升。
  故假设：logit↑ 与生成净 0 的解耦主要由解码层重复循环放大造成，而非编辑
  未改变任务能力。若抑制重复后 (after-before) 生成通过数出现净提升且与 logit
  同向，则 H1-decode 成立；否则 logit 增益属 teacher-forcing 锚定效应。

结构（单次加载，编辑一次，省 2 次 7B 加载 + 2 次梯度遍历）：
  baseline x 3 解码条件 -> global_sparse_edit(一次) -> after x 3 解码条件
  条件: A_raw(rep=1.0,ngram=0) / B_rep1.15 / C_ngram4

verifier（写进产物）:
  v1) A_raw 条件必须复现既有读数 before=18/30, after=18/30；否则口径漂移，结果作废。
  v2) 所有对比在同解码条件的 (before,after) 配对内进行，不跨条件比绝对值。
  v3) 编辑函数沿用云端已验签的 cloud_e1_run.global_sparse_edit（含 bf16 符号修复）。
"""
import argparse, collections, json, re, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/root")
from cloud_e1_run import (build_prompt, answer_number, load_items,
                          global_sparse_edit)

CONDS = [dict(name="A_raw", rep=1.0, nrng=0),
         dict(name="B_rep1.15", rep=1.15, nrng=0),
         dict(name="C_ngram4", rep=1.0, nrng=4)]


def gen_decode(model, tok, prompt, max_new, rep, nrng):
    ids = tok(prompt, return_tensors="pt").to(model.device)
    kw = dict(max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
    if rep != 1.0:
        kw["repetition_penalty"] = rep
    if nrng:
        kw["no_repeat_ngram_size"] = nrng
    try:
        with torch.inference_mode():
            out = model.generate(**ids, **kw)
        return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
    finally:
        del ids
        if "out" in locals():
            del out
        torch.cuda.empty_cache()


def evaluate_cond(model, tok, items, cond, tag):
    model.eval()
    if getattr(model, "config", None) is not None:
        model.config.use_cache = True
    rows = []
    for i, it in enumerate(items, 1):
        is_d2 = it["family"].startswith("D2")
        mode = "fewshot" if is_d2 else "plain"
        mn = 512 if is_d2 else 96
        try:
            txt = gen_decode(model, tok, build_prompt(tok, it["question"], mode),
                             mn, cond["rep"], cond["nrng"])
        except Exception as e:
            txt = ""
            print(f"  ! {it['id']}: {type(e).__name__}: {e}", file=sys.stderr)
        got = answer_number(txt)
        ok = got is not None and abs(got - float(it["answer"])) <= float(it["tol"])
        rows.append({"id": it["id"], "family": it["family"], "ok": ok, "got": got,
                     "expected": float(it["answer"]), "output": txt})
        if i % 15 == 0:
            print(f"  [{tag}] {i}/{len(items)}", file=sys.stderr)
    return rows


def rep_metrics(txt):
    toks = re.findall(r"\S+", txt)
    n = len(toks)
    if n == 0:
        return 0.0, 0
    max_run, cur, prev = 1, 1, None
    for t in toks:
        cur = cur + 1 if t == prev else 1
        max_run = max(max_run, cur)
        prev = t
    return round(1 - len(set(toks)) / n, 3), max_run


def summ(rows):
    out = {}
    for fam in ("D1", "D2", "_overall"):
        sub = rows if fam == "_overall" else [r for r in rows if r["family"].startswith(fam)]
        out[fam] = dict(passed=sum(r["ok"] for r in sub), n=len(sub))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="{MODEL_DIR}/Qwen2.5-7B")
    ap.add_argument("--items", default="{DATA_DIR}/d1d2_items.jsonl")
    ap.add_argument("--frac", type=float, default=1.6e-4)
    ap.add_argument("--ulps", type=int, default=3)
    ap.add_argument("--out", default="{OUT_DIR}/e1_decode_repair.json")
    a = ap.parse_args()
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    items = load_items(a.items)
    result = dict(model=a.model, frac=a.frac, ulps=a.ulps,
                  hypothesis="H1-decode", conds={})

    # ---- baseline: 三条件一次跑完（未编辑）----
    before = {}
    for c in CONDS:
        rows = evaluate_cond(model, tok, items, c, f"before-{c['name']}")
        before[c["name"]] = rows
        print(f"[before {c['name']}] {summ(rows)} ({time.time()-t0:.0f}s)", file=sys.stderr)

    # ---- 编辑一次 ----
    n_ed, total = global_sparse_edit(model, tok, items, a.frac, a.ulps,
                                     select="grad", seed=0)
    result["edited_positions"] = n_ed
    result["candidate_params"] = total

    # ---- after: 三条件（编辑态）----
    after = {}
    for c in CONDS:
        rows = evaluate_cond(model, tok, items, c, f"after-{c['name']}")
        after[c["name"]] = rows
        print(f"[after {c['name']}] {summ(rows)} ({time.time()-t0:.0f}s)", file=sys.stderr)

    # ---- 汇总 ----
    for c in CONDS:
        name = c["name"]
        rb = [rep_metrics(r["output"]) for r in before[name]]
        ra = [rep_metrics(r["output"]) for r in after[name]]
        sb, sa = summ(before[name]), summ(after[name])
        result["conds"][name] = dict(
            rep=c["rep"], ngram=c["nrng"], before=sb, after=sa,
            delta_overall=sa["_overall"]["passed"] - sb["_overall"]["passed"],
            mean_rep_before=round(sum(x[0] for x in rb) / len(rb), 3),
            mean_rep_after=round(sum(x[0] for x in ra) / len(ra), 3),
            cycles_before=sum(1 for x in rb if x[1] >= 8),
            cycles_after=sum(1 for x in ra if x[1] >= 8),
            before_rows=before[name], after_rows=after[name])

    # verifier v1: A_raw 复现断言（诚实记录，不强行通过）
    raw = result["conds"]["A_raw"]
    result["verifier"] = dict(
        v1_raw_reproduces_prior=(
            raw["before"]["_overall"]["passed"] == 18 and
            raw["after"]["_overall"]["passed"] == 18),
        v1_observed=dict(before=raw["before"]["_overall"]["passed"],
                         after=raw["after"]["_overall"]["passed"],
                         expected=dict(before=18, after=18)),
        v2_paired_within_cond=True,
        v3_edit_fn="cloud_e1_run.global_sparse_edit (bf16 sign-fixed)")
    json.dump(result, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"[done] wrote {a.out} v1={result['verifier']['v1_raw_reproduces_prior']} "
          f"({time.time()-t0:.0f}s)", file=sys.stderr)


if __name__ == "__main__":
    main()
