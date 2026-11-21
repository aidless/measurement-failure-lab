#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""d2_ceiling_probe.py — D2 天花板检查：D2 全 0 到底是"编辑无效"还是"D2 本身解不出"。

做法：放开 CoT（不禁止解释、不截断到 96 token），给足 512 new tokens，
只判"最终数值是否落在答案里"，与 E1 判分同一口径（last_number + tol）。
三档提示：
  plain  = 只要求给数值（与 E1 同 prompt）
  cot    = 明确允许逐步推理后再给最终数值
  fewshot= 给一个同类示例（同族题型，非本题）
输出每族族内通过率；若 D2 在 cot/fewshot 下明显抬升，则 D2 全 0 是"prompt/预算"问题而非"任务不可解"。
"""
import argparse
import json
import re
import sys
import collections

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

NUM_RE = re.compile(r"(-?\d+(?:\.\d+)?(?:e-?\d+)?)", re.I)

PROMPTS = {
    "plain": "{q}\n请只给出最终数值答案，不要解释。",
    "cot": "{q}\n请一步一步推理，最后单独一行写成「答案：<数值>」。",
    "fewshot": "示例：一家店原价 100 元，先打八折再减 10 元，现价多少？\n推理：100×0.8=80，80−10=70。答案：70\n\n现在回答：{q}\n请一步一步推理，最后单独一行写成「答案：<数值>」。",
}


def last_number(text):
    nums = NUM_RE.findall(str(text).replace(",", ""))
    if not nums:
        return None
    try:
        return float(nums[-1])
    except ValueError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--families", default="D2-F1,D2-F2,D2-F3")
    args = ap.parse_args()

    items = [json.loads(l) for l in open(args.items, encoding="utf-8") if l.strip()]
    fams = [f.strip() for f in args.families.split(",") if f.strip()]
    sel = [it for it in items if it["family"] in fams]
    print(f"[probe] items={len(sel)} families={fams}", file=sys.stderr)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True)
    model.eval()

    out = {}
    for mode, tpl in PROMPTS.items():
        rows = []
        for it in sel:
            msgs = [{"role": "user", "content": tpl.format(q=it["question"])}]
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            ids = tok(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                o = model.generate(**ids, max_new_tokens=args.max_new, do_sample=False,
                                   pad_token_id=tok.eos_token_id)
            txt = tok.decode(o[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
            # 优先取「答案：」后面的数
            m = re.search(r"答案[:：]\s*([^\n]*)", txt)
            got = last_number(m.group(1)) if m else last_number(txt)
            ok = got is not None and abs(got - float(it["answer"])) <= float(it["tol"])
            rows.append({"id": it["id"], "family": it["family"], "mode": mode, "ok": ok,
                         "got": got, "expected": float(it["answer"]), "output": txt[:400]})
            del ids, o
            torch.cuda.empty_cache()
        fam = collections.defaultdict(lambda: [0, 0])
        for r in rows:
            fam[r["family"]][1] += 1
            fam[r["family"]][0] += 1 if r["ok"] else 0
        out[mode] = {"per_family": {f: {"pass": v[0], "n": v[1]} for f, v in sorted(fam.items())},
                     "rows": rows}
        print(f"[{mode}]", json.dumps(out[mode]["per_family"], ensure_ascii=False))

    json.dump({"model": args.model, "max_new": args.max_new, "modes": out},
              open(args.out, "w"), ensure_ascii=False, indent=2)
    print("[done]", args.out)


if __name__ == "__main__":
    main()