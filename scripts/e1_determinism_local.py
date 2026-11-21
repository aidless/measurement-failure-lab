#!/usr/bin/env python3
"""E1 确定性闸 · 本地替代验证（RTX 3060 + Qwen2.5-1.5B）。

云端 7B 实例销毁后，用本地小模型直接测同一实现层面的性质：
同进程、同 eval_spec、贪心解码重跑 N 次，通过集与逐题文本哈希是否完全一致。

判据（与 e1_determinism_gate.py 相同）：
  DETERMINISTIC   before 与 after 重跑都逐题一致
  NONDET_GEN      before 一致、after 不一致（编辑状态引入非确定）
  NONDET_BASE     before 就不一致（生成本身非确定）

只读：不写远端、不联网、不改落盘权重（编辑只在内存张量上）。
"""
import argparse, hashlib, json, sys, time


def txt_hash(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def compare(runs):
    base = runs[0]
    ok0 = {r["id"] for r in base if r["ok"]}
    res = {"n_runs": len(runs), "n_items": len(base),
           "total_ok": [sum(1 for r in rn if r["ok"]) for rn in runs], "pairs": []}
    for j in range(1, len(runs)):
        okj = {r["id"] for r in runs[j] if r["ok"]}
        res["pairs"].append({"vs": j, "sym_diff": len(ok0 ^ okj),
                             "only_base": sorted(ok0 - okj), "only_this": sorted(okj - ok0)})
    h0 = {r["id"]: r["hash"] for r in base}
    res["text_mismatch_ids"] = sorted({r["id"] for rn in runs[1:] for r in rn
                                       if r["hash"] != h0.get(r["id"])})
    res["text_mismatch_n"] = len(res["text_mismatch_ids"])
    return res


def verdict(before_cmp, after_cmp):
    if (before_cmp["text_mismatch_n"] > 0 or
            any(p["sym_diff"] > 0 for p in before_cmp["pairs"])):
        return "NONDET_BASE", "before 重跑就不一致：生成本身非确定"
    if (after_cmp["text_mismatch_n"] > 0 or
            any(p["sym_diff"] > 0 for p in after_cmp["pairs"])):
        return "NONDET_GEN", "before 一致但 after 不一致：编辑状态引入非确定"
    return "DETERMINISTIC", "before/after 重跑均逐题一致：贪心解码确定，『±1~3 噪声带』假设不成立"


def run_once(model, tok, items, tag, answer_number, build_prompt, gen,
             d1_max_new=96, d2_mode="fewshot", d2_max_new=512):
    import torch
    model.eval()
    model.config.use_cache = True
    rows = []
    for i, it in enumerate(items, 1):
        is_d2 = it["family"].startswith("D2")
        mode = d2_mode if is_d2 else "plain"
        mn = d2_max_new if is_d2 else d1_max_new
        txt = gen(model, tok, build_prompt(tok, it["question"], mode), mn)
        got = answer_number(txt)
        ok = got is not None and abs(got - float(it["answer"])) <= float(it["tol"])
        rows.append({"id": it["id"], "family": it["family"], "ok": bool(ok),
                     "got": got, "expected": float(it["answer"]),
                     "hash": txt_hash(txt), "output": txt})
    torch.cuda.empty_cache()
    return rows


def _rows(ids, ok_ids, salt):
    return [{"id": i, "family": "D1", "ok": i in ok_ids, "got": 1, "expected": 1.0,
             "hash": hashlib.sha256(f"{salt}:{i}".encode()).hexdigest()[:16]}
            for i in ids]


def selftest():
    fails, ids = [], ["a", "b", "c", "d"]
    r1 = _rows(ids, {"a", "c"}, "x")
    same = compare([r1, _rows(ids, {"a", "c"}, "x")])
    if same["pairs"][0]["sym_diff"] != 0 or same["text_mismatch_n"] != 0:
        fails.append("同内容两次应判一致")
    diff = compare([r1, _rows(ids, {"a", "b"}, "x")])
    if diff["pairs"][0]["sym_diff"] != 2: fails.append("通过集差应为 2")
    diff2 = compare([r1, _rows(ids, {"a", "c"}, "y")])
    if diff2["text_mismatch_n"] != 4: fails.append("文本全不同应报 4")
    both = compare([r1, _rows(ids, {"a", "b"}, "y")])
    if verdict(both, same)[0] != "NONDET_BASE": fails.append("before 不一致应判 NONDET_BASE")
    if verdict(same, both)[0] != "NONDET_GEN": fails.append("after 不一致应判 NONDET_GEN")
    if verdict(same, same)[0] != "DETERMINISTIC": fails.append("全一致应判 DETERMINISTIC")
    if fails:
        print("SELFTEST FAIL:", "; ".join(fails)); return False
    print("SELFTEST PASS"); return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--model", default="models/Qwen2.5-1.5B")
    ap.add_argument("--items", default="model-self-modification/experiments/E1-ood-transfer/d1d2_items.jsonl")
    ap.add_argument("--n-before", type=int, default=2)
    ap.add_argument("--n-after", type=int, default=2)
    ap.add_argument("--frac", type=float, default=1.6e-4)
    ap.add_argument("--ulps", type=int, default=3)
    ap.add_argument("--out", default="reports/cloud_e1/e1_determinism_local.json")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)

    sys.path.insert(0, "scripts")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from cloud_e1_run import answer_number, build_prompt, gen, global_sparse_edit, load_items

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16,
                                                 device_map="cuda").eval()
    items = load_items(a.items)
    spec = {"d1_max_new": 96, "d2_mode": "fewshot", "d2_max_new": 512}
    out = {"model": a.model, "spec": spec, "frac": a.frac, "ulps": a.ulps,
           "hardware": "RTX 3060 Laptop 6GB", "read_only": True, "network_calls": 0,
           "ran_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "substitute_for": "e1_determinism_gate.py (cloud 7B instance destroyed)"}

    before = []
    for k in range(a.n_before):
        rows = run_once(model, tok, items, f"before#{k}", answer_number,
                        build_prompt, gen, **spec)
        before.append(rows)
        print(f"[before#{k}] ok={sum(1 for r in rows if r['ok'])}/30 "
              f"({time.time()-t0:.0f}s)", file=sys.stderr, flush=True)
    out["before_runs"] = before
    out["before_cmp"] = compare(before)

    n_ed, total = global_sparse_edit(model, tok, items, a.frac, a.ulps,
                                     select="grad", seed=0)
    out["edited_positions"], out["candidate_params"] = n_ed, total
    print(f"[edit] {n_ed}/{total} ({time.time()-t0:.0f}s)", file=sys.stderr, flush=True)

    after = []
    for k in range(a.n_after):
        rows = run_once(model, tok, items, f"after#{k}", answer_number,
                        build_prompt, gen, **spec)
        after.append(rows)
        print(f"[after#{k}] ok={sum(1 for r in rows if r['ok'])}/30 "
              f"({time.time()-t0:.0f}s)", file=sys.stderr, flush=True)
    out["after_runs"] = after
    out["after_cmp"] = compare(after)

    out["verdict"], out["verdict_reason"] = verdict(out["before_cmp"], out["after_cmp"])
    out["elapsed_s"] = round(time.time() - t0, 1)
    json.dump(out, open(a.out, "w"), ensure_ascii=False)
    print(f"[done] {out['verdict']} :: {out['verdict_reason']} :: {a.out} "
          f"({out['elapsed_s']}s)", file=sys.stderr)


if __name__ == "__main__":
    main()
