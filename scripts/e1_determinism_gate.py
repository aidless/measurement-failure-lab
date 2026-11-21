#!/usr/bin/env python3
"""E1 确定性闸：同一模型/同一编辑状态，贪心解码重跑多次，是否必然得到同一通过集与同一文本？

背景：v5 报告把"两次独立 baseline 答对集合不同"记成"生成在 96 token 上限下不稳定"，
并把编辑净效应记成"落在 ±1~3 噪声带"。离线 join 显示那批差异全部可归因到脚本版本
（output 截断 300 字符 / 判分器 last_number vs answer_number / d2_max_new），
没有一项需要"运行时随机"来解释。本闸直接测这件事：同进程、同 eval_spec、重跑 N 次。

预注册判据：
  DETERMINISTIC  before 重跑一致 且 after 重跑一致  -> 贪心确定，"噪声带"假设不成立
  NONDET_GEN     before 一致、after 不一致          -> 编辑状态引入非确定性
  NONDET_BASE    before 就不一致                    -> 生成本身非确定，全部跨臂比较作废

只读：不写远端、不联网、不改权重落盘。产物写 --out。
顶层不 import 重依赖，selftest 可在任何机器上跑（不触盘）。
"""
import argparse, hashlib, json, sys, time


def txt_hash(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def compare(runs):
    """逐题比较 N 次运行：通过集对称差 + 文本哈希不一致。"""
    base = runs[0]
    ok0 = {r["id"] for r in base if r["ok"]}
    res = {"n_runs": len(runs), "n_items": len(base),
           "total_ok": [sum(1 for r in rn if r["ok"]) for rn in runs], "pairs": []}
    for j in range(1, len(runs)):
        okj = {r["id"] for r in runs[j] if r["ok"]}
        res["pairs"].append({"vs": j, "sym_diff": len(ok0 ^ okj),
                             "only_base": sorted(ok0 - okj),
                             "only_this": sorted(okj - ok0)})
    h0 = {r["id"]: r["hash"] for r in base}
    mismatch_ids = sorted({r["id"] for rn in runs[1:] for r in rn
                           if r["hash"] != h0.get(r["id"])})
    res["text_mismatch_ids"] = mismatch_ids
    res["text_mismatch_n"] = len(mismatch_ids)
    return res


def verdict(before_cmp, after_cmp):
    if before_cmp["text_mismatch_n"] > 0 or before_cmp["total_ok"][0] is None or \
       any(p["sym_diff"] > 0 for p in before_cmp["pairs"]):
        return "NONDET_BASE", "before 重跑就不一致：生成本身非确定，全部跨臂比较作废"
    if after_cmp["text_mismatch_n"] > 0 or any(p["sym_diff"] > 0 for p in after_cmp["pairs"]):
        return "NONDET_GEN", "before 一致但 after 不一致：编辑状态引入非确定性"
    return "DETERMINISTIC", "before/after 重跑均逐题一致：贪心解码确定，『±1~3 噪声带』假设不成立"


# ---------------- 云端主流程 ----------------
def run_once(model, tok, items, tag, answer_number, build_prompt, gen,
             d1_max_new=96, d2_mode="fewshot", d2_max_new=512):
    """单次全 30 题贪心生成，口径与 cloud_e1_run.evaluate 完全一致。"""
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
        if i % 10 == 0:
            print(f"  [{tag}] {i}/{len(items)}", file=sys.stderr)
    torch.cuda.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="{MODEL_DIR}/Qwen2.5-7B")
    ap.add_argument("--items", default="{DATA_DIR}/d1d2_items.jsonl")
    ap.add_argument("--frac", type=float, default=1.6e-4)
    ap.add_argument("--ulps", type=int, default=3)
    ap.add_argument("--n-before", type=int, default=2)
    ap.add_argument("--n-after", type=int, default=3)
    ap.add_argument("--out", default="{OUT_DIR}/e1_determinism.json")
    a = ap.parse_args()

    sys.path.insert(0, "/root")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from cloud_e1_run import (answer_number, build_prompt, gen,
                              global_sparse_edit, load_items)

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16,
                                                 device_map="cuda")
    model.eval()
    items = load_items(a.items)
    spec = {"d1_max_new": 96, "d2_mode": "fewshot", "d2_max_new": 512}
    out = {"model": a.model, "spec": spec, "frac": a.frac, "ulps": a.ulps,
           "read_only": True, "network_calls": 0,
           "ran_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    before = []
    for k in range(a.n_before):
        rows = run_once(model, tok, items, f"before#{k}", answer_number, build_prompt,
                        gen, **spec)
        before.append(rows)
        print(f"[before#{k}] ok={sum(1 for r in rows if r['ok'])}/30 "
              f"({time.time()-t0:.0f}s)", file=sys.stderr)
    out["before_runs"] = before
    out["before_cmp"] = compare(before)

    n_ed, total = global_sparse_edit(model, tok, items, a.frac, a.ulps, select="grad", seed=0)
    out["edited_positions"], out["candidate_params"] = n_ed, total

    after = []
    for k in range(a.n_after):
        rows = run_once(model, tok, items, f"after#{k}", answer_number, build_prompt,
                        gen, **spec)
        after.append(rows)
        print(f"[after#{k}] ok={sum(1 for r in rows if r['ok'])}/30 "
              f"({time.time()-t0:.0f}s)", file=sys.stderr)
    out["after_runs"] = after
    out["after_cmp"] = compare(after)

    code, why = verdict(out["before_cmp"], out["after_cmp"])
    out["verdict"], out["verdict_reason"] = code, why
    out["elapsed_s"] = round(time.time() - t0, 1)
    json.dump(out, open(a.out, "w"), ensure_ascii=False)
    print(f"[done] {code} :: {why} :: {a.out} ({out['elapsed_s']}s)", file=sys.stderr)


# ---------------- selftest：纯合成，不触盘、不 import 重依赖 ----------------
def _rows(ids, ok_ids, salt):
    return [{"id": i, "family": "D1", "ok": i in ok_ids, "got": 1, "expected": 1.0,
             "hash": hashlib.sha256(f"{salt}:{i}".encode()).hexdigest()[:16]}
            for i in ids]


def _selftest():
    fails, ids = [], ["a", "b", "c", "d"]
    r1 = _rows(ids, {"a", "c"}, "x")

    # 1. 同内容两次 -> 一致
    same = compare([r1, _rows(ids, {"a", "c"}, "x")])
    if same["pairs"][0]["sym_diff"] != 0 or same["text_mismatch_n"] != 0:
        fails.append("同内容两次应判一致，实测不一致")

    # 2. 通过集不同 -> 对称差与题号都要对
    diff = compare([r1, _rows(ids, {"a", "b"}, "x")])
    if diff["pairs"][0]["sym_diff"] != 2:
        fails.append(f"通过集差 {diff['pairs'][0]['sym_diff']} != 2")
    if diff["pairs"][0]["only_base"] != ["c"] or diff["pairs"][0]["only_this"] != ["b"]:
        fails.append("差的题号记错")

    # 3. 红：通过集相同但文本不同（最容易漏判的分支）
    txt = compare([r1, _rows(ids, {"a", "c"}, "y")])
    if txt["text_mismatch_n"] != 4:
        fails.append(f"文本不同但通过集同：mismatch_n={txt['text_mismatch_n']} 应为 4")
    if verdict(same, txt)[0] != "NONDET_GEN":
        fails.append("after 文本不一致未判 NONDET_GEN")

    # 4. 红：before 就不一致 -> NONDET_BASE（不是 NONDET_GEN）
    if verdict(diff, diff)[0] != "NONDET_BASE":
        fails.append("before 不一致未判 NONDET_BASE")

    # 5. 绿：两段都一致
    if verdict(same, same)[0] != "DETERMINISTIC":
        fails.append("全一致未判 DETERMINISTIC")

    # 6. 反证：两个指标必须互相独立 —— 文本相同但通过集不同 -> mismatch=0 而 sym_diff>0
    cross = compare([r1, _rows(ids, {"a", "b"}, "x")])   # 同 salt、不同 ok
    if cross["text_mismatch_n"] != 0:
        fails.append("文本相同却报了 mismatch（指标串了）")
    if cross["pairs"][0]["sym_diff"] == 0:
        fails.append("通过集不同却没报 sym_diff（指标串了）")

    print("SELFTEST", "PASS" if not fails else "FAIL")
    for f in fails:
        print(" -", f)
    return 0 if not fails else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    main()
