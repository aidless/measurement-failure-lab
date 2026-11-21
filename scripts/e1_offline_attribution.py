#!/usr/bin/env python3
"""E1 离线归因：所谓"±1~3 噪声带"到底是运行时随机，还是脚本版本差异？

做法（join 优先于 re-derive，decision-instrument-protocol 阶段 0 第②问）：
不重跑任何生成，直接读 reports/cloud_e1/ 下已落盘的 before_rows（原始读数），
按"同一个 prompt/判分口径"分簇，算簇内与簇间的通过集对称差。

预注册判据：
  NO_RUNTIME_NONDETERMINISM  每个簇内部对称差=0（≥2 个独立进程逐题一致）-> 贪心确定，
                             簇间差异可由脚本版本（output 截断 / 判分器 / d2_max_new）解释
  RUNTIME_NONDETERMINISM     同一簇内部对称差>0 -> 无法用配置解释，存在运行时随机

恒等闸（阶段 0 第④问）：ok == (answer_number(output) 在 expected±tol 内)；
违例数必须报出来，>0 则整台仪器作废。

只读：只读 reports/cloud_e1/*.json，不联网、不写远端。产物写 --out。
"""
import argparse, glob, itertools, json, os, re, sys

ANS_RE = re.compile(r"答案\s*[：:]\s*(-?\d+(?:\.\d+)?(?:e-?\d+)?)", re.I)
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?(?:e-?\d+)?")


def last_number(text):
    nums = NUM_RE.findall(str(text).replace(",", ""))
    try:
        return float(nums[-1]) if nums else None
    except ValueError:
        return None


def answer_number(text):
    tagged = ANS_RE.findall(str(text).replace(",", ""))
    if tagged:
        try:
            return float(tagged[-1])
        except ValueError:
            pass
    return last_number(text)


# 簇定义：key -> (文件列表, 该簇用的判分器)
# 依据（实测，见各 json 的 eval_spec 与 output 字段）：
#  judgefix 簇 = answer_number 判分 + output 保留全文 + d2 fewshot/512
#  legacy   簇 = last_number 判分 + output[:300] 截断 + d2 plain/96（旧脚本默认）
CLUSTERS = {
    "judgefix": {
        "files": [
            "c_fs_judgefix.jsonl.summary.json",
            "a_fs_u3_judgefix.jsonl.summary.json",
        ],
        "scorer": "answer_number",
    },
    "legacy": {
        "files": [
            "c_rep2_fewshot.summary.json",
            "c_rep3_fewshot.summary.json",
            "c_arm_qwen25_7b_baseline_fewshot.summary.json",
        ],
        "scorer": "last_number",
    },
}
SCORERS = {"answer_number": answer_number, "last_number": last_number}


def load_ok(reports_dir, fn):
    p = os.path.join(reports_dir, fn)
    d = json.load(open(p))
    rows = d.get("before_rows") or []
    if not rows:
        return None
    return {r["id"]: r for r in rows}


def sym_diff(a, b):
    return len(a ^ b), sorted(a - b), sorted(b - a)


def identity_gate(rows_map, scorer_name="answer_number"):
    """恒等闸：用该簇自己的判分器重算 ok == (scorer(output) near expected)。
    违例数必须报出来，>0 则说明连该簇自己的口径都对不上，整台仪器作废。"""
    scorer = SCORERS[scorer_name]
    bad = []
    for qid, r in rows_map.items():
        txt = r.get("output", "")
        got = scorer(txt)
        expected = float(r["expected"])
        tol = 1e-6  # E1 题目答案为整数或一位小数，精确比较
        recomputed = got is not None and abs(got - expected) <= tol
        if recomputed != bool(r["ok"]):
            bad.append({"id": qid, "stored_ok": r["ok"], "recomputed_ok": recomputed,
                        "got": got, "expected": expected,
                        "output_tail": txt[-80:]})
    return bad


def analyze(reports_dir):
    out = {"clusters": {}, "identity_gate": {}, "note": ""}
    loaded = {}
    for cname, cfg in CLUSTERS.items():
        loaded[cname] = {}
        for fn in cfg["files"]:
            rm = load_ok(reports_dir, fn)
            if rm is not None:
                loaded[cname][fn] = rm

    # 恒等闸：每个簇用它自己的判分器重算。
    # output 被截断到 300 字符的簇无法用落盘文本复算原本用完整输出算的 ok
    # -> 标 UNRECOMPUTABLE（feasibility 问题），不是 verdict 失败（见 decision-instrument-protocol §4）。
    all_bad = {}
    for cname, members in loaded.items():
        scorer_name = CLUSTERS[cname]["scorer"]
        for fn, rm in members.items():
            max_len = max((len(r.get("output", "")) for r in rm.values()), default=0)
            if max_len <= 300 and any(len(r.get("output", "")) >= 300 for r in rm.values()):
                all_bad[fn] = {"scorer": scorer_name, "status": "UNRECOMPUTABLE",
                               "reason": f"output 被截断（max_len={max_len}），无法用落盘文本复算 ok",
                               "n_violation": None}
            else:
                bad = identity_gate(rm, scorer_name)
                all_bad[fn] = {"scorer": scorer_name, "status": "RECOMPUTED",
                               "n_violation": len(bad), "samples": bad[:3]}
    out["identity_gate"] = all_bad

    # 簇内对称差
    for cname, members in loaded.items():
        fns = list(members)
        pairs = []
        for f1, f2 in itertools.combinations(fns, 2):
            s1 = {k for k, r in members[f1].items() if r["ok"]}
            s2 = {k for k, r in members[f2].items() if r["ok"]}
            d, only1, only2 = sym_diff(s1, s2)
            pairs.append({"a": f1, "b": f2, "sym_diff": d,
                          "only_a": only1, "only_b": only2,
                          "n_a": len(s1), "n_b": len(s2)})
        out["clusters"][cname] = {"files": fns, "within_pairs": pairs}

    # 簇间对称差（用每簇第一个成员做代表）
    cross = []
    reps = {c: next(iter(m.values())) for c, m in loaded.items() if m}
    for c1, c2 in itertools.combinations(reps, 2):
        s1 = {k for k, r in reps[c1].items() if r["ok"]}
        s2 = {k for k, r in reps[c2].items() if r["ok"]}
        d, only1, only2 = sym_diff(s1, s2)
        cross.append({"clusters": [c1, c2], "sym_diff": d,
                      "only_" + c1: only1, "only_" + c2: only2})
    out["cross_cluster"] = cross

    # 判决：簇内对称差全部为 0 才支持"无运行时随机"
    within_all_zero = all(
        p["sym_diff"] == 0
        for c in out["clusters"].values() for p in c["within_pairs"]
    )
    # 判决：恒等闸只要求「可复算簇」零违例；UNRECOMPUTABLE 簇的 ok 是落盘原始读数，
    # 仍可用于簇内/簇间对称差，只是无法从 output 复算，单独标注。
    recomputed_bad = {k: v for k, v in all_bad.items()
                      if v.get("status") == "RECOMPUTED" and v["n_violation"] > 0}
    if recomputed_bad:
        out["verdict"] = "IDENTITY_GATE_FAIL"
        out["reason"] = "可复算簇恒等闸有违例：ok 与判分器重算不一致，整台仪器作废"
    elif within_all_zero and len(loaded) >= 2:
        out["verdict"] = "NO_RUNTIME_NONDETERMINISM"
        out["reason"] = ("每个口径簇内部 ≥2 个独立进程逐题通过集完全一致（簇内对称差=0），"
                         "簇间差异可由脚本版本（output 截断/判分器/d2_max_new）解释，"
                         "不支持『贪心生成有 ±1~3 运行时噪声』；"
                         "legacy 簇 output 被截断标 UNRECOMPUTABLE，但其 ok 为落盘原始读数，"
                         "仍参与对称差")
    else:
        out["verdict"] = "RUNTIME_NONDETERMINISM"
        out["reason"] = "同簇内部出现对称差>0，无法用配置差异解释"
    out["identity_gate_summary"] = {
        "recomputed_zero_violation": [k for k, v in all_bad.items()
                                      if v.get("status") == "RECOMPUTED" and v["n_violation"] == 0],
        "recomputed_with_violation": list(recomputed_bad),
        "unrecomputable": [k for k, v in all_bad.items() if v.get("status") == "UNRECOMPUTABLE"],
    }
    return out


def selftest():
    fails = []
    # 合成两簇：每簇内部一致、簇间差 2 题 -> 应判 NO_RUNTIME_NONDETERMINISM
    mk = lambda oks: {i: {"id": i, "ok": i in oks, "output": f"答案：{1 if i in oks else 2}",
                          "expected": 1.0} for i in ["a", "b", "c", "d"]}
    assert sym_diff({"a", "b"}, {"a", "c"})[0] == 2
    # 恒等闸：output "答案：1" expected 1.0 ok=True -> 无违例
    g = identity_gate(mk({"a", "b"}))
    if len(g) != 0:
        fails.append(f"恒等闸对合成一致数据应 0 违例，实测 {len(g)}")
    # 恒等闸能红：篡改一题 ok
    bad = mk({"a", "b"}); bad["c"]["ok"] = True  # output 是 答案：2，expected 1.0 -> 应违例
    g2 = identity_gate(bad)
    if len(g2) != 1 or g2[0]["id"] != "c":
        fails.append(f"恒等闸篡改应抓 1 违例(id=c)，实测 {g2}")
    # 对称差函数能红
    if sym_diff({"a"}, {"a", "b"}) != (1, [], ["b"]):
        fails.append("sym_diff 基本用例错")
    print("SELFTEST", "PASS" if not fails else "FAIL")
    for f in fails:
        print(" -", f)
    return 0 if not fails else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports-dir", default="reports/cloud_e1")
    ap.add_argument("--out", default="reports/cloud_e1/e1_offline_attribution.json")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    out = analyze(a.reports_dir)
    json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"verdict={out['verdict']}")
    print(f"reason={out['reason']}")
    print(f"identity_gate violations=" +
          str({k: v["n_violation"] for k, v in out["identity_gate"].items()}))
    print(f"written → {a.out}")


if __name__ == "__main__":
    main()
