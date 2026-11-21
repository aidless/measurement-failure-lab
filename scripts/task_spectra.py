#!/usr/bin/env python3
"""task_spectra.py — 评测集覆盖率与区分度审计（L1/L2/L3 + power + k_eff + 补测清单）。

设计依据：eval-suite-coverage-audit skill（power>=3 才有残差自由度；文件数≠统计功效；
常数列不抬高 σ1；恒定任务先于一切统计被发现；hash 锁失配只报告不重写）。

用法:
  python3 scripts/task_spectra.py scores   <results-dir> [--json]
  python3 scripts/task_spectra.py coverage <results-dir> --manifest <bench_manifest.json> [--json] [--out FILE]
  python3 scripts/task_spectra.py validate <manifest.json> [--targets T001,T002] [--json]
  python3 scripts/task_spectra.py selftest

结果文件格式（自动识别）:
  A) {"scores": {"T001": 1.0, ...}, "overall":..., "verified":..., "provenance": {...}}
  B) {"T001": 1.0, "T002": 0.0, ...}            # 顶层 task->score 映射
  C) [{"id":"T001","score":1.0}, ...]            # list of dicts
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import math
import os
import re
import sys
from datetime import datetime, timezone

try:
    import numpy as np
except ImportError:
    np = None

EXCLUDE_DIRS = {"__pycache__", ".git", "node_modules"}
POWER_MIN = 3


# ---------------------------------------------------------------- io helpers
def _iter_json_files(root):
    if os.path.isfile(root):
        yield root
        return
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for fn in sorted(filenames):
            if fn.endswith(".json") or fn.endswith(".jsonl"):
                yield os.path.join(dirpath, fn)


def _canon(obj):
    """项目 hash 锁的同款 canonicalization（见 wse-bench-public/build_bench.py:20）。"""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True).encode("utf-8")


def _load_manifest(path):
    with open(path, encoding="utf-8") as f:
        m = json.load(f)
    tasks = m.get("tasks", m if isinstance(m, list) else [])
    if isinstance(tasks, dict):
        tasks = [dict(v, id=k) for k, v in tasks.items() if isinstance(v, dict)]
    return m, tasks


def _verify_hashes(tasks):
    """返回 (ok_count, mismatches)。绝不修改任何 hash 字段。"""
    ok = 0
    mismatches = []
    for t in tasks:
        stored = t.get("hash")
        if not stored:
            continue
        computed = hashlib.sha256(_canon({k: v for k, v in t.items() if k != "hash"})).hexdigest()
        if computed == stored:
            ok += 1
        else:
            mismatches.append({
                "id": t.get("id"), "stored": stored[:16], "computed": computed[:16],
                "cohort": t.get("cohort"),
            })
    return ok, mismatches


def _read_records(path):
    """读 .json（单对象）或 .jsonl（逐行对象，如 agent-databench results）。"""
    if path.endswith(".jsonl"):
        recs = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
        return recs
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _extract_configs(obj, path):
    """把单个文件内容解析成配置列表 [(scores_dict, meta)]；无法识别返回 []。"""
    out = []

    def one(d, meta):
        if d and len(d) >= 2:
            out.append((d, meta))

    if isinstance(obj, list) and obj and all(isinstance(x, dict) for x in obj):
        # .jsonl 逐行记录：按 (llm, config) 分组，每组独立成一个配置
        groups = {}
        for x in obj:
            tid = x.get("task_id") or x.get("id") or x.get("task")
            if not tid:
                continue
            key = (str(x.get("llm", "")), str(x.get("config", "")))
            groups.setdefault(key, {})
            if "score" in x and isinstance(x["score"], (int, float)):
                groups[key][str(tid)] = float(x["score"])
            elif "passed" in x:
                groups[key][str(tid)] = 1.0 if x["passed"] else 0.0
            elif "ok" in x:
                groups[key][str(tid)] = 1.0 if x["ok"] else 0.0
        for (llm, cfg), d in groups.items():
            one(d, {"llm": llm, "config": cfg})
        return out

    if isinstance(obj, dict):
        if isinstance(obj.get("scores"), dict):
            s = {k: float(v) for k, v in obj["scores"].items() if isinstance(v, (int, float))}
            one(s, {k: v for k, v in obj.items() if k not in ("scores", "answers", "answer")})
            return out
        if isinstance(obj.get("results"), list):  # agent-evalkit
            d = {}
            for x in obj["results"]:
                if not isinstance(x, dict):
                    continue
                tid = x.get("task_id") or x.get("id") or x.get("task")
                if not tid:
                    continue
                if isinstance(x.get("score"), (int, float)):
                    d[str(tid)] = float(x["score"])
                elif "ok" in x:
                    d[str(tid)] = 1.0 if x["ok"] else 0.0
            one(d, {k: v for k, v in obj.items() if k != "results"})
            return out
        # 顶层 T### -> score 映射；要求全部数字键都是 T###，排除 total_pass/headroom 元数据
        numeric = {k: float(v) for k, v in obj.items()
                   if isinstance(v, (int, float)) and not isinstance(v, bool)}
        if len(numeric) >= 2 and all(re.fullmatch(r"[Tt]\d{3,}", k) for k in numeric):
            one(numeric, {})
            return out
        return out
    return out


def _identity_fp(meta):
    """配置身份指纹：llm/config/model/name 中实际出现的字段。
    用于去重——得分向量相同但身份不同的配置（如 1.5b 与 7b 都全 0）
    必须算作不同配置，不能折叠；身份全空（{"scores":...} 无 meta）时为空串。"""
    keys = ("llm", "config", "model", "name")
    parts = []
    for k in keys:
        v = meta.get(k)
        if v not in (None, ""):
            parts.append(f"{k}={v}")
    return "|".join(sorted(parts))


def load_configs(results_dir):
    """返回 configs: list[{file, vector, scores, meta}]。
    去重键 = (配置身份指纹, 得分向量)：身份相同且向量相同才折叠（同一份结果被复制到
    两个路径）；身份不同即使向量相同也保留为不同配置（避免把得分碰巧相同的不同模型合并）。"""
    seen = {}
    skipped = []
    for path in _iter_json_files(results_dir):
        try:
            obj = _read_records(path)
        except Exception:
            skipped.append({"file": path, "reason": "json_parse_error"})
            continue
        got = _extract_configs(obj, path)
        if not got:
            skipped.append({"file": path, "reason": "not_score_file"})
            continue
        for scores, meta in got:
            vec = tuple(sorted((k, round(v, 6)) for k, v in scores.items()))
            key = (_identity_fp(meta), vec)
            if key in seen:
                seen[key]["duplicate_files"].append(os.path.basename(path))
                continue
            seen[key] = {
                "file": os.path.basename(path),
                "path": path,
                "vector": vec,
                "scores": scores,
                "meta": meta,
                "duplicate_files": [],
            }
    return list(seen.values()), skipped


# ---------------------------------------------------------------- core stats
def compute_spectra(configs, tasks=None):
    """k_eff（participation ratio）与谱；列按列中心化，缺失填 0。"""
    if np is None:
        return {"available": False, "reason": "numpy unavailable"}
    if not configs or not tasks:
        return {"available": False, "reason": "no configs or no tasks"}
    n_cfg, n_task = len(configs), len(tasks)
    tidx = {t: i for i, t in enumerate(tasks)}
    M = np.zeros((n_cfg, n_task), dtype=float)
    for ci, c in enumerate(configs):
        for tid, sc in c["scores"].items():
            if tid in tidx:
                M[ci, tidx[tid]] = sc
    X = M.copy()
    present = (M != 0).astype(float)  # 用原始是否有键更准，但 0 分合法；下方重算
    for ci, c in enumerate(configs):
        for tid in c["scores"]:
            if tid in tidx:
                present[ci, tidx[tid]] = 1.0
    # 只在存在的格子上中心化：缺失保持 0，不参与均值
    for ti in range(n_task):
        mask = present[:, ti] > 0
        if mask.any():
            mean = M[mask, ti].mean()
            X[mask, ti] = M[mask, ti] - mean
        X[~mask, ti] = 0.0
    s = np.linalg.svd(X, compute_uv=False)
    denom = float(s @ s)
    pr = float((s.sum() ** 2) / denom) if denom > 0 else 0.0
    return {
        "available": True,
        "n_tasks": n_task,
        "n_configs": n_cfg,
        "k_eff_participation_ratio": round(pr, 4),
        "upper_bound_min_tasks_configs": min(n_task, n_cfg),
        "singular_values_top10": [round(float(x), 4) for x in s[:10]],
        "note": "k_eff 同时是任务冗余与配置多样性的函数；必须与上界 min(n_tasks, n_configs) 同报，跨配置数不可比。",
    }


def audit(results_dir, manifest_path=None):
    configs, skipped = load_configs(results_dir)
    task_cfg = collections.defaultdict(set)
    for ci, c in enumerate(configs):
        for tid in c["scores"]:
            task_cfg[tid].add(ci)

    if manifest_path:
        manifest, mtasks = _load_manifest(manifest_path)
        L1_ids = [t["id"] if isinstance(t, dict) else str(t) for t in mtasks]
        hash_ok, hash_mismatch = _verify_hashes([t for t in mtasks if isinstance(t, dict)])
        hash_info = {
            "verified": hash_ok, "mismatched": len(hash_mismatch),
            "mismatches": hash_mismatch,
            "policy": "失配只报告，绝不重写 hash；目标集失配须中止补测。",
        }
    else:
        L1_ids = sorted(task_cfg.keys())
        hash_info = {"verified": 0, "mismatched": 0, "mismatches": [], "policy": "no manifest"}
    L1 = len(L1_ids)

    measured = [t for t in L1_ids if t in task_cfg]
    L2_ids = [t for t in L1_ids if len(task_cfg.get(t, set())) >= POWER_MIN]

    # 恒定任务：在所有它出现的配置里得分完全相同
    constant = []
    for t in L2_ids:
        vals = {configs[ci]["scores"][t] for ci in task_cfg[t] if t in configs[ci]["scores"]}
        if len(vals) == 1:
            constant.append({"id": t, "score": next(iter(vals)), "n_configs": len(task_cfg[t])})
    L3_ids = [t for t in L2_ids if t not in {c["id"] for c in constant}]

    spectra = compute_spectra(configs, L2_ids)

    backfill = []
    for t in L1_ids:
        p = len(task_cfg.get(t, set()))
        if p < POWER_MIN:
            backfill.append({"id": t, "power": p, "reason": "never_measured" if p == 0 else "underpowered"})

    return {
        "generated": datetime.now(timezone.utc).isoformat(),
        "inputs": {"results_dir": os.path.abspath(results_dir), "manifest": manifest_path},
        "files": {
            "json_scanned": sum(1 for _ in _iter_json_files(results_dir)),
            "score_files_used": len(configs),
            "duplicate_files_collapsed": sum(len(c["duplicate_files"]) for c in configs),
            "skipped": skipped[:20],
        },
        "caliber": {
            "L1_manifest": L1,
            "L2_measured_power_ge_3": len(L2_ids),
            "L3_discriminative": len(L3_ids),
            "inflation_L1_over_L2": round(L1 / len(L2_ids), 3) if L2_ids else None,
            "measured_at_least_once": len(measured),
            "constant_tasks_in_L2": len(constant),
        },
        "hash_lock": hash_info,
        "constant_tasks": constant,
        "spectra": spectra,
        "backfill_targets": backfill,
        "verdict": _verdict(L1, len(L2_ids), len(L3_ids), len(backfill)),
    }


def _verdict(l1, l2, l3, n_backfill):
    if l2 == 0:
        return "NO_MEASUREMENT: 没有任何任务达到 power>=3，任何任务间关系都不可估。"
    if l3 == 0:
        return "NO_DISCRIMINATION: 所有可测任务在所有配置下恒分，评测器是常数仪器。"
    return (f"清单 {l1} 条；其中 {l2} 条有 >=3 种不同配置的测量 ({l2}/{l1})；"
            f"再其中 {l3} 条仍有区分度。'N 条' 描述清单，'{l2} 条' 才是实测规模，"
            f"高估 {round(l1 / l2, 2)} 倍。{n_backfill} 条待补测。")


# ---------------------------------------------------------------- selftest
def selftest():
    """非空性反证：注入违规必须 exit != 0。"""
    import tempfile
    fails = []

    def check(name, cond, detail=""):
        print(("PASS" if cond else "FAIL"), name, detail)
        if not cond:
            fails.append(name)

    with tempfile.TemporaryDirectory() as td:
        rd = os.path.join(td, "res")
        os.makedirs(rd)
        # 3 个真正不同的配置：T1 恒 1.0；T2 有差异；T3 只在 2 个配置出现；T4 从不出现
        configs = [
            {"T001": 1.0, "T002": 0.0, "T003": 1.0},
            {"T001": 1.0, "T002": 1.0, "T003": 0.0},
            {"T001": 1.0, "T002": 0.5},
        ]
        for i, c in enumerate(configs):
            with open(os.path.join(rd, f"c{i}.json"), "w") as f:
                json.dump({"scores": c}, f)
        # 重复文件：同向量不得增加配置数
        with open(os.path.join(rd, "c0_copy.json"), "w") as f:
            json.dump({"scores": configs[0]}, f)
        mf = os.path.join(td, "manifest.json")
        tasks = [{"id": t, "prompt": t,
                  "hash": hashlib.sha256(_canon({"id": t, "prompt": t})).hexdigest()}
                 for t in ["T001", "T002", "T003", "T004"]]
        with open(mf, "w") as f:
            json.dump({"tasks": tasks}, f)

        r = audit(rd, mf)
        cal = r["caliber"]
        check("distinct configs dedup", r["files"]["score_files_used"] == 3,
              f"got {r['files']['score_files_used']}")
        check("duplicate collapsed", r["files"]["duplicate_files_collapsed"] == 1)
        check("L1=4", cal["L1_manifest"] == 4)
        check("L2=2 (T001,T002)", cal["L2_measured_power_ge_3"] == 2, f"got {cal['L2_measured_power_ge_3']}")
        check("constant T001 detected", any(x["id"] == "T001" for x in r["constant_tasks"]))
        check("L3=1 (T002)", cal["L3_discriminative"] == 1, f"got {cal['L3_discriminative']}")
        check("backfill includes T003,T004", {x["id"] for x in r["backfill_targets"]} == {"T003", "T004"})

        # 反证 1：篡改 manifest 正文 -> hash 失配必须被报出
        tasks[1]["prompt"] = "TAMPERED"
        with open(mf, "w") as f:
            json.dump({"tasks": tasks}, f)
        r2 = audit(rd, mf)
        check("hash tamper detected (red)", r2["hash_lock"]["mismatched"] == 1,
              f"got {r2['hash_lock']['mismatched']}")

        # 反证 2：常数列不抬高 σ1 —— 加一个所有配置都相同的任务 T009，σ1 不变
        s1 = compute_spectra(load_configs(rd)[0], ["T001", "T002"])["singular_values_top10"]
        rd2 = os.path.join(td, "res2")
        os.makedirs(rd2)
        for i, c in enumerate(configs):
            with open(os.path.join(rd2, f"c{i}.json"), "w") as f:
                json.dump({"scores": dict(c, T009=1.0)}, f)
        s2 = compute_spectra(load_configs(rd2)[0], ["T001", "T002", "T009"])["singular_values_top10"]
        check("constant col does not raise sigma1 (red)", abs(s1[0] - s2[0]) < 1e-9,
              f"sigma1 {s1[0]:.6f} -> {s2[0]:.6f}")

        # 反证 3：身份不同、向量相同的两个配置不得折叠（1.5b 与 7b 都全 0 的真实场景）
        rd3 = os.path.join(td, "res3")
        os.makedirs(rd3)
        def _dump_jsonl(path, cfg):
            with open(path, "w") as f:
                for tid, p in (("T001", False), ("T002", False)):
                    f.write(json.dumps({"task_id": tid, "llm": "openai-compat",
                                        "config": cfg, "passed": p}) + "\n")
        _dump_jsonl(os.path.join(rd3, "a.jsonl"), "cfgA")
        _dump_jsonl(os.path.join(rd3, "b.jsonl"), "cfgB")
        # 同一身份+同向量再来一份，应折叠为 cfgA 的重复
        _dump_jsonl(os.path.join(rd3, "a_copy.jsonl"), "cfgA")
        r3 = load_configs(rd3)[0]
        check("identity-different same-vector configs NOT folded (red)",
              len(r3) == 2 and sum(len(c["duplicate_files"]) for c in r3) == 1,
              f"got {len(r3)} configs, dups={sum(len(c['duplicate_files']) for c in r3)}")

    print(f"\n{'ALL PASS' if not fails else 'FAILED: ' + ', '.join(fails)}")
    return 0 if not fails else 1


# ---------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("scores", "coverage"):
        p = sub.add_parser(name)
        p.add_argument("results_dir")
        p.add_argument("--manifest")
        p.add_argument("--json", action="store_true")
        p.add_argument("--out")
    pv = sub.add_parser("validate")
    pv.add_argument("manifest")
    pv.add_argument("--targets", help="comma separated task ids; hash mismatch in target => exit 2")
    pv.add_argument("--json", action="store_true")
    sub.add_parser("selftest")
    args = ap.parse_args()

    if args.cmd == "selftest":
        return selftest()

    if args.cmd == "validate":
        m, tasks = _load_manifest(args.manifest)
        ok, bad = _verify_hashes([t for t in tasks if isinstance(t, dict)])
        targets = set(args.targets.split(",")) if args.targets else set()
        target_bad = [x for x in bad if x["id"] in targets]
        out = {"verified": ok, "mismatched": bad, "target_mismatch": target_bad}
        print(json.dumps(out, ensure_ascii=False, indent=2) if args.json else
              f"hash ok {ok}/{ok+len(bad)}; mismatched {len(bad)}; target mismatch {len(target_bad)}")
        if target_bad:
            print("TARGET HASH MISMATCH — 补测中止（绝不重写 hash 字段）", file=sys.stderr)
            return 2
        return 0

    manifest = args.manifest
    if args.cmd == "scores" and not manifest:
        # scores 子命令默认不找 manifest
        pass
    r = audit(args.results_dir, manifest)
    if args.cmd == "scores" and manifest is None:
        # 纯 scores：去掉 manifest 专属字段
        r["caliber"]["L1_manifest"] = r["caliber"]["measured_at_least_once"]
        r["caliber"]["inflation_L1_over_L2"] = (
            round(r["caliber"]["L1_manifest"] / r["caliber"]["L2_measured_power_ge_3"], 3)
            if r["caliber"]["L2_measured_power_ge_3"] else None)
    text = json.dumps(r, ensure_ascii=False, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"written: {args.out}")
    if args.json or not args.out:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
