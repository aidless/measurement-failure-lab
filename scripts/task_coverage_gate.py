#!/usr/bin/env python3
"""task_coverage_gate.py — 评测覆盖率门（fail-closed，可失败）

把 task_spectra.py coverage 的读数变成有退出码的闸。判据（任一 FAIL 即 exit 1）：
  F1 NO_MEASUREMENT      L2 == 0                                   → 完全没测到
  F2 区分度塌缩           L3/L2 < --min-l3-ratio (默认 0.60)         → 多数任务无区分度
  F3 恒定任务污染         constant_tasks / L2 > --max-const-ratio (默认 0.20)
  W1 口径高估             inflation_L1_over_L2 > --max-inflation (默认 2.0) → 警告不阻断
  W2 hash 失配            hash_lock.mismatched > 0                  → 按策略只报告

用法：
  python3 scripts/task_coverage_gate.py reports/task_spectra_wse_20261119.json
  python3 scripts/task_coverage_gate.py --dir reports --pattern 'task_spectra_*_2026*.json'
  python3 scripts/task_coverage_gate.py --selftest          # 4 反证，必须真红
  python3 scripts/task_coverage_gate.py --prove-red         # 对真实三仓跑，打印每份裁决
"""
from __future__ import annotations
import argparse, glob, json, os, sys


def judge(cov: dict, min_l3: float, max_const: float, max_infl: float):
    """返回 (fails, warns)。fails 非空 => 红。"""
    cal = cov.get('caliber') or {}
    L2 = int(cal.get('L2_measured_power_ge_3') or 0)
    L3 = int(cal.get('L3_discriminative') or 0)
    const = len(cov.get('constant_tasks') or [])
    infl = cal.get('inflation_L1_over_L2')
    hash_mm = int((cov.get('hash_lock') or {}).get('mismatched') or 0)
    fails, warns = [], []
    if L2 == 0:
        fails.append(f'F1 NO_MEASUREMENT: L2=0（{cov.get("inputs",{}).get("results_dir","?")} 没有任何任务达到 power>=3）')
    else:
        ratio = L3 / L2
        if ratio < min_l3:
            fails.append(f'F2 区分度塌缩: L3/L2={ratio:.3f} < {min_l3}（{L2} 条有效测量里只有 {L3} 条仍有区分度）')
        cr = const / L2
        if cr > max_const:
            fails.append(f'F3 恒定任务污染: {const}/{L2}={cr:.3f} > {max_const}（这些任务在任何配置下同分，却在计分分母里）')
    if infl is not None and infl > max_infl:
        warns.append(f'W1 口径高估: L1/L2={infl:.3f} > {max_infl}')
    if hash_mm:
        warns.append(f'W2 hash 失配: {hash_mm} 条（只报告，不重写）')
    return fails, warns


def fmt(cov, fails, warns):
    v = cov.get('verdict', '')
    L1 = (cov.get('caliber') or {}).get('L1_manifest')
    return (f'  {v}\n'
            f'  fails={fails}\n  warns={warns}\n')


def selftest() -> int:
    """4 个 fixture：1 合规 + 3 违规；违规必须 exit≠0（反证）。"""
    base = {'inputs': {'results_dir': 'fixture'}, 'caliber': {'L1_manifest': 100, 'L2_measured_power_ge_3': 80,
            'L3_discriminative': 60, 'inflation_L1_over_L2': 1.25},
            'constant_tasks': [{'id': f'C{i}', 'score': 1.0} for i in range(8)],
            'hash_lock': {'mismatched': 0}, 'verdict': 'ok'}
    ok = judge(base, 0.60, 0.20, 2.0)
    fixtures = {
        'PASS(合规)': (base, []),
        'FAIL(NO_MEASUREMENT)': ({**base, 'caliber': {**base['caliber'], 'L2_measured_power_ge_3': 0,
                                'L3_discriminative': 0, 'inflation_L1_over_L2': None}}, ['F1']),
        'FAIL(区分度塌缩)': ({**base, 'caliber': {**base['caliber'], 'L3_discriminative': 20}}, ['F2']),
        'FAIL(恒定污染)': ({**base, 'constant_tasks': [{'id': f'C{i}'} for i in range(30)]}, ['F3']),
    }
    bad = 0
    assert not (ok[0] or ok[1]), f'PASS fixture 不该有 fails/warns: {ok}'
    print('[selftest] PASS fixture: 绿 ✔')
    for name, (cov, mustcodes) in fixtures.items():
        f, w = judge(cov, 0.60, 0.20, 2.0)
        if name.startswith('PASS'):
            if f:
                print(f'[selftest] ✘ {name} 误报: {f}'); bad += 1
            else:
                print(f'[selftest] ✔ {name} 绿')
            continue
        hit = [c for c in mustcodes if any(x.startswith(c) for x in f)]
        if not hit:
            print(f'[selftest] ✘ {name} 未变红（注入违规必须 exit≠0）: {f}'); bad += 1
        else:
            print(f'[selftest] ✔ {name} 红 ({hit})')
    print(f'\n[selftest] {"全部通过" if bad==0 else f"{bad} 项失败"}')
    return 0 if bad == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('coverage', nargs='?', help='coverage JSON 路径')
    ap.add_argument('--dir', help='目录（配合 --pattern 批量）')
    ap.add_argument('--pattern', default='task_spectra_*_2026*.json')
    ap.add_argument('--min-l3-ratio', type=float, default=0.60)
    ap.add_argument('--max-const-ratio', type=float, default=0.20)
    ap.add_argument('--max-inflation', type=float, default=2.0)
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--prove-red', action='store_true', help='对 reports 下真实 coverage 跑并打印裁决')
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    files = []
    if a.prove_red:
        files = sorted(glob.glob(os.path.join('reports', 'task_spectra_*_2026*.json')))
        if not files:
            print('[prove-red] reports 下没有 coverage JSON', file=sys.stderr); return 2
    elif a.dir:
        files = sorted(glob.glob(os.path.join(a.dir, a.pattern)))
    elif a.coverage:
        files = [a.coverage]
    else:
        ap.error('需要 coverage 路径，或 --dir，或 --selftest / --prove-red')
    worst = 0
    for p in files:
        try:
            cov = json.load(open(p))
        except Exception as e:
            print(f'{p}: 读取失败 {e}', file=sys.stderr); worst = max(worst, 2); continue
        f, w = judge(cov, a.min_l3_ratio, a.max_const_ratio, a.max_inflation)
        print(f'\n{p}\n' + fmt(cov, f, w))
        worst = max(worst, 1 if f else 0)
    print(f'\n[gate] {len(files)} 份，最坏退出码={worst}')
    return worst


if __name__ == '__main__':
    sys.exit(main())
