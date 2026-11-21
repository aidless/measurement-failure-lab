# evalkit 恒 1.0 移出计分分母重算（第二步）

> 承接 `constant_task_attribution_20261119.md`。那份报告定位了 18 条恒 1.0（adv10+audit8），
> 本轮用脚本真实重算（而非手算 46−18），并验证覆盖率门裁决从红转绿。

## 1. 判据：opt-in 移出，方向可选

`scripts/exclude_constant_recompute.py` 复用 `task_spectra` 的读取口径，找 power≥3 且得分恒定的任务，
按方向（`const1` 天花板饱和 / `const0` 地板塌陷 / `both`）移出后重算 L1/L2/L3。
**不改原结果文件**，只输出新报告，符合 `eval-false-negative-repair` 的 opt-in 原则。

反证（selftest 6 项全过）：const1 抓恒 1.0 不误抓恒 0 / 排除后 const_after=0 / 排除后 L1 正确减少 / L3 保留有区分度的题 / const0 能抓地板塌陷。

## 2. evalkit 重算结果

```
18 条恒 1.0 = adv-01,02,03,04,06,07,08,10,11,12 + audit-01,02,04,05,06,08,09,10

移出前: L1=62  L2=46  L3=28  const=18  inflation=1.348  → F3 红 (18/46=0.391)
移出后: L1=44  L2=28  L3=28  const=0   inflation=1.571  → 门绿（L3/L2=1.0）
```

- **L3 保持 28 不变**：移掉的是无区分度的恒定列，有区分度的题一条没少。这是"移出分母"该有的效果。
- **inflation 从 1.348 升到 1.571**：分母 L1 从 62 降到 44，是因为移走的 18 条本来就在 L1 清单里。这不是坏事——现在 L1/L2 的差距反映的是"真待补测题"，而不是"假饱和题"。

## 3. 顺带发现：evalkit 的 power 分布暴露补测不足

```
power 分布（每道题被几个配置测到）：
  power=2: 16 题   ← 永远进不了 L2（需要 ≥3）
  power=3: 2 题
  power=5: 2 题
  power=7: 24 题
```

evalkit 的 12 个"配置"其实是不同的 `run_id`，混了两个模型（deepseek-v4.1-flash / Qwen3-4B-AWQ）、
不同 `max_tokens`（512/1024/2048）、不同 `max_turns`（1/6）。每个 run 题数不齐（36/10/20/6/24），
导致 **16 道题只被 2 个配置测到，power=2<3，这些题即使有区分度也进不了 L2**。

这是与 databench/wse 不同的第四种现象：**不是恒定列，而是欠采样列**（power 不足）。
移出恒 1.0 解决的是"分母混入无区分度题"，但 16 条 power=2 的题仍需补 1 个配置才能进入统计。

## 4. 复算命令

```bash
python3 scripts/exclude_constant_recompute.py --selftest   # 6 项含反证，全过
python3 scripts/exclude_constant_recompute.py \
  --results agent-evalkit/results --direction const1 \
  --label evalkit_drop18 --out reports/evalkit_drop18_recompute_20261119.json
```

产物 `reports/evalkit_drop18_recompute_20261119.json` 含：excluded_ids（18 条）、before/after 全套口径、power 分布。
