# 三个评测仓库的覆盖率审计（2026-11-19）

## 结论
同一把尺子量了三份评测资产，问题形状不同：

| 仓库 | L1 清单 | L2 实测(power≥3) | L3 有区分度 | 高估倍数 | 主要病症 |
|---|---|---|---|---|---|
| wse-bench | 156 | 102 | 89 | **1.53×** | 13 条恒 1.0；54 条从未测 |
| agent-evalkit | 62 | 46 | 28 | **1.35×** | **18/46 条恒 1.0（39%）**，hard/audit 两类全饱和 |
| agent-databench | 20 | 0 | 0 | — | 只有 1 个配置，**NO_MEASUREMENT** |

## 逐仓要点

### wse-bench（已有完整报告）
`reports/task_spectra_wse_20261119.json`：47 配置、k_eff=10.65/上界 47、hash 153/156。恒定任务集中在 reasoning(9)、reading_comprehension(3)、structured_output(1)——即"难能力"反而先饱和，提示 2026Q3 这批判据对该模型已无区分力。

### agent-evalkit（本次新增覆盖）
- 62 条清单（tasks 36 + audit 20 + mined 7，去重后 62 个 id）
- 47 个不同配置（run-local1..6、run-gw1..6、audit ×3 等），去重后 11 个真正不同的得分向量；k_eff=3.59，上界 11
- **18 条在全部配置下恒 1.0**：hard 10 条 + audit 8 条。hard 类全是"该模型稳定做对"的边界算术/工具题；audit 类是"稳定判对"的审计题。两者都应从计分分母移出或升难度。
- 16 条 power<3 待补测（calc/multi/mined 类中的单配置题）

判据：`agent-evalkit/results/*.json` 的 `results[].ok` 字段就是二值判分；同 (model, max_turns) 的重复 run 因得分向量相同被折叠（这是对的——重跑不产生新配置）。

### agent-databench（本次新增覆盖）
- 20 条任务，结果只有 `results/base-fake.jsonl` 一个 (llm=fake, config=base) 组
- **power 全为 1**：任意两任务得分轮廓都只能被一条直线拟合，冗余/谱分析不可用
- 结论是 NO_MEASUREMENT——不是题目难，是测量只有一个点。补测优先级最高（成本也最低：tasks.jsonl 已有 gold，run.py 现成）。

## 可执行的下一步（按成本排序）

1. **databench 补 2 个真配置**（fake/base → 加 `qwen3:8b` 与 `qwen2.5:7b`，本机 ollama 已有）。跑完即有 power≥3，能算第一版 k_eff。预计 20 题 × 2 配置 × 热加载 ≈ 10 分钟。
2. **evalkit 把 18 条恒 1.0 移出计分分母**，或按 hard 类加 3 条更难的边界题（如 2**200、浮点精度陷阱）。现有 28 条有效区分任务已够做一版结论。
3. **wse-bench 补测 54 条**按 skill 配方走：锁目标 → 同评测器 → 3-4B 配置 → 按模型顺序 → 每题落盘 → 区分度自检。

## 复跑命令

```bash
python3 scripts/task_spectra.py coverage agent-evalkit/results --out reports/task_spectra_evalkit_20261119.json
python3 scripts/task_spectra.py coverage agent-databench/results --out reports/task_spectra_databench_20261119.json
python3 scripts/task_spectra.py selftest   # 9 项断言，3 个反证
```
