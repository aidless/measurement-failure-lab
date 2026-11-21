# measurement-failure-lab

**评测饱和到底是任务变简单了，还是测量失效了？**

这个仓库是一套可复算的审计工具 + 一个已经跑完的实证：把「跑分收敛」拆成**判据失真、口径错配、区分度塌缩**三层，逐层给数、逐层给判据，哪一坏修哪一。

两件互补的东西放在一起：

| 部分 | 问题 | 产物 |
|---|---|---|
| **A. 覆盖率审计** | 一个 benchmark 清单写着 N 条，实际有多少条真被测出区分度？ | `scripts/task_spectra.py` + `scripts/task_coverage_gate.py` |
| **B. 权重编辑实验（E1）** | 对模型权重做极小稀疏编辑（0.016%），增益是真的还是测量造的？ | `scripts/cloud_e1_run.py` 等 + `results/e1-weight-edit-report.md` |

B 的结论不是重点，**B 在过程中被五个测量陷阱各翻转一次结论**才是。两个部分共享同一个命题：**当你看到一个数字，先问这把尺子怎么刻的。**

---

## 快速开始

无需 GPU 的部分（覆盖率审计）立刻可跑：

```bash
# 自检：10 项断言，含 2 条「注入违规必须变红」的反证
python3 scripts/task_spectra.py selftest

# 拿仓库自带的演示夹具跑一遍（4 配置 × 6 任务，其中 2 条任务恒定）
python3 scripts/task_spectra.py scores data/demo_results
python3 scripts/task_coverage_gate.py --selftest
```

`task_spectra.py` 能通过就能跑真实数据：

```bash
python3 scripts/task_spectra.py coverage <results-dir> \
  --manifest <bench_manifest.json> --out audit.json --json
```

结果文件支持三种形态（自动识别）：

```jsonc
{"scores": {"T001": 1.0, ...}, "overall": 0.5, "verified": true, "provenance": {"llm": "...", "config": "..."}}
{"T001": 1.0, "T002": 0.0, ...}
[{"id": "T001", "score": 1.0}, ...]
```

需要 GPU 的部分（E1 权重编辑）见 [`results/e1-weight-edit-report.md`](results/e1-weight-edit-report.md) 的复现命令；单卡 6 GB 即可跑 1.5B 版本：

```bash
python3 scripts/e1_determinism_local.py --selftest
python3 scripts/e1_determinism_local.py \
  --model {MODEL_DIR}/Qwen2.5-1.5B \
  --items data/d1d2_items.jsonl \
  --out determinism_local.json
```

---

## 三层口径（任何对外数字必须落到一层）

| 层 | 定义 | 该怎么说 |
|---|---|---|
| **L1** | manifest 清单条数 | 只能说「清单 N 条」 |
| **L2** | 有 ≥3 种**不同**得分向量测过的条数 | **实测规模，该引用的数字** |
| **L3** | L2 中仍有区分度的条数 | 主张冗余/低秩时用 |

`power < 3` 时任意两点共线，相关系数只能是 ±1 —— 所以 L2 的门槛是 3，不是 2。恒定任务先于一切统计被发现：它不抬高 σ1，但它让 L2 里混进永远不可分的条目。

**高估倍数 = L1 / L2。** 这是唯一诚实的"这个 benchmark 有多虚"的指标。

---

## 已跑出的实测读数（三个真实 benchmark）

| 资产 | L1 | L2 | L3 | 高估倍数 | 关键现象 |
|---|---|---|---|---|---|
| wse-bench | 156 | 126 | 104 | 1.24× | 22 条恒定：21 条恒 1.0 + 1 条**恒 0.0** |
| agent-evalkit（原始） | 62 | 46 | 28 | 1.35× | 18 条全饱和（const=1.0） |
| agent-evalkit（移出 18 条后） | 44 | 28 | 28 | 1.57× | const=0；L3/L2=1.0 → 剩下的每条**都有区分度** |
| agent-databench | 20 | 20 | **1** | 1.00× | 19/20 对 1.5B–7B **全部 0 分** |

evalkit 那两行要连起来读：移出 18 条恒定后 L1 从 62 掉到 44（因为被移出的条目从清单里也删了），但 L3 纹丝不动（28）—— **被删掉的 18 条一条都不曾贡献区分度**。这才是"虚"的准确定义。

原始读数：[`results/audit_wse-bench.json`](results/audit_wse-bench.json)、[`results/audit_agent-evalkit.json`](results/audit_agent-evalkit.json)、[`results/audit_agent-evalkit-drop18-recompute.json`](results/audit_agent-evalkit-drop18-recompute.json)、[`results/audit_agent-databench.json`](results/audit_agent-databench.json)。

最后一行是全仓库最值得看一眼的数：**power 达标（L2=20）但 19/20 任务对三个模型全部 0 分。** 与 wse 的「全部 1.0」恰是两个方向 —— 恒定任务有**天花板和地板**两种，只查一个会漏掉一半。

---

## 恒定任务的两个方向

| 方向 | 表现 | 意味着什么 | 例子 |
|---|---|---|---|
| 天花板塌缩 | 全配置 1.0 | 题目太简单，测不出差异 | wse-bench 21 条、evalkit 18 条 |
| 地板塌陷 | 全配置 0.0 | 题目对被测模型不可能，同样测不出差异 | databench 19/20 条、wse-bench T152（1 条） |

只做 `const==1` 检测的审计会漏掉整个地基塌陷的那一半。

---

## E1 实验：五个测量陷阱

这部分的价值在**方法**不在结论。同一个实验里，五处测量细节每处都曾单独制造一个假结论：

| # | 陷阱 | 假象 | 修正后 |
|---|---|---|---|
| 1 | `evaluate()` 没有恢复 `model.eval()` | 编辑后系统性偏低，报 0/30 | 4/30；避免"编辑毁灭性"误判 |
| 2 | `max_new` 与提示档位当常数 | D2 全 0，读成"任务不可解" | fewshot 档实测 12/15，96 token 下 15 条全截断成半句 |
| 3 | **编辑符号对负权重走反** | 梯度臂增益为负，读成"梯度≈随机" | `step = sign(g)·sign(w)`；修正后 logit 30/30 全改善 |
| 4 | `last_number` 抓到重复循环截断处的尾数 | 生成臂从 18/30 崩到 8/30 | 优先取「答案：<数值>」；after 8/30 → **18/30** |
| 5 | 把"重复循环"当生成失败的原因 | 看似能解释 logit↑ 与生成净 0 脱节 | 三解码条件判决：压制重复后配对 delta −4/+3，方向不一致 → 该假设**被证伪** |

陷阱 3 的算术（torch 四象限实测）：

```text
w=+1.0  int16 +1 → +1.007812(幅值↑)  -1 → +0.996094(幅值↓)   g=+1 要下降 → 用 -1
w=-1.0  int16 +1 → -1.007812(幅值↑)  -1 → -0.996094(幅值↓)   g=+1 要下降 → 用 +1
⇒ bf16 的幅值升降与符号位无关；梯度下降 step 必须 = sign(g)·sign(w)
```

陷阱 4 的实证（同一份生成输出，两种判分器）：

```text
D2-F3-02  输出含「答案：11」（正确），但重复循环截断在「1. 从第 1 行…」
         answer_number = 11  ✓        last_number = 1  ✗
```

完整报告：[`results/e1-weight-edit-report.md`](results/e1-weight-edit-report.md)（含判决性实验设计、预注册判据、噪声带归因、确定性格）。

---

## 一个被反复验证的教训

**两次重跑结果不同，先查脚本版本，不要先怪运行时随机。**

E1 里曾有"±1~3 噪声带"的说法，后来用离线 join 对 5 个独立进程做归因：按判分口径分两簇后，**簇内逐题通过集对称差 = 0**，簇间差异 3 条且方向单一 —— 全部来自三处脚本变更（判分器 `last_number`→`answer_number`、输出截断 `[:300]`→全文、D2 提示档 `plain/96`→`fewshot/512`）。`temperature=0` 下生成是确定的；看起来的"噪声"是尺子被换了。

复算脚本：[`scripts/e1_offline_attribution.py`](scripts/e1_offline_attribution.py)；直接验证：[`scripts/e1_determinism_local.py`](scripts/e1_determinism_local.py)（`verdict=DETERMINISTIC`，含"同通过集换文本必须判不确定"的反证）。

---

## 仓库结构

```text
scripts/     审计与实验脚本（每个都有 --selftest，自检必须能变红）
data/        E1 题库（30 题）+ 覆盖率演示夹具
results/     已跑出的读数（JSON 汇总，不含原始大文件）
experiments/ E1 实验设计与题型分类
evidence/    立项事实快照与云端探针记录
```

## 纪律

1. **数字可复算** —— 任何百分数都能从落盘证据重算，重算不一致即阻塞。
2. **门必须能失败** —— 每个校验门配一个"注入违规必须 exit≠0"的反证，否则是装饰。
3. **不合并不同口径** —— 改判分器/改 token 预算/改提示档 = 换尺子，必须写进产物，不许悄悄并表。

## 许可

MIT，见 [LICENSE](LICENSE)。
