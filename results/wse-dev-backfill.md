# wse-bench dev 24 条补测（第三步）

> 承接 `task_spectra_wse_20261119.json` 的 54 条 never_measured。本轮用本机 ollama 补测
> **只补 dev split 的 24 条**（hidden 16 + fresh 14 受 `provenance_lock` / `fresh_test_set`
> 保护，不碰）。评分复用它自己的确定性 `score_contains`，不引入 LLM judge。

## 1. 结果：L2 102→126，L3 89→104，门全绿

```
补测前: L1=156  L2=102  L3=89   const=13  inflation=1.529
补测后: L1=156  L2=126  L3=104  const=22  inflation=1.238
```

- 24 条 dev 题被 3 个本机配置（qwen2.5:3b / qwen2.5:1.5b / llama3.2:3b）各测一次 → power=3，全部进入 L2（+24）。
- 其中 **15 条有真区分度**（进 L3），**9 条成了恒定列**（8 条恒 1.0 + T152 恒 0.0）→ L3 +15、const +9。
- 覆盖率门 `task_coverage_gate.py reports/task_spectra_wse_backfilled_20261119.json`：**fails=[]**，仅 `W2 hash 失配 3 条`（T029/030/031，只报告不重写）。
- 剩余 30 条待补测 = hidden 16 + fresh 14，**刻意未碰**。

三模型读数（24 题，temperature=0，确定性 contains 判定）：

| 配置 | pass | mean_score |
|---|---|---|
| qwen2.5:3b | 17/24 | 0.785 |
| qwen2.5:1.5b | 14/24 | 0.670 |
| llama3.2:3b | 15/24 | 0.695 |

## 2. 新发现的第五种现象：模型输出语言漂移（T152 恒 0）

T152（protocol_mcp，dev，contains）三模型**全部 0.0**。查提交原文：

- 题目要求「逐字引用 MCP 文档中关于 sampling…的原文短语」，`required` 是英文短语
  （`request LLM sampling` / `human in the loop` / `preference system`）。
- 三模型都把英文原文**翻译成中文**后作答（「让服务器向客户端请求 LLM 的采样」…），
  而 `norm()` 只折叠空白+小写、不做跨语言映射 → 全不命中 → 0.0。

**归因**：这不是能力不足，是判据失真——评分器只认原文语言，题目 prompt 是中文、材料是英文，
模型选择译而非引。与 evalkit「同一事实同时出现在丢失与幻觉列表」同属**判据侧失效**，
不是**被测侧失效**。

**修法（下一步）**：在 `score_contains` 外包一层语言提示（"逐字引用，保持原文语言"）或
对 contains 题加语言一致性校验（提交与 required 的字符集重叠率），否则任何中文 prompt 的
英文材料题都会系统性假负。

## 3. 附带发现：训练数据被当成评测配置（口径污染）

`wse-bench-public/sft_dev_r18.jsonl` 与 `sft_dev_r22_500plus.jsonl` 是 **SFT 训练样本**
（字段 instruction/input/output/task_id/score），因含 `task_id`+`score` 被 `_extract_configs`
当成两个"配置"。实测影响：

```
含 sft 当配置: n_cfg=52  L1=126 L2=126 L3=104 const=22
剔除 sft:      n_cfg=50  L1=126 L2=126 L3=104 const=22
```

L2/L3/const 完全不变（它们只覆盖已测题，不增不减），只污染 `n_configs` 与 `k_eff` 分母。
**结论**：口径数字（L1/L2/L3）可信；`n_configs=52` 应为 50，k_eff 的分母虚高 2。
这与 databench 的 `_plumbing_archive` 同类——训练/管道产物混入评测结果目录。

## 4. 复算命令

```bash
python3 scripts/wse_dev_backfill.py --selftest   # 8 项含反证，全过
python3 scripts/wse_dev_backfill.py --plan       # 24 条 dev，全 contains
python3 scripts/wse_dev_backfill.py --run --out wse-bench-public/results_dev_backfill
python3 scripts/task_spectra.py scores wse-bench-public \
  --manifest wse-bench-public/assets/bench_manifest.json \
  --json --out reports/task_spectra_wse_backfilled_20261119.json
python3 scripts/task_coverage_gate.py reports/task_spectra_wse_backfilled_20261119.json   # exit 0
```

## 5. 边界声明（事实纪律）

- 补测用本机 Qwen2.5/Llama3.2 小模型，**不是原论文的评测模型**；这些读数只用于"覆盖率口径"，
  不构成"某个模型能力"的结论。
- 只补 dev，hidden/fresh 30 条保持未测（保护 fresh 只测一次的口径）。
- T029/030/031 hash 失配是既有状态（占位符探针题），本轮只报告不改写。
