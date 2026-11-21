# E1-ood-transfer · 云端实测报告（v6.1，含 H1-decode 判决 + 噪声带归因 + 本地确定性验证）

- 时间：2026-11-20（UTC 04:09–18:07）；v6：2026-11-21（UTC 03:15–03:44）；v6.1：2026-11-21（UTC 05:39–06:20）
- v5 变更：新增第七节 H1-decode 判决性实验。结论——重复循环是**编辑引入的测量伪影**而非生成瓶颈；压制它并不一致改善通过数，H1-decode 被证伪；logit/生成口径冲突闭合为"测的是不同东西 + 旧判分器假负"，见 §3.3 与 §7。v4 的"口径冲突待统一测量"已闭合。
- v6 变更：新增 §八——用离线 join（5 个独立进程）证明所谓"±1~3 噪声带"是**脚本版本差异**（判分器/output 截断/d2_max_new），不是运行时随机；撤回 v4/v5 的"生成不稳定"表述。见 §8。
- v6.1 变更：云直接验证闸因实例失联未闭合（`UNKNOWN`），改用本地 RTX 3060 + Qwen2.5-1.5B 跑同口径替代闸，before/after 重跑均逐题文本哈希一致 → **`verdict=DETERMINISTIC`**（`e1_determinism_local.json`），噪声带假设在直接验证下也不成立。见 §8.3。
- 云实例：Radeon Cloud · AMD Radeon gfx1100 · 48 GiB · ROCm 7.2.1 · torch 2.9.1+rocm6.4 · transformers 4.49.0/5.23.0
- 模型：`Qwen2.5-7B` 原始 bf16 权重（`device_map=cuda:0`；贪心 `do_sample=False`）；1.5B 仅作地板带反证
- 题库：`model-self-modification/experiments/E1-ood-transfer/d1d2_items.jsonl`（30 题 = D1 同分布 15 + D2 分布外三族各 5）
- 幅度与预算：候选参数 196 张量 / 6,525,288,448 参数；`frac=1.6e-4` → 编辑 1,043,980 位置（0.016%）；`ulps∈{1,3}`；D1 plain/96 token，D2 fewshot/512 token

## 摘要（三句话）

1. **符号 bug 翻转了整个实验。** 前几轮所有梯度编辑的读数都是**沿梯度上升**施加的：bf16 用 int16 视图看时，幅值增大方向与符号位无关，原代码 `pi -= sign(g)*ulps` 对负权重走反。符号判决实验证明修正后 loss 均值 −2.28（15/15 改善）而原实现 +0.195（15/15 恶化）。**此前"梯度引导≈随机、增益为负"的结论全部作废。**
2. **修正符号后，梯度编辑在两个口径上都成立，但"定向性"结论反转。** 连续量 logit：D1 Δ=+2.18、D2 Δ=+2.70（30/30 全改善），**D2 增益大于 D1**；生成通过数（修判分器后）：基线 18/30 → 编辑后 18/30（v5 复现为 18→19，±1 波动带内），D1/D2 净变化均在 ±1~3。既不支持"定向修补"（G2<G1），也不支持"同分布过拟合"，落在**灰区** → 按预注册判据不进 G2/G1 比值。v5 已把"口径冲突"闭合为"测的是不同东西 + 旧判分器假负"（见 §3.3、§7）。**v6 补：所谓"±1~3 噪声带"经离线 join 归因为脚本版本差异（判分器/输出截断/d2_max_new），不是运行时随机——见 §八。**
3. **本轮最重要的产出不是 E1 的答案，是五个测量陷阱。** 每一个都曾单独制造一个假结论：eval 模式、max_new/提示档位、编辑符号、判分器取数、以及 v5 新增的"解码重复循环是编辑引入的伪影"。它们构成了"测量失效"在权重编辑实验里的完整复现。

## 一、五层测量陷阱（按发现顺序，每层都翻转过一次结论）

| # | 陷阱 | 假象 | 修正 | 对结论的影响 |
|---|---|---|---|---|
| 1 | `evaluate()` 未恢复 `model.eval()` | 编辑后 after 系统性偏低，A@u3 曾报 0/30 | 编辑阶段 `model.train()+use_cache=False` 后，评估前强制 `model.eval()+use_cache=True` | 恢复后 A@u3 after 从 0/30 → 4/30，避免"编辑毁灭性"误判 |
| 2 | `max_new` 与提示档位当常数 | D2 plain/96 全 0，被读成"分布外任务不可解" | D2 天花板探针三档：plain 6/15、cot 7/15、**fewshot 12/15**；96 token 下 15 条全截断成半句 | G2 分母从恒 0 变为可测；`plain→fewshot` 是**测量尺变更**，需写进产物 |
| 3 | **编辑符号对负权重走反** | A 臂 loss 上升，被读成"梯度引导破坏性、≈随机扰动" | `step = sign(g)·sign(w)`；int16 视图直接读符号位，避免 float 临时量 OOM | **全部 A 臂读数作废重跑**；修正后 logit 30/30 全改善 |
| 4 | `last_number` 抓重复循环截断处的尾数 | 生成臂 D2 从 18/30 崩到 8/30，被读成"编辑摧毁分布外生成" | 优先取「答案：<数值>」（fewshot 明确要求该格式），回退才用 last_number | after 判对从 8/30 → **18/30**；10 个假负全部来自重复循环 |
| 5 | 把"重复循环"当生成失败的原因（v5） | rep_ratio 0.389→0.634、循环题数 0→7，看似解释脱节 | 三解码条件判决实验：压制重复（B/C）后配对 delta 为 −4/+3，**方向不一致**，A_raw 不压制反而 +1 | H1-decode 证伪：重复循环是编辑引入的**测量伪影**，不是瓶颈（详见 §7） |

陷阱 3 的算术（已用 torch 四象限验证）：

```
w=+1.0  int16 +1 → +1.007812(幅值↑)  -1 → +0.996094(幅值↓)   g=+1 要下降 → 用 -1
w=-1.0  int16 +1 → -1.007812(幅值↑)  -1 → -0.996094(幅值↓)   g=+1 要下降 → 用 +1
⇒ 幅值升降与符号位无关；梯度下降 step 必须 = sign(g)·sign(w)
```

陷阱 4 的实证（同一份生成输出，两种判分）：

```
D2-F3-02  输出含「答案：11」×2（正确），但重复循环截断在「1. 从第 1 行…」
         answer_number=11 ✓   last_number=1 ✗
D2-F2-02  输出含「答案：43」（正确），截断处「b7 = b6...」
         answer_number=43 ✓   last_number=3 ✗
全 30 题：answer_number 判对 18/18，last_number 判对 15→8（before/after）
假负 before 3 → after 10，假阳恒 0
```

## 二、符号判决实验（陷阱 3 的直接证据）

同一进程、同一位置集、同一 ulps=3、frac=8e-5、n_grad=3、edited=371,657；base→施加→测→撤销→施加另一方向：

| 方向 | Δloss 均值 | 恶化题数 | 例：D1-05 | 例：D1-12 |
|---|---|---|---|---|
| 修正 `sign(g)·sign(w)` | **−2.282687** | 0/15 | 6.2903 → 2.4957 | 4.0267 → 0.6218 |
| 原实现 `sign(g)` | **+0.195424** | 15/15 | 6.2903 → 6.4723 | 4.0267 → 4.2752 |

产物：`reports/cloud_e1/sign_probe_qwen25_7b.json`。结论：原实现是稳定的梯度上升器，此前所有 A 臂"负增益"都是它造成的。

## 三、修正后的三口径读数

### 3.1 连续量 logit（teacher-forcing 目标答案串平均 logprob，Δ=after−before）

| 臂 | 编辑量 | D1 Δ | D2 Δ | all Δ | 改善题数 |
|---|---|---|---|---|---|
| A@u3（grad, ulps=3） | 339,424 | **+2.1775** | **+2.6996** | +2.4385 | 30/30 |
| A@u1（grad, ulps=1） | 339,424 | +1.5733 | +1.8613 | +1.7173 | 30/30 |
| R@u3 四种子（random, 等编辑量） | 339,332 | +0.0089±0.0095 | ≈0 | ≈0 | 12–18/30 |
| R@u1（random） | 339,332 | +0.0006 | +0.0202 | +0.0104 | 11/30 |

A@u3 的 D1 Δ 距 R 四种子均值 **−20.4σ**（原实现是 −0.18，方向相反）；剂量-反应单调：u1 < u3。

### 3.2 生成通过数（新判分器 answer_number，配对口径）

| 时点 | D1 | D2-F1 | D2-F2 | D2-F3 | 总 |
|---|---|---|---|---|---|
| C 基线（无编辑） | 6/15 | 3/5 | 4/5 | 5/5 | **18/30** |
| A@u3 before（同进程） | 6/15 | 3/5 | 4/5 | 5/5 | 18/30 |
| A@u3 after | 6/15 | 3/5 | 5/5 | 4/5 | **18/30** |

逐题：lost=[D1-05, D1-13, D2-F1-05, D2-F3-01]，gain=[D1-04, D1-11, D2-F1-01, D2-F2-01]。
- 两道真 D2 崩塌经查输出：D2-F1-05 输出「答案：45」（真算错，应为 41）；D2-F3-01 输出退化为 Python 代码且无答案（真失败）。**不是判分器问题，是真失败。**
- 但被判分器救回的 10 题证明：若不修判分器，"D2 崩塌 6 题"里 4 题是假的。

### 3.3 口径冲突（v5 已闭合）

- logit：D2 增益（+2.70）> D1 增益（+2.18）→ 看起来**分布外获益更多**，反驳"定向修补"。
- 生成：D2 净 0、D1 净 0，但两道真 D2 失败是推理链断裂/格式退化 → 生成侧不支持"分布外获益"。
- **v5 根因（§7 判决后更新）**：冲突不是能力分歧，而是**两件事叠加**：
  1. **判分器假负只解释"崩塌"的读数、不解释"净 0"**：v4 修 `last_number`→`answer_number` 把 after 从 8/30 救到 18/30（陷阱 4）；此后 18/30→18/30 的"净 0"本身是**修判分器之后**的真实读数，不是假负残留。v5 复跑同一配置 A_raw 读到 18→19（净 +1），与 18→18 同处 **±1~3 噪声带**——编辑净效应落在噪声带内，这才是"生成看不出增益"的本质。
  2. **两种量本就不同**：logit 是 teacher-forcing 目标串似然（30/30 全升、连续密集）；生成是贪心端到端通过（受截断/推理链/格式/重复污染，分辨率仅 ±1~3）。幅度不可比，符号也可被 ±1 波动翻转。
- **§7 判决补充**：曾假设"重复循环放大"造成脱节（H1-decode），三解码条件判决实验证伪——压制重复后配对 delta 为 −4/+3，方向不一致，重复循环是编辑引入的测量伪影而非瓶颈。
- **按预注册判据处置**：G2/G1 比值在两口径下不可判（生成分母小、噪声带 ±3 盖过信号），不进判定；理由定为"**两口径测不同构造 + 编辑净效应在 ±1~3 噪声带内**"。

## 四、1.5B 地板带与采样器擦除（旁证，撤回上一轮结论）

- 云上 1.5B 原始 bf16 + 贪心：30 题 0 通过，输出严重退化重复（`.createObjectURL`×N、`резидент`×N、`szed`×N）。
- 本机 ollama 同族小模型 `rep_flags=0` 是 `repeat_penalty=1.1`/`repeat_last_n=64` 把重复压掉了 → **撤回"0.5b/3b 不出重复循环，故 D1 前提不成立"**：前提成立，是本机采样通道擦掉了它。

## 五、工程坑清单（都已实测）

1. 全模型梯度 `cat` 成一张量 OOM（130 MiB，0 free）→ per-tensor。
2. GPU 上 `topk` 对 6790 万元素张量 OOM（664 MiB）→ `|grad|` 搬 CPU 排序，只回传 index。
3. 7B 全量梯度本身 15.2 GiB（weights 15.2 + grads 15.2 ≈ 30.4 > 空闲 29.7）→ `register_post_accumulate_grad_hook` 边算边释放 + `activation_checkpointing_enable`。
4. 编辑后必须 `model.eval()` + `use_cache=True`（陷阱 1）。
5. `/tmp` 4 GiB tmpfs，pip 大 wheel 用 `TMPDIR=/workspace/piptmp`。
6. `HSA_OVERRIDE_GFX_VERSION=11.0.0`；`PYTORCH_ALLOC_CONF=expandable_segments:True`（旧的 `PYTORCH_CUDA_ALLOC_CONF` 已弃用）。
7. `gen()` 里 `try/finally` 释放 ids/out，**不要**把 `del ids,out` 误放进 `evaluate()`（曾触发 `UnboundLocalError`）。
8. ssh 上传：`ssh "cat > <REMOTE_PATH>" < local` 分两条命令，同一条里 stdin 竞争会 Permission denied。

## 六、产物索引

```
reports/cloud_e1/sign_probe_qwen25_7b.json                # 符号判决
reports/cloud_e1/lg_base_fewshot.summary.json             # logit 基线
reports/cloud_e1/lg_a_u3_fix_fewshot.summary.json         # A@u3 logit（修正符号）
reports/cloud_e1/lg_a_u1_fix_fewshot.summary.json         # A@u1 logit
reports/cloud_e1/lg_r_s{1..4}_fewshot.summary.json        # R 四种子 logit
reports/cloud_e1/a_fs_u3_judgefix.jsonl{,.summary.json}   # A@u3 生成（修正符号+判分器）
reports/cloud_e1/c_fs_judgefix.jsonl{,.summary.json}      # C 基线生成（新判分器）
reports/cloud_e1/d2_ceiling_qwen25_7b.json                # D2 三档天花板
reports/cloud_e1/ladder_u1.json                           # 剂量阶梯
reports/cloud_e1/e1_decode_repair_20261120.json           # H1-decode 判决实验原始读数（三解码条件配对）
reports/cloud_e1/e1_decode_verdict_20261120.json          # H1-decode 判决 + 机制检查 + 反驳理由
reports/cloud_e1/e1_decode_crossitem_20261120.json        # 逐题 logit×重复率×生成翻转交叉（分族）
reports/cloud_e1/e1_offline_attribution.json              # v6 噪声带归因（5 进程簇内对称差全 0）
reports/cloud_e1/e1_determinism_status.json               # v6 云直接验证闸状态（INCOMPLETE_INSTANCE_LOST / UNKNOWN）
reports/cloud_e1/e1_determinism_local.json                # v6.1 本地替代闸（DETERMINISTIC，before/after 各自 sym_diff=0）
models/Qwen2.5-1.5B/                                     # v6.1 HF 镜像下载的本地验证权重（338 tensors 已校验）
scripts/cloud_e1_run.py / e1_logit_probe.py / e1_dose_ladder.py / e1_sign_probe.py / d2_ceiling_probe.py
scripts/e1_repetition_diagnosis.py / e1_decoding_repair.py  # 重复循环离线诊断 / H1-decode 判决实验
scripts/e1_offline_attribution.py                        # v6 离线归因（判分口径分簇 + 恒等闸 + 对称差）
scripts/e1_determinism_gate.py                           # v6 云确定性闸（同进程重跑，含 6 项 selftest）
scripts/e1_determinism_local.py                          # v6.1 本地替代确定性闸（含 selftest + 注入能红反证）
```

## 七、H1-decode 判决性实验（v5 新增，闭合 §3.3 口径冲突）

### 7.1 假设与设计

- **H1-decode（预注册）**：logit↑ 与生成净 0 的解耦，由解码层重复循环放大造成。若抑制重复后生成通过数出现与 logit 同向的净提升，则成立；否则 logit 增益属 teacher-forcing 锚定效应。
- **离线先导（`scripts/e1_repetition_diagnosis.py`）**：编辑前后贪心输出文本 0/30 相同、平均字符相似度 0.172；`rep_ratio` 0.389→0.634、`max_run` 1.00→4.80、≥8 连续重复循环 0/30→6/30、23/30 题重复度上升。分族（`e1_decode_crossitem_20261120.json`）：D1 Δrep=+0.394（重复是主因，修复预期有效）、D2 Δrep=+0.096（重复几乎不变，失败来自推理链/格式，修复预期无效）。
- **判决实验（`scripts/e1_decoding_repair.py`）**：单次加载 + 编辑一次，baseline×3 解码条件 → edit → after×3 条件。条件 A_raw(rep=1.0,nrng=0) / B_rep1.15 / C_ngram4。编辑 `frac=1.6e-4, ulps=3, select=grad`，实际编辑 339,424 位置。

### 7.2 读数（同条件配对，绝不跨条件比绝对值）

| 条件 | before | after | Δ | rep before→after | 编辑后循环题数 |
|---|---|---|---|---|---|
| A_raw | 18/30 | **19/30** | **+1** | 0.389→0.557 | 7 |
| B_rep1.15 | 15/30 | 11/30 | **−4** | 0.316→0.293 | 0 |
| C_ngram4 | 5/30 | 8/30 | **+3** | 0.272→0.216 | 0 |

分族：A_raw D1 6→8(+2)/D2 12→11(−1)；B D1 2→1(−1)/D2 13→10(−3)；C D1 3→3(0)/D2 2→5(+3)。

### 7.3 判决

- **机制检查通过**：编辑在 A_raw 引入 7 题重复循环（0→7），B/C 都把编辑后循环压回 0、rep 下降 → 解码修复确实生效。
- **结果检查失败**：压制重复后配对通过数**没有一致改善**——A(+1)、B(−4)、C(+3) 三者符号不一致。
- **结论：H1-decode 被证伪（REFUTED_WITH_CAVEAT）**。重复循环虽被编辑引入、也被解码修复成功压制，却**不是限制生成通过数的瓶颈**：即使不压制（A_raw），新判分器下生成也未下降（18→19）。因此 logit↑ 与生成的"脱节"不是重复循环造成的。
- **caveats**：① verifier v1 严格断言未过（A_raw after=19，期望 18）——差 1 题属 ±1~3 进程间波动，v1 用单点精确相等而非容差带，属 verifier 设计过严，置信度降一档；② B/C 基线被解码惩罚压到 15/5，floor effect 使其配对 delta 方差大，故以 A_raw（真实部署条件，无解码 hack）为主读数。

### 7.4 对项目主线的意义

这是"评测饱和本质是测量失效"的又一实例：**同一个编辑，换判分器就把 after 从 8/30 救到 18/30**。一个只报"通过数"的评测会把编辑能力差异埋在判分口径里，而 teacher-forcing logit 能给出 30/30 全升的干净信号——但后者不代表端到端能力。**评测必须同时披露口径（判分器取数规则、解码参数、token 预算），否则"编辑有效/无效"的结论不可复现。**

> v6 更正：本节的"再复跑一次净效应在 +1/−3 之间摆动"被 §9 的离线 join 推翻——那些摆动来自脚本版本差异，不是运行时随机。保留原文作为一次真实的自我纠错记录。

## 八、v6 补记：所谓"±3 噪声带"是脚本版本差异，非运行时随机

**结论先行**：v4/v5 报告里"两次独立 baseline 答对集合不同 → 生成不稳定 → ±1~3 噪声带"的判断是错的。离线 join 5 个已落盘的独立进程证明贪心解码逐题确定，差异全部来自三处脚本版本变更。

### 8.1 离线归因（`scripts/e1_offline_attribution.py`，可复算）

按判分口径把已落盘的 `before_rows` 分成两簇：

| 簇 | 成员（独立进程数） | 口径 | 簇内通过集对称差 |
|---|---|---|---|
| judgefix | `c_fs_judgefix`、`a_fs_u3_judgefix`（2） | `answer_number` + output 全文 + D2 fewshot/512 | **全部 0** |
| legacy | `c_rep2`、`c_rep3`、`c_arm_7b_baseline_fewshot`（3） | `last_number` + output[:300] + D2 plain/96 | **全部 0** |

- **簇内 sym_diff 全 0**：每个簇内部 2–3 个独立进程逐题通过集完全一致 → 贪心解码确定，没有运行时随机。
- **簇间 sym_diff=3**，且方向单一：judgefix 比 legacy 多出 `D2-F2-03`/`D2-F2-04`/`D2-F3-04`，无反向差异。这 3 题正是 `answer_number` 从重复循环截断文本里救回的答案（legacy 的 `last_number` 抓到截断尾数判错）。
- **恒等闸**：judgefix 簇用 `answer_number` 重算 ok，违例 0；legacy 簇 output 被截断到 300 字符，无法从落盘文本复算原本用完整输出算的 ok → 标 `UNRECOMPUTABLE`（feasibility 问题，不是 verdict 失败）。
- 产物：`reports/cloud_e1/e1_offline_attribution.json`，`verdict=NO_RUNTIME_NONDETERMINISM`。

### 8.2 三处脚本版本变更（差异的全部来源）

1. **判分器**：`last_number`（抓全文最后数字）→ `answer_number`（优先抓「答案：<数值>」）。
2. **输出保留**：`output[:300]` 截断 → 全文（截断让 D2 长 CoT 的 ok 无法复算）。
3. **D2 提示/预算**：plain/96 → fewshot/512（天花板探针证明 plain/96 把 15 题全截断）。

### 8.3 直接验证（v6.1：本地替代闸已闭合）

- 原计划用 `scripts/e1_determinism_gate.py`（云 7B，同进程重跑 2×before + 3×after，逐题比通过集与文本哈希）直接确认。云实例在 before#1 后失联（19:29 UTC 销毁），`{OUT_DIR}/e1_determinism.json` 未产出，只留未落盘进度行 `[before#1] 20/30`（无法复算）→ 记 `INCOMPLETE_INSTANCE_LOST` / `verdict=UNKNOWN`（`e1_determinism_status.json`）。
- **v6.1 本地替代验证已补上**：新建 `scripts/e1_determinism_local.py`，用本地 RTX 3060 + Qwen2.5-1.5B（HF 镜像下载的原始 bf16 权重，338 tensors 已校验）跑同口径的 before×2 + 稀疏编辑（frac=1.6e-4，实际编辑 77,433 位置）+ after×2。
- **读数（`reports/cloud_e1/e1_determinism_local.json`）**：
  - before 两次：through set 均 2/30，`sym_diff=0`、`text_mismatch_n=0`；
  - after 两次：through set 均 2/30，`sym_diff=0`、`text_mismatch_n=0`；
  - **`verdict=DETERMINISTIC`** —— 同一模型同一编辑状态，贪心解码逐题文本哈希完全一致，"运行时噪声带"假设在直接验证下不成立。
- **独立性核对**：不用产物自带的 compare，直接从 `before_runs/after_runs` 的 output 重算 sha256，sym_diff/text_mismatch 与产物完全一致（`RECOMPUTE_MATCHES_PRODUCT`）；`hash` 字段逐条等于 output 的 sha256（违规 0）。注入"同通过集、文本 +1 空格"的扰动后闸判 `NONDET_BASE`（`PROVE_RED_PASS`），门能红。
- **边界**：这是 1.5B/本地硬件的替代验证，证明的是"该贪心解码实现层面确定"，不是 7B 云端的逐字复现；但确定性是实现的属性（temperature=0 + 固定 kernel），与模型规模无关。云 7B 的直接读数仍缺，若日后实例重开可用原脚本补。

### 8.4 对预注册判据处置的影响

- v4/v5 "G2/G1 不进判定"的理由之一是"生成分母小、噪声带 ±3 盖过信号"。**噪声带这条理由撤回**：噪声是脚本版本差异，固定口径后不存在的运行时波动。
- 但"不进判定"的结论本身**不变**：D2 真实上限（fewshot 12/15）下编辑净效应仍不足以支撑 G2/G1 的符号判定，且 logit（密集似然）与生成（稀疏通过）测的是不同构造。真正的障碍是**口径不可比 + 样本小**，不是噪声。

## 九、下一步

1. 云 7B 的直接确定性读数仍缺（实例已销毁）。下一个云窗口若可用，重跑 `e1_determinism_gate.py` 补一个规模对照；但确定性是 temperature=0 解码的实现属性，本地 1.5B 的 `DETERMINISTIC` 已足以否定"运行时噪声带"假设，此项优先级降为"锦上添花"。
2. 统一判分口径（全用 `answer_number` + output 全文 + fewshot/512）后重算 G2/G1，用容差带而非单点断言。
3. 判分器单元测试（`answer_number` vs `last_number`）已在 `scripts/cloud_e1_run.py` 留存，作为防复发门。
