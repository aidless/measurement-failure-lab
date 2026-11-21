# E0 — AMD 云权重编辑通道前置验证（2026-11-20）

> 目的：在 E1（分布外迁移检验）之前，先证明这台 AMD 云实例能完成"加载小模型→改少量权重→生成→对比"的完整链路。
> **本文件只记实测，不记声称**；所有数字来自本轮 ssh 实跑。

## 1. 通道与版本（实测）

| 项 | 实测值 |
|---|---|
| GPU | AMD Radeon Graphics, gfx1100 (0x744b), ~48GB VRAM |
| ROCm（系统） | 7.2.1 |
| torch | 2.9.1+rocm6.4（`pip index` 选 6.4 而非 7.2：gfx1100 在 6.4 支持更成熟） |
| HIP | 6.4.43484-123eb5128 |
| torch.cuda.is_available() | True, device_count=1 |
| matmul | 1024×1024 bf16 @ on cuda 通过（HSA_OVERRIDE_GFX_VERSION=11.0.0） |
| transformers | 4.49.0；accelerate 1.3.0；safetensors 0.5.2 |

**踩坑（事实）**：首次 `pip install torch` 失败 `OSError: [Errno 28] No space left on device`——根因不是磁盘满（overlay 还有 1.5T），而是 **pip 默认把 4.47GB wheel 下载/解压到 `/tmp`，而 `/tmp` 是 4GB tmpfs**。解法：`export TMPDIR=/workspace/piptmp`（/workspace 是 98G 真盘，79G 可用）+ `--no-cache-dir`。这个坑记入 MEMORY 候选。

## 2. 最小 bit-flip 编辑探针（`min_edit_probe.py`，实测）

- 模型：**云上现成的 Qwen2.5-0.5B**（`{MODEL_DIR}/Qwen2.5-0.5B`，HF/safetensors 格式，494,032,768 参数）——**不是** E1 设计里的 Qwen3.5-0.8B，不能混写。
- 操作：把 `model.layers[0].self_attn.q_proj.weight` 视为 int16，翻转前 4096 个元素的 mantissa LSB（bit 0）。
- 实测改动比例：**4096 / 494032768 = 0.000829%**（本探针实际值，不是论文里的 0.016%）。
- prompt：`Question: What is 15 times 17? Let me think step by step.\nAnswer:`
- 结果：
  - base: `" 15 times 17 is 255. The answer is 255."`
  - edited: `" 15 times 17 is 255. The answer is 255."`
  - `behavior_changed: false`

**结论（阴性，但有价值）**：随机翻转 q_proj 的 4096 个 mantissa LSB 在该题上不改变输出。这符合"定向编辑才有效"的叙事——不是随便翻几位就能改行为。整条 pipeline（加载/编辑/生成/对比/落盘）在 AMD 云上跑通，E1 的技术底座成立。

## 3. 事实纪律

1. 0.000829% 与 0.016% 是两回事：前者是本探针翻位数占比，后者是论文声称的编辑比例，引用时分开。
2. 本探针用的是 Qwen2.5-0.5B；E1 目标模型 Qwen3.5-0.8B 尚未在云上就位（云上另有 Qwen2.5-1.5B/7B）。要严格复现 E1 需先备齐目标权重。
3. 该阴性结果是"单次单题"观察，不足以支撑任何能力结论；只证明 pipeline 可用 + 随机扰动不足以改行为。

## 4. 复现命令

```bash
ssh -p <REMOTE_PORT> root@<REMOTE_HOST>
export HSA_OVERRIDE_GFX_VERSION=11.0.0
python3 {OUT_DIR}/min_edit.py   # 脚本源：experiments/E1-ood-transfer/min_edit_probe.py
```
