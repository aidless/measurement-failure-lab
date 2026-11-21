#!/usr/bin/env python3
"""符号判决探针：验证 mantissa LSB 编辑方向是否与梯度下降一致。

背景（算术上已确定，本脚本做实测确认）：
    bf16 用 int16 视图看时，**幅值增大的方向与符号位无关**：
        w=+1.0  ++1 → +1.007812（幅值↑）   --1 → +0.996094（幅值↓）
        w=-1.0  ++1 → -1.007812（幅值↑）   --1 → -0.996094（幅值↓）
    而梯度下降要的是 w_new = w - lr·g，对负权重而言"g 为正时幅值应增大"。
    因此 `pi -= sign(g)*ulps` 只在 w>0 时方向正确；w<0 时方向相反（相当于沿 +g 走）。
    正确规则：`pi -= sign(g)*sign(w)*ulps`。

本脚本在同一进程内做三次读数（同一批题、同一位置集、同一 ulps）：
    base → 施加【修正后方向】→ 测 → 撤销 → 施加【原方向】→ 测 → 撤销
若修正方向的 loss 下降而原方向上升，则符号 bug 成立。
"""
import argparse, json, os, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cloud_e1_run import build_prompt  # noqa
from e1_logit_probe import target_text  # noqa


def loss_of(model, tok, it, mode):
    prompt = build_prompt(tok, it["question"], mode)
    pids = tok(prompt, return_tensors="pt").to(model.device)
    tids = tok(" " + target_text(it), return_tensors="pt", add_special_tokens=False).to(model.device)
    full = torch.cat([pids["input_ids"], tids["input_ids"]], dim=1)
    labels = full.clone(); labels[:, :pids["input_ids"].shape[1]] = -100
    with torch.inference_mode():
        l = float(model(input_ids=full, labels=labels).loss)
    del pids, tids, full, labels
    torch.cuda.empty_cache()
    return l


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--ulps", type=int, default=3)
    ap.add_argument("--frac", type=float, default=8e-5); ap.add_argument("--n-grad", type=int, default=3)
    ap.add_argument("--max-len", type=int, default=96)
    a = ap.parse_args()
    os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
    items = [json.loads(l) for l in open(a.items, encoding="utf-8") if l.strip()]
    d1 = [it for it in items if it["family"] == "D1"]
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16,
                                                 device_map="cuda:0", trust_remote_code=True)
    model.eval()
    if model.config is not None: model.config.use_cache = True
    base = {it["id"]: loss_of(model, tok, it, "plain") for it in d1}
    print("[base]", json.dumps({k: round(v, 4) for k, v in base.items()}))

    params = [(n, p) for n, p in model.named_parameters()
              if p.dtype == torch.bfloat16 and p.dim() >= 2
              and "embed_tokens" not in n and "lm_head" not in n]
    total = sum(p.numel() for _, p in params)
    model.train()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.zero_grad(set_to_none=True)
    sel = {}
    def mk(n):
        def h(p):
            g = p.grad
            if g is None: return
            kk = max(1, int(p.numel() * a.frac / a.n_grad))
            gf = g.detach().float().cpu().abs().flatten()
            _, idx = torch.topk(gf, kk)
            sv = torch.sign(g.detach().float().cpu().flatten()[idx]).to(torch.int8)
            sel.setdefault(n, []).append((idx, sv))
            p.grad = None
        return h
    hs = [p.register_post_accumulate_grad_hook(mk(n)) for n, p in params]
    for it in d1[: a.n_grad]:
        prompt = build_prompt(tok, it["question"], "plain")
        pids = tok(prompt, return_tensors="pt").to(model.device)
        tids = tok(" " + target_text(it), return_tensors="pt", add_special_tokens=False).to(model.device)
        cut = pids["input_ids"][:, -a.max_len:]
        full = torch.cat([cut, tids["input_ids"]], dim=1)
        labels = full.clone(); labels[:, :cut.shape[1]] = -100
        l = model(input_ids=full, labels=labels).loss
        (l / a.n_grad).backward()
        print(f"  [grad] {it['id']} {float(l):.4f}", file=sys.stderr)
        del pids, tids, cut, full, labels, l
        torch.cuda.empty_cache()
    for h in hs: h.remove()

    plan = {}
    for n, chunks in sel.items():
        idx = torch.cat([c[0] for c in chunks]); sv = torch.cat([c[1] for c in chunks])
        uniq, inv = torch.unique(idx, return_inverse=True)
        last = torch.zeros(uniq.numel(), dtype=torch.int8); last[inv] = sv
        plan[n] = (uniq, last)

    def apply(mode, ulps):
        cnt = 0
        with torch.no_grad():
            for n, p in params:
                if n not in plan: continue
                u, s = plan[n]
                pi = p.view(torch.int16).flatten()
                wsign = torch.sign(p.detach().float().flatten()[u.to(p.device)]).cpu().to(torch.int8)
                step = s * wsign if mode == "fixed" else s
                seg = u.to(p.device)
                pi[seg] = pi[seg] - step.to(p.device).to(torch.int16) * ulps
                cnt += int(seg.numel())
                del pi, seg, wsign, step
        torch.cuda.empty_cache(); return cnt

    model.eval(); model.config.use_cache = True
    res = {}
    for mode in ("fixed", "original"):
        n_ed = apply(mode, a.ulps)
        m = {it["id"]: loss_of(model, tok, it, "plain") for it in d1}
        apply(mode, -a.ulps)  # 反向撤销，回到基线
        dmean = sum(m[k] - base[k] for k in base) / len(base)
        res[mode] = {"edited": n_ed, "delta_loss_mean": round(dmean, 6),
                     "rows": {k: round(m[k], 4) for k in m},
                     "n_worsened": sum(1 for k in base if m[k] > base[k])}
        print(f"[{mode}] edited={n_ed} delta_loss_mean={dmean:+.4f} worsened={res[mode]['n_worsened']}/15",
              file=sys.stderr)

    json.dump({"model": a.model, "ulps": a.ulps, "frac": a.frac, "n_grad": a.n_grad,
               "base": base, "result": res}, open(a.out, "w"), ensure_ascii=False, indent=2)
    print("[done]", a.out)

if __name__ == "__main__":
    main()
