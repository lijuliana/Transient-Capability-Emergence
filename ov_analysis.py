"""
OV circuit analysis for transient-probe mechanistic investigation.

For each seed and each attention head, compute:
1. OV cosine similarity between peak and final checkpoints
2. OV Frobenius norms at peak and final
3. Top-10 tokens promoted by the OV matrix at peak (logit lens)

Usage:
    python ov_analysis.py [--seeds 99 17]
"""

import os
import sys
import argparse
import glob
import re

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from patching import load_checkpoint, best_peak_checkpoint, _get_train_ns

PROBE_LOG = {
    99: "probe_log_seed99.tsv",
    17: "probe_log_seed17.tsv",
}


def get_ov_matrix(model, block_idx, head_idx):
    """Return W_O_h @ W_V_h as a (n_embd, n_embd) matrix."""
    n_embd = model.config.n_embd
    n_head = model.config.n_head
    head_dim = n_embd // n_head

    attn = model.transformer.h[block_idx].attn

    # W_V: (n_embd, n_embd) — the V projection weight; rows are output dims
    # In GPT-NanoX style models, c_attn projects to q, k, v concatenated
    # c_attn.weight shape: (3*n_embd, n_embd) [for full QKV]
    # We extract V part: rows [2*n_embd : 3*n_embd]
    # But some architectures use separate projections. Check:
    if hasattr(attn, 'c_v'):
        # Separate Q/K/V projections
        w_v_full = attn.c_v.weight  # (n_embd, n_embd)
    elif hasattr(attn, 'c_attn'):
        # Combined QKV projection: weight (3*n_embd, n_embd)
        w_qkv = attn.c_attn.weight  # (3*n_embd, n_embd)
        w_v_full = w_qkv[2 * n_embd:3 * n_embd, :]  # (n_embd, n_embd)
    elif hasattr(attn, 'v_proj'):
        w_v_full = attn.v_proj.weight  # (n_embd, n_embd)
    else:
        raise RuntimeError(f"Cannot find V projection in block {block_idx}")

    # W_O: c_proj maps from (n_embd,) to (n_embd,)
    w_o_full = attn.c_proj.weight  # (n_embd, n_embd)

    # Extract head-specific slices
    # W_V_h: (head_dim, n_embd) — maps from residual to value space for head h
    w_v_h = w_v_full[head_idx * head_dim:(head_idx + 1) * head_dim, :]  # (head_dim, n_embd)
    # W_O_h: (n_embd, head_dim) — maps from head h value space to residual
    w_o_h = w_o_full[:, head_idx * head_dim:(head_idx + 1) * head_dim]  # (n_embd, head_dim)

    # OV = W_O_h @ W_V_h: (n_embd, n_embd) — maps input residual to output contribution
    ov = w_o_h @ w_v_h
    return ov.detach().float()


def ov_cosine(ov_peak, ov_final):
    """Cosine similarity of flattened OV matrices."""
    a = ov_peak.flatten()
    b = ov_final.flatten()
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def top_tokens_for_head(model, block_idx, head_idx, tokenizer, n=10, checkpoint_label=""):
    """
    Compute the top tokens that the OV circuit of head (block_idx, head_idx)
    would promote if it wrote to the residual stream at the identity position.
    This is: lm_head(OV) — i.e., project each column of OV through lm_head.
    More precisely, for each input token e_i (standard basis vector), compute
    OV @ e_i to get the output contribution, then project to vocab via lm_head.
    We average over input directions by looking at the top singular direction.
    Actually simpler: take the top-left singular vector of OV, multiply by OV to
    get the dominant output direction, then project to vocab.
    """
    ov = get_ov_matrix(model, block_idx, head_idx)  # (n_embd, n_embd)
    # The OV matrix maps x -> OV @ x. The output direction that is strongest
    # is the top singular vector. But "which tokens does this head promote?"
    # is better answered by: for each output position, what is the logit increase?
    # Using logit lens: look at the output of OV applied to the identity (full embed)
    # A common approach: take the top-n rows of W_E^T @ OV^T (tokens that most
    # align with output). But simplest is: for each vocab token embedding e_v,
    # compute the logit it produces in the output: lm_head(OV @ e_v).item().
    # This is expensive for large vocab but manageable for 8192.
    lm_head_w = model.lm_head.weight.detach().float()  # (vocab, n_embd)
    wte = model.transformer.wte.weight.detach().float()  # (vocab, n_embd)

    # For each input token t, OV maps wte[t] -> output vector
    # Then logit for output token s = lm_head_w[s] @ OV @ wte[t]
    # We want: which output tokens s have the highest average logit increase?
    # Average over all input tokens t:
    # avg_s = mean_t(lm_head_w[s] @ OV @ wte[t])
    #       = lm_head_w[s] @ OV @ mean_t(wte[t])
    mean_wte = wte.mean(0)  # (n_embd,)
    ov_output = ov @ mean_wte  # (n_embd,)
    logits = lm_head_w @ ov_output  # (vocab,)
    top_ids = logits.topk(n).indices.tolist()
    tokens = [tokenizer.decode([i]) for i in top_ids]
    return list(zip(top_ids, tokens, logits[top_ids].tolist()))


def select_checkpoints_for_eos(seed, device="cpu"):
    """Return (peak_model, peak_step, final_model, final_step) for end_of_sentence."""
    probe_log = PROBE_LOG[seed]
    ckpt_dir = "checkpoints"
    ckpt_paths = sorted(glob.glob(f"{ckpt_dir}/seed{seed}_step*.pt"))
    ckpt_paths = [p for p in ckpt_paths if "step00000" not in p]

    peak_path, peak_step = best_peak_checkpoint(ckpt_paths, probe_log, "end_of_sentence")

    final_paths = sorted([p for p in ckpt_paths if "_final" in p])
    fine_finals = [p for p in final_paths if "step02" in p]
    final_path = fine_finals[0] if fine_finals else final_paths[0]
    final_step = int(re.search(r"step(\d+)", final_path).group(1))

    peak_model, _, _ = load_checkpoint(peak_path, device)
    final_model, _, _ = load_checkpoint(final_path, device)
    print(f"  Seed {seed}: peak={peak_step}, final={final_step}")
    return peak_model, peak_step, final_model, final_step


def run_ov_analysis(seed, device="cpu"):
    print(f"\n=== OV circuit analysis — seed {seed} ===")
    peak_model, peak_step, final_model, final_step = select_checkpoints_for_eos(seed, device)

    n_layer = peak_model.config.n_layer
    n_head = peak_model.config.n_head
    n_embd = peak_model.config.n_embd

    rows = []
    for b in range(n_layer):
        for h in range(n_head):
            ov_peak = get_ov_matrix(peak_model, b, h)
            ov_final = get_ov_matrix(final_model, b, h)
            cos = ov_cosine(ov_peak, ov_final)
            norm_peak = ov_peak.norm().item()
            norm_final = ov_final.norm().item()
            rows.append({
                "head": f"B{b}H{h}", "cos": cos,
                "norm_peak": norm_peak, "norm_final": norm_final,
            })
            print(f"  B{b}H{h}: OV cosine={cos:.3f}  norm_peak={norm_peak:.1f}  norm_final={norm_final:.1f}")

    # Top-token analysis for the critical head
    # Load tokenizer
    from prepare import Tokenizer
    tok = Tokenizer.from_directory()
    # Pick critical head (lowest OV cosine in blocks 0-2)
    early_rows = [r for r in rows if int(r["head"][1]) < 3]
    critical = min(early_rows, key=lambda r: r["cos"])
    crit_b = int(critical["head"][1])
    crit_h = int(critical["head"][3])
    print(f"\n  Critical head (lowest cosine in blocks 0-2): {critical['head']} (cos={critical['cos']:.3f})")

    print(f"\n  Top-10 tokens promoted by {critical['head']} OV circuit (PEAK):")
    top_peak = top_tokens_for_head(peak_model, crit_b, crit_h, tok, n=10, checkpoint_label="peak")
    for tid, tok_str, logit in top_peak:
        print(f"    id={tid:5d}  logit={logit:8.3f}  '{repr(tok_str)}'")

    print(f"\n  Top-10 tokens promoted by {critical['head']} OV circuit (FINAL):")
    top_final = top_tokens_for_head(final_model, crit_b, crit_h, tok, n=10, checkpoint_label="final")
    for tid, tok_str, logit in top_final:
        print(f"    id={tid:5d}  logit={logit:8.3f}  '{repr(tok_str)}'")

    return rows, critical, top_peak, top_final


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[99, 17])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Device: {args.device}")
    all_rows = {}
    for seed in args.seeds:
        rows, critical, top_peak, top_final = run_ov_analysis(seed, args.device)
        all_rows[seed] = {"rows": rows, "critical": critical,
                          "top_peak": top_peak, "top_final": top_final}

    # Print summary table
    print("\n\n=== Summary: OV cosine by seed and head ===")
    seeds = sorted(all_rows.keys())
    # Header
    hdr = "head"
    for s in seeds:
        hdr += f"  | seed{s} cos  norm_pk  norm_fn"
    print(hdr)
    # All rows from first seed
    heads = [r["head"] for r in all_rows[seeds[0]]["rows"]]
    for head in heads:
        line = head
        for s in seeds:
            r = next(x for x in all_rows[s]["rows"] if x["head"] == head)
            line += f"  | {r['cos']:.3f}     {r['norm_peak']:6.1f}   {r['norm_final']:6.1f}"
        print(line)

    print("\nDone.")
