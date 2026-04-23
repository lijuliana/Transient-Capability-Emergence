"""
Head-level forward patching and per-item direct logit attribution.

Experiment 1: head output patching (sufficiency test)
  For seed 17, patch B0H0's peak output into the final model and measure
  end_of_sentence accuracy. If accuracy recovers, B0H0 at peak is SUFFICIENT
  (combined with the ablation showing it is necessary).

Experiment 2: per-item direct logit attribution
  For each end_of_sentence probe item, extract what B0H0 writes to the
  residual stream at the critical position, project through lm_head, and
  compute the 'correct' vs 'distractor' logit difference attributed to B0H0.
  Compare peak vs final models.

Usage:
    python head_patching.py [--seeds 17 99]
"""

import os
import sys
import glob
import re
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from patching import (
    load_checkpoint, best_peak_checkpoint, load_prepared_probes,
    build_probe_batch, probe_acc_from_logits_batch, _get_train_ns,
)

PROBE_LOG = {
    99: "probe_log_seed99.tsv",
    17: "probe_log_seed17.tsv",
}
CRITICAL_HEADS = {
    17: (0, 0),  # B0H0
    99: (0, 1),  # B0H1
}


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint selection
# ──────────────────────────────────────────────────────────────────────────────

def select_checkpoints(seed, probe_name, device):
    ckpt_dir = "checkpoints"
    ckpt_paths = sorted(glob.glob(f"{ckpt_dir}/seed{seed}_step*.pt"))
    ckpt_paths = [p for p in ckpt_paths if "step00000" not in p]
    peak_path, peak_step = best_peak_checkpoint(
        [p for p in ckpt_paths if "_final" not in p], PROBE_LOG[seed], probe_name)
    final_paths = sorted([p for p in ckpt_paths if "_final" in p])
    fine_finals = [p for p in final_paths if "step02" in p]
    final_path = fine_finals[0] if fine_finals else final_paths[0]
    final_step = int(re.search(r"step(\d+)", final_path).group(1))
    peak_model, _, _ = load_checkpoint(peak_path, device)
    final_model, _, _ = load_checkpoint(final_path, device)
    print(f"  peak={peak_step}, final={final_step}")
    return peak_model, peak_step, final_model, final_step


# ──────────────────────────────────────────────────────────────────────────────
# Head output extraction / injection hooks
# ──────────────────────────────────────────────────────────────────────────────

def extract_head_output(model, block_idx, head_idx, probe_seqs, device):
    """
    Run model forward; return list of tensors of shape (T,) where each
    value is B{block}H{head}'s scalar dot-product contribution at each position.
    Actually: return the full head output vector (T, n_embd) and lm_head logits.
    """
    n_embd = model.config.n_embd
    n_head = model.config.n_head
    head_dim = n_embd // n_head
    attn = model.transformer.h[block_idx].attn

    stored = {}

    def _pre_hook(module, inp):
        x = inp[0]
        # x is (B, T, n_embd) — the concatenated head outputs fed to c_proj
        # Extract head h slice: channels [h*head_dim : (h+1)*head_dim]
        h_start = head_idx * head_dim
        h_end = (head_idx + 1) * head_dim
        stored["head_in"] = x[:, :, h_start:h_end].detach().clone()  # (B, T, head_dim)
        return None  # don't modify

    handle = attn.c_proj.register_forward_pre_hook(_pre_hook)

    ns = _get_train_ns()
    norm_fn = ns["norm"]
    model.eval()
    results = []

    with torch.no_grad():
        for seq_raw in probe_seqs:
            stored.clear()
            idx = torch.tensor(seq_raw, dtype=torch.long).unsqueeze(0).to(device)
            T = idx.size(1)
            cos_sin = model.cos[:, :T], model.sin[:, :T]
            x = model.transformer.wte(idx)
            x = norm_fn(x)
            x0 = x.clone()
            for i, block in enumerate(model.transformer.h):
                x = model.resid_lambdas[i] * x + model.x0_lambdas[i] * x0
                ve = (model.value_embeds[str(i)](idx)
                      if str(i) in model.value_embeds else None)
                x = block(x, ve, cos_sin, model.window_sizes[i])
            x = norm_fn(x)
            softcap = 15
            logits = model.lm_head(x).float()
            logits = softcap * torch.tanh(logits / softcap)
            # head_in shape: (1, T, head_dim) from hook
            results.append({
                "logits": logits.squeeze(0),  # (T, V)
                "head_in": stored["head_in"].squeeze(0),  # (T, head_dim)
            })

    handle.remove()
    return results


def patch_head_output_forward(peak_model, final_model, block_idx, head_idx,
                              probe_seqs, device):
    """
    Run final model with B{block}H{head}'s c_proj input replaced by the
    peak model's B{block}H{head} c_proj input (same item).
    Returns logits_batch for probe accuracy evaluation.
    """
    n_embd = final_model.config.n_embd
    n_head = final_model.config.n_head
    head_dim = n_embd // n_head
    h_start = head_idx * head_dim
    h_end = (head_idx + 1) * head_dim

    ns = _get_train_ns()
    norm_fn_peak = ns["norm"]  # shared; both models use same norm variant
    norm_fn_final = ns["norm"]

    peak_model.eval()
    final_model.eval()

    logits_batch = []
    with torch.no_grad():
        for seq_raw in probe_seqs:
            idx = torch.tensor(seq_raw, dtype=torch.long).unsqueeze(0).to(device)
            T = idx.size(1)

            # Run peak model to extract head output at c_proj input
            peak_stored = {}
            attn_peak = peak_model.transformer.h[block_idx].attn

            def _peak_hook(module, inp):
                peak_stored["x"] = inp[0].detach().clone()
            h_pk = attn_peak.c_proj.register_forward_pre_hook(_peak_hook)
            try:
                cos_sin = peak_model.cos[:, :T], peak_model.sin[:, :T]
                xp = peak_model.transformer.wte(idx)
                xp = norm_fn_peak(xp)
                x0p = xp.clone()
                for i, block in enumerate(peak_model.transformer.h):
                    xp = peak_model.resid_lambdas[i] * xp + peak_model.x0_lambdas[i] * x0p
                    ve = (peak_model.value_embeds[str(i)](idx)
                          if str(i) in peak_model.value_embeds else None)
                    xp = block(xp, ve, cos_sin, peak_model.window_sizes[i])
            finally:
                h_pk.remove()
            peak_head_slice = peak_stored["x"][:, :, h_start:h_end]  # (1,T,head_dim)

            # Run final model with B0H0 slice replaced from peak
            def _inject_hook(module, inp):
                x = inp[0].clone()
                x[:, :, h_start:h_end] = peak_head_slice
                return (x,)

            attn_final = final_model.transformer.h[block_idx].attn
            h_inj = attn_final.c_proj.register_forward_pre_hook(_inject_hook)
            try:
                cos_sin = final_model.cos[:, :T], final_model.sin[:, :T]
                xf = final_model.transformer.wte(idx)
                xf = norm_fn_final(xf)
                x0f = xf.clone()
                for i, block in enumerate(final_model.transformer.h):
                    xf = final_model.resid_lambdas[i] * xf + final_model.x0_lambdas[i] * x0f
                    ve = (final_model.value_embeds[str(i)](idx)
                          if str(i) in final_model.value_embeds else None)
                    xf = block(xf, ve, cos_sin, final_model.window_sizes[i])
                xf = norm_fn_final(xf)
                softcap = 15
                logits = final_model.lm_head(xf).float()
                logits = softcap * torch.tanh(logits / softcap)
                logits_batch.append(logits.squeeze(0))
            finally:
                h_inj.remove()

    return logits_batch


# ──────────────────────────────────────────────────────────────────────────────
# Experiment 2: per-item direct logit attribution
# ──────────────────────────────────────────────────────────────────────────────

def direct_logit_attribution(results_list, probe_meta, block_idx, head_idx,
                              model, device):
    """
    For each probe item, compute the logit difference (correct - distractor)
    attributed directly to B0H0 via the DLA formula:
      DLA_h = lm_head(W_O_h @ head_in_h) at the critical position
    where critical position = the position BEFORE the predicted token
    (last position in the prefix).

    Returns array of per-item (correct_logit - distractor_logit) from B0H0 alone.
    """
    n_embd = model.config.n_embd
    n_head = model.config.n_head
    head_dim = n_embd // n_head
    attn = model.transformer.h[block_idx].attn
    w_o_full = attn.c_proj.weight.detach().float()  # (n_embd, n_embd)
    w_o_h = w_o_full[:, head_idx * head_dim:(head_idx + 1) * head_dim]  # (n_embd, head_dim)
    lm_head_w = model.lm_head.weight.detach().float()  # (vocab, n_embd)

    # Precompute: DLA projection matrix for this head
    # dla_proj: (vocab, head_dim) — maps head_in to logit over vocab
    dla_proj = lm_head_w @ w_o_h  # (vocab, head_dim)

    diffs = []
    for i, result in enumerate(results_list):
        head_in = result["head_in"].float()  # (T, head_dim) on device
        meta = probe_meta[i]
        correct_id = meta.get("correct_token_id")
        distractor_id = meta.get("distractor_token_id")
        prefix_len = meta.get("prefix_len")

        if correct_id is None or distractor_id is None or prefix_len is None:
            diffs.append(float("nan"))
            continue

        # Critical position: last position of prefix (0-indexed = prefix_len - 1)
        pos = min(prefix_len - 1, head_in.shape[0] - 1)
        h = head_in[pos, :].to("cpu")  # (head_dim,)

        # Logit at correct vs distractor token from this head alone
        logit_correct = (dla_proj[correct_id] @ h).item()
        logit_distractor = (dla_proj[distractor_id] @ h).item()
        diffs.append(logit_correct - logit_distractor)

    return np.array(diffs)


def get_token_ids_from_meta(prepared_probes, probe_name, tokenizer):
    """
    For each probe item, get the correct_token_id, distractor_token_id,
    and prefix_len. Returns list of dicts.
    """
    probe = None
    for p in prepared_probes:
        if hasattr(p, "name") and p.name == probe_name:
            probe = p
            break
        elif isinstance(p, dict) and p.get("name") == probe_name:
            probe = p
            break

    if probe is None:
        return None

    # Try to extract items
    items = getattr(probe, "items", None) or (probe.get("items") if isinstance(probe, dict) else None)
    if items is None:
        return None

    meta_list = []
    for item in items:
        if hasattr(item, "prefix"):
            prefix = item.prefix
            correct = item.correct
            distractor = item.distractor
        elif isinstance(item, dict):
            prefix = item.get("prefix", "")
            correct = item.get("correct", "")
            distractor = item.get("distractor", "")
        else:
            meta_list.append({})
            continue

        prefix_ids = tokenizer.encode(prefix)
        correct_ids = tokenizer.encode(correct)
        distractor_ids = tokenizer.encode(distractor)

        meta_list.append({
            "prefix_len": len(prefix_ids),
            "correct_token_id": correct_ids[0] if correct_ids else None,
            "distractor_token_id": distractor_ids[0] if distractor_ids else None,
        })
    return meta_list


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def run_experiments(seed, device):
    print(f"\n=== Seed {seed} ===")
    block_idx, head_idx = CRITICAL_HEADS[seed]
    head_name = f"B{block_idx}H{head_idx}"

    peak_model, peak_step, final_model, final_step = select_checkpoints(
        seed, "end_of_sentence", device)

    ns = _get_train_ns()
    prepared = ns["prepare_probes"]
    from prepare import Tokenizer
    tok = Tokenizer.from_directory()
    prepared_probes = prepared(tok)

    seqs, meta = build_probe_batch(prepared_probes, ["end_of_sentence"])
    n_items = len(seqs)
    print(f"  {n_items} probe sequences")

    # ── Experiment 1: head output patching (sufficiency) ──
    print(f"\n  [Exp 1] Head output patching: inject peak {head_name} into final model")
    # Baseline accs
    from head_ablation import run_model_on_batch, baseline_acc
    from patching import probe_acc_from_logits_batch
    base_peak = baseline_acc(peak_model, seqs, meta, device).get("end_of_sentence", float("nan"))
    base_final = baseline_acc(final_model, seqs, meta, device).get("end_of_sentence", float("nan"))
    print(f"    baseline peak={base_peak:.3f}  final={base_final:.3f}")

    patched_logits = patch_head_output_forward(peak_model, final_model,
                                               block_idx, head_idx, seqs, device)
    patched_acc = probe_acc_from_logits_batch(patched_logits, meta).get("end_of_sentence", float("nan"))
    print(f"    patched final (peak {head_name} injected): {patched_acc:.3f}")
    delta = patched_acc - base_final
    print(f"    Δacc (patched − final baseline): {delta:+.3f}")

    # ── Experiment 2: per-item direct logit attribution ──
    print(f"\n  [Exp 2] Direct logit attribution from {head_name}")

    # Get probe item meta (token ids, prefix lengths)
    token_meta = get_token_ids_from_meta(prepared_probes, "end_of_sentence", tok)
    if token_meta is None:
        print("    Could not extract token meta from probe; skipping DLA.")
        dla_results = None
    else:
        # Extract head outputs during full forward pass
        peak_results = extract_head_output(peak_model, block_idx, head_idx, seqs, device)
        final_results = extract_head_output(final_model, block_idx, head_idx, seqs, device)

        dla_peak = direct_logit_attribution(
            peak_results, token_meta, block_idx, head_idx, peak_model, device)
        dla_final = direct_logit_attribution(
            final_results, token_meta, block_idx, head_idx, final_model, device)

        valid = ~(np.isnan(dla_peak) | np.isnan(dla_final))
        print(f"    DLA valid items: {valid.sum()}/{len(valid)}")
        if valid.sum() > 0:
            print(f"    Peak model {head_name} mean DLA logit diff (correct−distractor): "
                  f"{np.mean(dla_peak[valid]):.3f} ± {np.std(dla_peak[valid]):.3f}")
            print(f"    Final model {head_name} mean DLA logit diff: "
                  f"{np.mean(dla_final[valid]):.3f} ± {np.std(dla_final[valid]):.3f}")
            frac_pos_peak = (dla_peak[valid] > 0).mean()
            frac_pos_final = (dla_final[valid] > 0).mean()
            print(f"    Peak: {frac_pos_peak:.1%} items where {head_name} favours correct token")
            print(f"    Final: {frac_pos_final:.1%} items where {head_name} favours correct token")
        dla_results = {"peak": dla_peak, "final": dla_final, "valid": valid}

    return {
        "seed": seed,
        "head": head_name,
        "base_peak": base_peak,
        "base_final": base_final,
        "patched_acc": patched_acc,
        "delta_patch": delta,
        "dla": dla_results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 99])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    all_results = {}
    for seed in args.seeds:
        all_results[seed] = run_experiments(seed, args.device)

    print("\n\n=== Summary ===")
    for seed, r in sorted(all_results.items()):
        print(f"Seed {seed} ({r['head']}):")
        print(f"  baseline: peak={r['base_peak']:.3f}  final={r['base_final']:.3f}")
        print(f"  head-output-patched final = {r['patched_acc']:.3f}  (Δ={r['delta_patch']:+.3f})")
        if r["dla"] is not None and r["dla"]["valid"].sum() > 0:
            v = r["dla"]["valid"]
            dp = r["dla"]["peak"][v]
            df = r["dla"]["final"][v]
            print(f"  DLA peak: mean={np.mean(dp):.3f}  frac_pos={np.mean(dp>0):.1%}")
            print(f"  DLA final: mean={np.mean(df):.3f}  frac_pos={np.mean(df>0):.1%}")
    print("Done.")
