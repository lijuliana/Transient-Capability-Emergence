"""
Compute attention entropy for the critical head on end_of_sentence probes
at peak and final checkpoints, for both seeds.

Also computes per-head attention patterns on a sample of probe items
to support the "stable attention, changed OV" claim.
"""
import os, sys, glob, re, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

sys.path.insert(0, '.')
from patching import _get_train_ns, load_checkpoint, best_peak_checkpoint
from prepare import Tokenizer

PROBE_LOG = {99: "probe_log_seed99.tsv", 17: "probe_log_seed17.tsv"}
CRITICAL_HEADS = {17: (0, 0), 99: (0, 1)}


def select_checkpoints(seed, device):
    paths = sorted(glob.glob(f"checkpoints/seed{seed}_step*.pt"))
    paths = [p for p in paths if "step00000" not in p]
    peak_path, peak_step = best_peak_checkpoint(
        [p for p in paths if "_final" not in p], PROBE_LOG[seed], "end_of_sentence")
    finals = [p for p in paths if "_final" in p]
    fine = [p for p in finals if "step02" in p]
    final_path = fine[0] if fine else finals[0]
    final_step = int(re.search(r"step(\d+)", final_path).group(1))
    pm, _, _ = load_checkpoint(peak_path, device)
    fm, _, _ = load_checkpoint(final_path, device)
    return pm, peak_step, fm, final_step


def compute_attn_weights(model, seqs, device, block_idx, head_idx):
    """
    Run model on each seq, capture attention weights for (block_idx, head_idx)
    at the last prefix position (excluding appended token).
    Returns list of attention weight vectors (one per seq).
    """
    ns = _get_train_ns()
    n_embd = model.config.n_embd
    n_head = model.config.n_head
    head_dim = n_embd // n_head

    attn_weights = []

    def make_hook(target_block, target_head):
        def _hook(module, input, output):
            # output from attention is typically (attn_out, weights) or just attn_out
            # We need to hook into a place that gives us the weights
            pass
        return _hook

    # Use a different approach: hook into the QK computation
    captured = {}

    def hook_fn(module, input, output):
        # This hooks post-softmax attention weights if available
        # For SDPA, we won't get weights directly; use manual computation instead
        pass

    model.eval()
    with torch.no_grad():
        for seq in seqs:
            idx = torch.tensor(seq, dtype=torch.long).unsqueeze(0).to(device)
            T = idx.size(1)
            cos_sin = model.cos[:, :T], model.sin[:, :T]
            x = model.transformer.wte(idx)
            x = ns["norm"](x)
            x0 = x.clone()

            for i, block in enumerate(model.transformer.h):
                x = model.resid_lambdas[i] * x + model.x0_lambdas[i] * x0
                ve = (model.value_embeds[str(i)](idx)
                      if str(i) in model.value_embeds else None)

                if i == block_idx:
                    # Manually compute attention weights for this block
                    attn = block.attn
                    B, T2, C = x.shape

                    # Get Q, K from projections
                    if hasattr(attn, 'c_q') and hasattr(attn, 'c_k'):
                        q = attn.c_q(x)   # (B, T, n_embd)
                        k = attn.c_k(x)   # (B, T, n_embd)
                    else:
                        qkv = attn.c_attn(x)
                        q, k, _ = qkv.split(C, dim=-1)

                    # Reshape to (B, n_heads, T, head_dim)
                    q = q.view(B, T2, n_head, head_dim).transpose(1, 2)
                    k = k.view(B, T2, n_head, head_dim).transpose(1, 2)

                    # Apply rotary embeddings: model.cos shape is (1, T, 1, 64)
                    # but q/k are (B, n_heads, T, head_dim) after transpose.
                    # RoPE covers only the first rope_dim=64 of 128 head_dim dims.
                    # Permute cos/sin from (1, T, 1, 64) → (1, 1, T, 64) for broadcast.
                    cos_, sin_ = cos_sin
                    rope_dim = cos_.shape[-1]  # 64
                    # cos_ shape: (1, T, 1, 64) → permute to (1, 1, T, 64)
                    cos_r = cos_.permute(0, 2, 1, 3)  # (1, 1, T, 64)
                    sin_r = sin_.permute(0, 2, 1, 3)

                    def rotate_half(t):
                        h = t.shape[-1] // 2
                        return torch.cat([-t[..., h:], t[..., :h]], dim=-1)

                    q1, q2 = q[..., :rope_dim], q[..., rope_dim:]
                    k1, k2 = k[..., :rope_dim], k[..., rope_dim:]
                    q1_rot = q1 * cos_r + rotate_half(q1) * sin_r
                    k1_rot = k1 * cos_r + rotate_half(k1) * sin_r
                    q = torch.cat([q1_rot, q2], dim=-1)
                    k = torch.cat([k1_rot, k2], dim=-1)

                    # Compute attention scores for target head
                    q_h = q[:, head_idx]  # (B, T, head_dim)
                    k_h = k[:, head_idx]  # (B, T, head_dim)
                    scale = head_dim ** -0.5
                    scores = torch.matmul(q_h, k_h.transpose(-2, -1)) * scale  # (B, T, T)

                    # Causal mask
                    mask = torch.triu(torch.ones(T2, T2, device=device), diagonal=1).bool()
                    scores = scores.masked_fill(mask.unsqueeze(0), float('-inf'))
                    weights = F.softmax(scores, dim=-1)  # (B, T, T)

                    # Get weights at last prefix position (T2-2: before appended token)
                    last_pos = T2 - 2  # position before the appended target token
                    if last_pos >= 0:
                        w = weights[0, last_pos, :last_pos+1].detach().cpu().numpy()
                        attn_weights.append(w)

                x = block(x, ve, cos_sin, model.window_sizes[i])

    return attn_weights


def entropy(w):
    """Shannon entropy of attention weight vector."""
    w = np.clip(w, 1e-10, 1.0)
    return -np.sum(w * np.log(w))


def run_analysis(seed, device):
    print(f"\n=== Seed {seed} / end_of_sentence attention entropy ===")
    block_idx, head_idx = CRITICAL_HEADS[seed]
    head_name = f"B{block_idx}H{head_idx}"

    peak_model, peak_step, final_model, final_step = select_checkpoints(seed, device)

    tok = Tokenizer.from_directory()
    ns = _get_train_ns()
    preps = ns["prepare_probes"](tok)
    items = preps["end_of_sentence"]

    PERIOD_ID = 46
    seqs = [item["prefix_ids"] + [PERIOD_ID] for item in items]
    print(f"  Computing attention for {len(seqs)} items, {head_name}")

    # Peak
    attn_pk = compute_attn_weights(peak_model, seqs, device, block_idx, head_idx)
    # Final
    attn_fn = compute_attn_weights(final_model, seqs, device, block_idx, head_idx)

    if not attn_pk or not attn_fn:
        print("  WARNING: could not extract attention weights (RoPE/architecture mismatch)")
        return None

    entropies_pk = [entropy(w) for w in attn_pk]
    entropies_fn = [entropy(w) for w in attn_fn]

    mean_pk = float(np.mean(entropies_pk))
    mean_fn = float(np.mean(entropies_fn))
    std_pk  = float(np.std(entropies_pk))
    std_fn  = float(np.std(entropies_fn))

    print(f"  {head_name} attention entropy at last prefix position:")
    print(f"    peak  (step {peak_step}): {mean_pk:.3f} ± {std_pk:.3f}")
    print(f"    final (step {final_step}): {mean_fn:.3f} ± {std_fn:.3f}")
    print(f"    Δentropy: {mean_fn - mean_pk:+.3f}")

    # Compute "same argmax position" fraction — measures QK routing stability
    same_argmax = 0
    n_valid = 0
    for w_pk, w_fn in zip(attn_pk, attn_fn):
        min_len = min(len(w_pk), len(w_fn))
        if min_len > 0:
            if np.argmax(w_pk[:min_len]) == np.argmax(w_fn[:min_len]):
                same_argmax += 1
            n_valid += 1
    if n_valid > 0:
        frac = same_argmax / n_valid
        print(f"    {head_name} same argmax position (peak vs final): {same_argmax}/{n_valid} = {frac:.1%}")

    # Also compute for ALL heads at this block
    print(f"\n  All heads at block {block_idx}:")
    for h in range(peak_model.config.n_head):
        ap = compute_attn_weights(peak_model, seqs[:10], device, block_idx, h)
        af = compute_attn_weights(final_model, seqs[:10], device, block_idx, h)
        if ap and af:
            ep = np.mean([entropy(w) for w in ap])
            ef = np.mean([entropy(w) for w in af])
            print(f"    B{block_idx}H{h}: peak={ep:.3f}  final={ef:.3f}  Δ={ef-ep:+.3f}")

    return dict(
        seed=seed, head=head_name,
        peak_step=peak_step, final_step=final_step,
        mean_pk=mean_pk, mean_fn=mean_fn,
        std_pk=std_pk, std_fn=std_fn,
        entropies_pk=entropies_pk, entropies_fn=entropies_fn,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 99])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    for seed in args.seeds:
        r = run_analysis(seed, args.device)
        if r:
            print(f"\nSeed {seed}: entropy peak={r['mean_pk']:.3f}±{r['std_pk']:.3f}  "
                  f"final={r['mean_fn']:.3f}±{r['std_fn']:.3f}  "
                  f"Δ={r['mean_fn']-r['mean_pk']:+.3f}")
    print("Done.")
