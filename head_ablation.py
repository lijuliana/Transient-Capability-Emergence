"""
Head-level ablation analysis for transient-probe mechanistic investigation.

For each of the 8 attention heads (4 blocks × 2 heads) in the peak and final
checkpoints, zero out that head's contribution to the residual stream and
measure the resulting change in probe accuracy (ΔAcc = ablated − baseline).

A large negative ΔAcc means the head is *critical* for the probe at that
checkpoint. Comparing peak vs. final reveals which specific heads changed
their functional role during the transient collapse.

Usage:
    python head_ablation.py [--seeds 99 17] [--probes end_of_sentence ...]
"""

import os
import sys
import argparse
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# Reuse infrastructure from patching.py
from patching import (
    load_checkpoint, build_probe_batch, load_prepared_probes,
    probe_acc_from_logits_batch, _get_train_ns, best_peak_checkpoint,
    nearest_checkpoint,
)

# Probe log paths per seed
PROBE_LOG = {
    99: "probe_log_seed99.tsv",
    17: "probe_log_seed17.tsv",
}

# Which checkpoints to use (same as patching.py fine-checkpoint runs)
# Format: {seed: {"final": path, "peak": {probe: path}}}
# We determine these programmatically via best_peak_checkpoint.

TRANSIENT_PROBES = ["end_of_sentence", "modal_continuation", "adjective_order"]

# ---------------------------------------------------------------------------
# Head ablation via pre-forward hook on c_proj
# ---------------------------------------------------------------------------

def ablate_head_forward(model, probe_seqs, probe_meta, block_idx, head_idx, device="cpu"):
    """
    Run a full model forward pass with head (block_idx, head_idx) zeroed out
    at the c_proj input. Returns probe accuracy dict.
    """
    head_dim = model.config.n_embd // model.config.n_head
    attn = model.transformer.h[block_idx].attn

    def _hook(module, inp):
        x = inp[0].clone()
        start = head_idx * head_dim
        end = (head_idx + 1) * head_dim
        x[..., start:end] = 0.0
        return (x,)

    handle = attn.c_proj.register_forward_pre_hook(_hook)
    try:
        logits_batch = run_model_on_batch(model, probe_seqs, device)
        acc = probe_acc_from_logits_batch(logits_batch, probe_meta)
    finally:
        handle.remove()
    return acc


def run_model_on_batch(model, probe_seqs, device="cpu"):
    """
    Run model forward on each sequence in probe_seqs; return list of per-token
    logit tensors matching the order expected by probe_acc_from_logits_batch.
    """
    model.eval()
    ns = _get_train_ns()
    norm_fn = ns["norm"]
    n_layer = model.config.n_layer
    logits_batch = []
    with torch.no_grad():
        for seq_raw in probe_seqs:
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
            logits_batch.append(logits.squeeze(0))
    return logits_batch


def baseline_acc(model, probe_seqs, probe_meta, device="cpu"):
    """Run model with no ablation; return acc dict."""
    logits_batch = run_model_on_batch(model, probe_seqs, device)
    return probe_acc_from_logits_batch(logits_batch, probe_meta)


# ---------------------------------------------------------------------------
# Checkpoint selection helpers (same logic as patching.py fine runs)
# ---------------------------------------------------------------------------

def select_checkpoints(seed, probe_name, device="cpu"):
    """
    Return (peak_model, peak_step, final_model, final_step) using the same
    same-trajectory selection as patching.py.
    """
    import glob
    ckpt_dir = "checkpoints"
    probe_log = PROBE_LOG[seed]

    # All checkpoints for this seed
    ckpt_paths = sorted(glob.glob(f"{ckpt_dir}/seed{seed}_step*.pt"))
    # Exclude step 0
    ckpt_paths = [p for p in ckpt_paths if "step00000" not in p]

    # Peak: best measured probe accuracy
    peak_path, peak_step = best_peak_checkpoint(ckpt_paths, probe_log, probe_name)

    # Final: same trajectory → use the *_final.pt for the run that produced peak
    # We pick the shortest final (fine-checkpoint run) to match patching.py
    final_paths = sorted([p for p in ckpt_paths if "_final" in p])
    # Use the final from the fine-checkpoint run (lower step number)
    fine_finals = [p for p in final_paths if "step02" in p]  # step ~2200-2256
    final_path = fine_finals[0] if fine_finals else final_paths[0]
    import re
    final_step = int(re.search(r"step(\d+)", final_path).group(1))

    peak_model, _, _ = load_checkpoint(peak_path, device)
    final_model, _, _ = load_checkpoint(final_path, device)
    print(f"  seed {seed} probe={probe_name}: peak step={peak_step} final step={final_step}")
    return peak_model, peak_step, final_model, final_step


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_ablation(seed, probe_name, device="cpu"):
    """
    Run head ablation for a single seed and probe.
    Returns dict with keys 'peak' and 'final', each a (4, 2) array of delta_acc.
    Also returns (baseline_peak_acc, baseline_final_acc).
    """
    print(f"\n=== Seed {seed}, probe={probe_name} ===")
    peak_model, peak_step, final_model, final_step = select_checkpoints(
        seed, probe_name, device)

    # Build probe batch
    ns = _get_train_ns()
    prepare = ns.get("prepare_probes")
    # Load tokenizer
    from prepare import Tokenizer
    tok = Tokenizer.from_directory()
    prepared = prepare(tok)
    seqs, meta = build_probe_batch(prepared, [probe_name])
    print(f"  {len(seqs)} probe sequences loaded")

    n_layer = peak_model.config.n_layer  # 4
    n_head = peak_model.config.n_head    # 2

    # Baselines
    base_peak = baseline_acc(peak_model, seqs, meta, device).get(probe_name, float("nan"))
    base_final = baseline_acc(final_model, seqs, meta, device).get(probe_name, float("nan"))
    print(f"  baseline: peak={base_peak:.3f}  final={base_final:.3f}")

    peak_delta = np.zeros((n_layer, n_head))
    final_delta = np.zeros((n_layer, n_head))

    for b in range(n_layer):
        for h in range(n_head):
            acc_p = ablate_head_forward(peak_model, seqs, meta, b, h, device)
            acc_f = ablate_head_forward(final_model, seqs, meta, b, h, device)
            dp = acc_p.get(probe_name, float("nan")) - base_peak
            df = acc_f.get(probe_name, float("nan")) - base_final
            peak_delta[b, h] = dp
            final_delta[b, h] = df
            print(f"  block={b} head={h}: peak Δ={dp:+.3f}  final Δ={df:+.3f}")

    return {
        "peak_delta": peak_delta,
        "final_delta": final_delta,
        "base_peak": base_peak,
        "base_final": base_final,
        "peak_step": peak_step,
        "final_step": final_step,
    }


def make_ablation_figure(results_by_seed, probe_name):
    """
    3-row × N-seed figure:
      Row 0: peak ΔAcc
      Row 1: final ΔAcc
      Row 2: Δ(Δ) = final_delta − peak_delta (positive = head lost importance)
    """
    seeds = sorted(results_by_seed.keys())
    fig, axes = plt.subplots(3, len(seeds), figsize=(4.5 * len(seeds), 9),
                             squeeze=False)
    vmax_ab = 0.0
    for r in results_by_seed.values():
        vmax_ab = max(vmax_ab,
                      np.nanmax(np.abs(r["peak_delta"])),
                      np.nanmax(np.abs(r["final_delta"])))
    vmax_ab = max(vmax_ab, 0.10)
    vmax_dd = vmax_ab  # same scale for difference

    for col, seed in enumerate(seeds):
        r = results_by_seed[seed]
        diff = r["final_delta"] - r["peak_delta"]  # Δ(Δ): positive = lost importance
        rows_cfg = [
            (r["peak_delta"],  vmax_ab,
             f"seed {seed} PEAK (step {r['peak_step']}, acc={r['base_peak']:.3f})\nΔAcc = ablated−baseline"),
            (r["final_delta"], vmax_ab,
             f"seed {seed} FINAL (step {r['final_step']}, acc={r['base_final']:.3f})\nΔAcc = ablated−baseline"),
            (diff, vmax_dd,
             f"seed {seed} Δ(Δ) = final−peak\n(+ve = head lost importance)"),
        ]
        for row, (data, vmax, title) in enumerate(rows_cfg):
            ax = axes[row][col]
            cmap = "RdBu" if row < 2 else "PuOr"
            im = ax.imshow(data, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["H0", "H1"], fontsize=9)
            ax.set_yticks(range(4))
            ax.set_yticklabels([f"B{b}" for b in range(4)], fontsize=9)
            ax.set_title(title, fontsize=8)
            ax.set_xlabel("head", fontsize=8)
            if col == 0:
                ax.set_ylabel("block", fontsize=8)
            for b in range(data.shape[0]):
                for h in range(data.shape[1]):
                    v = data[b, h]
                    ax.text(h, b, f"{v:+.2f}", ha="center", va="center",
                            fontsize=9, fontweight="bold" if abs(v) >= 0.15 else "normal",
                            color="white" if abs(v) > vmax * 0.55 else "black")
            plt.colorbar(im, ax=ax, fraction=0.05, pad=0.04)

    fig.suptitle(
        f"Head ablation — {probe_name.replace('_', ' ')}\n"
        "Red = ablating hurts accuracy; blue = ablating helps. "
        "Row 3 (Δ(Δ)): orange = head lost importance peak→final.",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs("figures", exist_ok=True)
    out = f"figures/fig_ablation_{probe_name}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")
    return out


def write_results_md(all_results, filename="head_ablation_results.md"):
    """Write a summary markdown table of ablation results."""
    lines = ["# Head Ablation Results\n",
             "ΔAcc = ablated_accuracy − baseline_accuracy. "
             "Negative = head is important for this probe.\n"]
    for probe_name, results_by_seed in all_results.items():
        lines.append(f"\n## {probe_name}\n")
        seeds = sorted(results_by_seed.keys())
        for seed in seeds:
            r = results_by_seed[seed]
            lines.append(f"\n### Seed {seed}\n")
            lines.append(f"- Peak checkpoint: step {r['peak_step']}, "
                         f"baseline acc = {r['base_peak']:.3f}\n")
            lines.append(f"- Final checkpoint: step {r['final_step']}, "
                         f"baseline acc = {r['base_final']:.3f}\n\n")
            lines.append("| block | head | peak ΔAcc | final ΔAcc | Δ(Δ) |\n")
            lines.append("|---|---|---|---|---|\n")
            n_layer, n_head = r["peak_delta"].shape
            for b in range(n_layer):
                for h in range(n_head):
                    dp = r["peak_delta"][b, h]
                    df = r["final_delta"][b, h]
                    dd = df - dp  # positive = head became less important; negative = more
                    lines.append(f"| {b} | {h} | {dp:+.3f} | {df:+.3f} | {dd:+.3f} |\n")
    with open(filename, "w") as f:
        f.writelines(lines)
    print(f"Wrote {filename}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[99, 17])
    parser.add_argument("--probes", nargs="+", default=["end_of_sentence"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Device: {args.device}")
    all_results = {}

    for probe_name in args.probes:
        results_by_seed = {}
        for seed in args.seeds:
            res = run_ablation(seed, probe_name, args.device)
            results_by_seed[seed] = res
        all_results[probe_name] = results_by_seed
        make_ablation_figure(results_by_seed, probe_name)

    write_results_md(all_results)
    print("\nDone.")
