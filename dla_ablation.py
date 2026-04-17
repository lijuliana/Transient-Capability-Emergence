"""
Direct logit attribution via ablation for B0H0 / B0H1.

For each end_of_sentence probe item, compute:
  logit_diff(item) = logp('.') - logp(',')
from the full model and from the head-ablated model.

Head attribution = full_logit_diff - ablated_logit_diff
Positive = head is increasing the period advantage over comma.

Run at both peak and final checkpoints.
"""
import os, sys, glob, re, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, '.')
from patching import _get_train_ns, build_probe_batch, load_checkpoint, best_peak_checkpoint
from prepare import Tokenizer

PROBE_LOG = {99: "probe_log_seed99.tsv", 17: "probe_log_seed17.tsv"}
CRITICAL_HEADS = {17: (0, 0), 99: (0, 1)}

PERIOD_ID  = 46  # '.'
COMMA_ID   = 44  # ','


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
    print(f"  Seed {seed}: peak={peak_step} final={final_step}")
    return pm, peak_step, fm, final_step


def run_forward(model, seqs, device, ablate_block=None, ablate_head=None):
    """
    Run model on each sequence; optionally ablate one head.
    Return list of (T, vocab) logit tensors.
    """
    ns = _get_train_ns()
    norm_fn = ns["norm"]
    n_embd = model.config.n_embd
    n_head = model.config.n_head
    head_dim = n_embd // n_head

    handle = None
    if ablate_block is not None and ablate_head is not None:
        attn = model.transformer.h[ablate_block].attn
        h_start = ablate_head * head_dim
        h_end = (ablate_head + 1) * head_dim
        def _hook(module, inp):
            x = inp[0].clone()
            x[..., h_start:h_end] = 0.0
            return (x,)
        handle = attn.c_proj.register_forward_pre_hook(_hook)

    model.eval()
    logits_batch = []
    try:
        with torch.no_grad():
            for seq_raw in seqs:
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
                logits_batch.append(logits.squeeze(0))  # (T, V)
    finally:
        if handle is not None:
            handle.remove()
    return logits_batch


def extract_logit_diffs(logits_batch, probe_items):
    """
    For each item, extract logit_diff = logit[PERIOD] - logit[COMMA]
    at the last prefix position (= len(prefix_ids) - 1).
    Returns (n_items,) array.
    """
    diffs = []
    for i, item in enumerate(probe_items):
        prefix_len = len(item["prefix_ids"])
        pos = prefix_len - 1  # last prefix token position
        logits = logits_batch[i]  # (T, V)
        if pos >= logits.shape[0]:
            diffs.append(float("nan"))
            continue
        ld = (logits[pos, PERIOD_ID] - logits[pos, COMMA_ID]).item()
        diffs.append(ld)
    return np.array(diffs)


def run_analysis(seed, device):
    print(f"\n=== Seed {seed} ===")
    block_idx, head_idx = CRITICAL_HEADS[seed]
    head_name = f"B{block_idx}H{head_idx}"

    peak_model, peak_step, final_model, final_step = select_checkpoints(seed, device)

    tok = Tokenizer.from_directory()
    ns = _get_train_ns()
    preps = ns["prepare_probes"](tok)
    probe_items = preps["end_of_sentence"]

    # Build sequences (need padded seqs for model; use simple version)
    seqs = []
    for item in probe_items:
        # Append period token to make a full sequence (for model forward)
        # but we only look at logits at the last prefix position
        full = item["prefix_ids"] + [PERIOD_ID]
        seqs.append(full)

    print(f"  {len(seqs)} probe items")

    # Run full and ablated forwards
    logits_peak_full    = run_forward(peak_model,  seqs, device)
    logits_peak_ablated = run_forward(peak_model,  seqs, device,
                                      ablate_block=block_idx, ablate_head=head_idx)
    logits_final_full   = run_forward(final_model, seqs, device)
    logits_final_ablated= run_forward(final_model, seqs, device,
                                      ablate_block=block_idx, ablate_head=head_idx)

    # Extract logit diffs at last prefix position
    ld_peak_full    = extract_logit_diffs(logits_peak_full,    probe_items)
    ld_peak_ablated = extract_logit_diffs(logits_peak_ablated, probe_items)
    ld_final_full   = extract_logit_diffs(logits_final_full,   probe_items)
    ld_final_ablated= extract_logit_diffs(logits_final_ablated,probe_items)

    # Head attribution = full - ablated
    attr_peak  = ld_peak_full  - ld_peak_ablated
    attr_final = ld_final_full - ld_final_ablated

    def stats(arr, label):
        valid = ~np.isnan(arr)
        m = np.mean(arr[valid])
        s = np.std(arr[valid])
        pos_frac = np.mean(arr[valid] > 0)
        print(f"  {label}: mean={m:+.3f} ± {s:.3f}  frac_pos={pos_frac:.1%}")
        return m, s, pos_frac

    print(f"\n  Full model logit_diff (period−comma) at last prefix position:")
    stats(ld_peak_full,    f"peak (step {peak_step})")
    stats(ld_final_full,   f"final (step {final_step})")

    print(f"\n  {head_name} attribution = full − ablated logit_diff:")
    m_pk, s_pk, pos_pk = stats(attr_peak,  f"peak attribution")
    m_fn, s_fn, pos_fn = stats(attr_final, f"final attribution")

    # argmax accuracies
    acc_peak_full    = np.mean(ld_peak_full    > 0)
    acc_peak_ablated = np.mean(ld_peak_ablated > 0)
    acc_final_full   = np.mean(ld_final_full   > 0)
    acc_final_ablated= np.mean(ld_final_ablated> 0)
    print(f"\n  Probe accuracy check:")
    print(f"    peak full={acc_peak_full:.3f}  peak ablated={acc_peak_ablated:.3f}  "
          f"ΔAcc={acc_peak_ablated-acc_peak_full:+.3f}")
    print(f"    final full={acc_final_full:.3f}  final ablated={acc_final_ablated:.3f}  "
          f"ΔAcc={acc_final_ablated-acc_final_full:+.3f}")

    return {
        "seed": seed, "head": head_name,
        "peak_step": peak_step, "final_step": final_step,
        "ld_peak_full": ld_peak_full, "ld_final_full": ld_final_full,
        "attr_peak": attr_peak, "attr_final": attr_final,
        "acc_peak": acc_peak_full, "acc_final": acc_final_full,
        "acc_peak_ablated": acc_peak_ablated, "acc_final_ablated": acc_final_ablated,
        "mean_attr_peak": m_pk, "mean_attr_final": m_fn,
        "pos_frac_peak": pos_pk, "pos_frac_final": pos_fn,
    }


def make_figure(results_by_seed):
    seeds = sorted(results_by_seed.keys())
    fig, axes = plt.subplots(2, len(seeds), figsize=(5.5 * len(seeds), 8), squeeze=False)

    for col, seed in enumerate(seeds):
        r = results_by_seed[seed]
        head = r["head"]

        # Row 0: scatter of full model logit_diff (period - comma)
        ax0 = axes[0][col]
        n = len(r["ld_peak_full"])
        jitter = np.random.randn(n) * 0.05
        ax0.scatter(r["ld_peak_full"], r["ld_final_full"], alpha=0.5, s=25, c="steelblue")
        ax0.axhline(0, color="red", linestyle="--", linewidth=0.8)
        ax0.axvline(0, color="red", linestyle="--", linewidth=0.8)
        ax0.set_xlabel(f"Peak logit_diff (period−comma)", fontsize=9)
        ax0.set_ylabel(f"Final logit_diff (period−comma)", fontsize=9)
        ax0.set_title(f"Seed {seed}: per-item logit diff\npeak={r['acc_peak']:.1%} final={r['acc_final']:.1%}", fontsize=9)
        ax0.grid(alpha=0.3)

        # Row 1: head attribution distribution at peak vs final
        ax1 = axes[1][col]
        bins = 30
        valid = ~(np.isnan(r["attr_peak"]) | np.isnan(r["attr_final"]))
        ax1.hist(r["attr_peak"][valid], bins=bins, alpha=0.6, color="C0",
                 label=f"peak (mean={r['mean_attr_peak']:+.3f})")
        ax1.hist(r["attr_final"][valid], bins=bins, alpha=0.6, color="C1",
                 label=f"final (mean={r['mean_attr_final']:+.3f})")
        ax1.axvline(0, color="black", linestyle="--", linewidth=0.8)
        ax1.set_xlabel(f"{head} attribution (full−ablated logit_diff)", fontsize=9)
        ax1.set_ylabel("Count", fontsize=9)
        ax1.set_title(f"Seed {seed}: {head} logit attribution\n"
                      f"peak pos_frac={r['pos_frac_peak']:.1%}, "
                      f"final pos_frac={r['pos_frac_final']:.1%}", fontsize=9)
        ax1.legend(fontsize=8)
        ax1.grid(alpha=0.3)

    fig.suptitle(
        "Per-item logit_diff (period−comma) and head attribution\n"
        "Row 1: scatter of peak vs final logit_diff (both >0 = both correct)\n"
        "Row 2: distribution of head attribution at peak vs final "
        "(positive = head favours period over comma)",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    os.makedirs("figures", exist_ok=True)
    out = "figures/fig_dla_ablation.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nWrote {out}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 99])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    results = {}
    for seed in args.seeds:
        results[seed] = run_analysis(seed, args.device)

    make_figure(results)

    print("\n=== Cross-seed summary ===")
    for seed, r in sorted(results.items()):
        print(f"Seed {seed} {r['head']}:")
        print(f"  mean attribution: peak={r['mean_attr_peak']:+.3f}  final={r['mean_attr_final']:+.3f}")
        print(f"  frac items where head favours period: peak={r['pos_frac_peak']:.1%}  final={r['pos_frac_final']:.1%}")
    print("Done.")
