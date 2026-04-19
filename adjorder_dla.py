"""
Per-item DLA for adjective_order.

Since adjective_order has per-item (correct_id, distractor_id) pairs,
we use per-item token selection:
  logit_diff = logit(correct_ids[0]) - logit(distractor_ids[0])
  head_attribution = full_logit_diff - ablated_logit_diff

Items with multi-token correct are included using the first subword,
with a flag for purity. Analysis run for both seeds on their respective
critical heads (B0H0 for seed 17, B0H1 for seed 99).
"""
import os, sys, glob, re, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, '.')
from patching import _get_train_ns, load_checkpoint, best_peak_checkpoint
from prepare import Tokenizer

PROBE_LOG = {99: "probe_log_seed99.tsv", 17: "probe_log_seed17.tsv"}
CRITICAL_HEADS = {17: (0, 0), 99: (0, 1)}


def select_checkpoints(seed, probe_name, device):
    paths = sorted(glob.glob(f"checkpoints/seed{seed}_step*.pt"))
    paths = [p for p in paths if "step00000" not in p]
    peak_path, peak_step = best_peak_checkpoint(
        [p for p in paths if "_final" not in p], PROBE_LOG[seed], probe_name)
    finals = [p for p in paths if "_final" in p]
    fine = [p for p in finals if "step02" in p]
    final_path = fine[0] if fine else finals[0]
    final_step = int(re.search(r"step(\d+)", final_path).group(1))
    pm, _, _ = load_checkpoint(peak_path, device)
    fm, _, _ = load_checkpoint(final_path, device)
    print(f"  Seed {seed}: peak={peak_step} final={final_step}")
    return pm, peak_step, fm, final_step


def run_forward(model, seqs, device, ablate_block=None, ablate_head=None):
    ns = _get_train_ns()
    n_embd = model.config.n_embd
    n_head = model.config.n_head
    head_dim = n_embd // n_head

    handle = None
    if ablate_block is not None and ablate_head is not None:
        attn = model.transformer.h[ablate_block].attn
        h_s = ablate_head * head_dim
        h_e = (ablate_head + 1) * head_dim
        def _hook(module, inp):
            x = inp[0].clone()
            x[..., h_s:h_e] = 0.0
            return (x,)
        handle = attn.c_proj.register_forward_pre_hook(_hook)

    model.eval()
    out = []
    try:
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
                    x = block(x, ve, cos_sin, model.window_sizes[i])
                x = ns["norm"](x)
                sc = 15
                logits = sc * torch.tanh(model.lm_head(x).float() / sc)
                out.append(logits.squeeze(0))
    finally:
        if handle:
            handle.remove()
    return out


def extract_per_item_diffs(logits_list, probe_items):
    """Use per-item correct_ids[0] vs distractor_ids[0]."""
    diffs, correct_single, distractor_single = [], [], []
    for i, item in enumerate(probe_items):
        pos = len(item["prefix_ids"]) - 1
        logits = logits_list[i]
        if pos >= logits.shape[0]:
            diffs.append(float("nan"))
            continue
        c_id = item["correct_ids"][0]
        d_id = item["distractor_ids"][0]
        ld = (logits[pos, c_id] - logits[pos, d_id]).item()
        diffs.append(ld)
        correct_single.append(len(item["correct_ids"]) == 1)
        distractor_single.append(len(item["distractor_ids"]) == 1)
    return np.array(diffs), np.array(correct_single), np.array(distractor_single)


def run_analysis(seed, device):
    print(f"\n=== Seed {seed} / adjective_order ===")
    block_idx, head_idx = CRITICAL_HEADS[seed]
    head_name = f"B{block_idx}H{head_idx}"

    peak_model, peak_step, final_model, final_step = select_checkpoints(
        seed, "adjective_order", device)

    tok = Tokenizer.from_directory()
    ns = _get_train_ns()
    preps = ns["prepare_probes"](tok)
    items = preps["adjective_order"]

    # Build sequences: append first correct token
    seqs = [item["prefix_ids"] + [item["correct_ids"][0]] for item in items]
    print(f"  {len(items)} items (per-item correct/distractor first-token comparison)")
    n_clean = sum(1 for it in items
                  if len(it["correct_ids"]) == 1 and len(it["distractor_ids"]) == 1)
    print(f"  {n_clean} items with both single-token (clean DLA)")

    lf_pk = run_forward(peak_model,  seqs, device)
    la_pk = run_forward(peak_model,  seqs, device,
                        ablate_block=block_idx, ablate_head=head_idx)
    lf_fn = run_forward(final_model, seqs, device)
    la_fn = run_forward(final_model, seqs, device,
                        ablate_block=block_idx, ablate_head=head_idx)

    ld_pk, csingle, dsingle = extract_per_item_diffs(lf_pk, items)
    ld_pk_abl, _, _ = extract_per_item_diffs(la_pk, items)
    ld_fn, _, _    = extract_per_item_diffs(lf_fn, items)
    ld_fn_abl, _, _ = extract_per_item_diffs(la_fn, items)

    attr_pk = ld_pk - ld_pk_abl
    attr_fn = ld_fn - ld_fn_abl
    other_pk = ld_pk_abl
    other_fn = ld_fn_abl

    # Also compute for "clean" items only (both single-token)
    clean = csingle & dsingle
    print(f"\n  --- ALL items (first-token approx, n={np.sum(~np.isnan(ld_pk))}) ---")
    def stats(arr, label):
        v = arr[~np.isnan(arr)]
        m, s, pf = float(np.mean(v)), float(np.std(v)), float(np.mean(v > 0))
        print(f"    {label}: mean={m:+.3f} ± {s:.3f}  frac_pos={pf:.1%}")
        return m, s, pf

    print(f"  Full logit_diff (correct_id[0] − distractor_id[0]):")
    stats(ld_pk, f"peak  (step {peak_step})")
    stats(ld_fn, f"final (step {final_step})")
    print(f"  {head_name} attribution:")
    m_pk, _, pos_pk = stats(attr_pk, "peak")
    m_fn, _, pos_fn = stats(attr_fn, "final")
    print(f"  Other-heads:")
    stats(other_pk, "peak")
    stats(other_fn, "final")

    acc_pk = float(np.mean(ld_pk[~np.isnan(ld_pk)] > 0))
    acc_fn = float(np.mean(ld_fn[~np.isnan(ld_fn)] > 0))
    acc_pk_abl = float(np.mean(ld_pk_abl[~np.isnan(ld_pk_abl)] > 0))
    acc_fn_abl = float(np.mean(ld_fn_abl[~np.isnan(ld_fn_abl)] > 0))
    print(f"  Accuracy: peak full={acc_pk:.3f} ablated={acc_pk_abl:.3f}  ΔAcc={acc_pk_abl-acc_pk:+.3f}")
    print(f"            final full={acc_fn:.3f} ablated={acc_fn_abl:.3f}  ΔAcc={acc_fn_abl-acc_fn:+.3f}")

    if n_clean >= 3:
        print(f"\n  --- CLEAN items only (both single-token, n={int(clean.sum())}) ---")
        ld_pk_c   = ld_pk[clean]
        ld_fn_c   = ld_fn[clean]
        attr_pk_c = attr_pk[clean]
        attr_fn_c = attr_fn[clean]
        stats(ld_pk_c, f"peak  full logit_diff")
        stats(ld_fn_c, f"final full logit_diff")
        stats(attr_pk_c, f"peak  {head_name} attribution")
        stats(attr_fn_c, f"final {head_name} attribution")

    return dict(
        seed=seed, head=head_name, peak_step=peak_step, final_step=final_step,
        attr_pk=attr_pk, attr_fn=attr_fn,
        ld_pk=ld_pk, ld_fn=ld_fn,
        other_pk=float(np.mean(other_pk[~np.isnan(other_pk)])),
        other_fn=float(np.mean(other_fn[~np.isnan(other_fn)])),
        mean_attr_pk=m_pk, mean_attr_fn=m_fn,
        pos_pk=pos_pk, pos_fn=pos_fn,
        acc_pk=acc_pk, acc_fn=acc_fn,
        acc_pk_abl=acc_pk_abl, acc_fn_abl=acc_fn_abl,
        clean=clean,
    )


def make_figure(results):
    """Side-by-side attribution distribution for both seeds."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    for col, r in enumerate(results):
        seed = r['seed']
        head = r['head']

        # Row 0: full logit_diff scatter peak vs final
        ax = axes[0][col]
        valid = ~(np.isnan(r['ld_pk']) | np.isnan(r['ld_fn']))
        ax.scatter(r['ld_pk'][valid], r['ld_fn'][valid],
                   alpha=0.6, s=30, c='steelblue')
        # highlight clean items
        c = r.get('clean', np.ones(len(r['ld_pk']), dtype=bool))
        c_valid = valid & c
        ax.scatter(r['ld_pk'][c_valid], r['ld_fn'][c_valid],
                   alpha=0.8, s=50, c='orange', zorder=3, label='single-token (clean)')
        ax.axhline(0, color='red', lw=0.8, ls='--')
        ax.axvline(0, color='red', lw=0.8, ls='--')
        ax.plot([-5, 5], [-5, 5], 'k--', lw=0.5, alpha=0.3)
        ax.set_xlabel(f'Peak logit_diff (step {r["peak_step"]})', fontsize=9)
        ax.set_ylabel(f'Final logit_diff (step {r["final_step"]})', fontsize=9)
        ax.set_title(
            f'Seed {seed}: adj_order logit_diff scatter\npeak={r["acc_pk"]:.1%} final={r["acc_fn"]:.1%}',
            fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

        # Row 1: attribution distribution
        ax = axes[1][col]
        bins = 20
        v = ~(np.isnan(r['attr_pk']) | np.isnan(r['attr_fn']))
        ax.hist(r['attr_pk'][v], bins=bins, alpha=0.6, color='C0',
                label=f'peak (mean={r["mean_attr_pk"]:+.3f})')
        ax.hist(r['attr_fn'][v], bins=bins, alpha=0.6, color='C1',
                label=f'final (mean={r["mean_attr_fn"]:+.3f})')
        ax.axvline(0, color='black', lw=0.8, ls='--')
        ax.set_xlabel(f'{head} attribution (full−ablated)', fontsize=9)
        ax.set_ylabel('Count', fontsize=9)
        ax.set_title(
            f'Seed {seed}: {head} on adj_order\npeak pos={r["pos_pk"]:.1%}  final pos={r["pos_fn"]:.1%}',
            fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle(
        'Adjective-order DLA: critical head attribution (first-token approx)\n'
        'Orange dots = items with both single-token correct and distractor',
        fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs("figures", exist_ok=True)
    out = "figures/fig_adjorder_dla.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 99])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    results = [run_analysis(s, args.device) for s in args.seeds]
    make_figure(results)

    print("\n=== Cross-seed summary (critical head on adjective_order) ===")
    for r in results:
        print(f"Seed {r['seed']} {r['head']}:")
        print(f"  peak:  attribution={r['mean_attr_pk']:+.3f} ({r['pos_pk']:.1%} pos.)  "
              f"other={r['other_pk']:+.3f}  acc={r['acc_pk']:.1%}")
        print(f"  final: attribution={r['mean_attr_fn']:+.3f} ({r['pos_fn']:.1%} pos.)  "
              f"other={r['other_fn']:+.3f}  acc={r['acc_fn']:.1%}")
    print("Done.")
