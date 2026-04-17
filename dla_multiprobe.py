"""
Generalised DLA for multiple probes and seeds.

For each probe item, compute:
  logit_diff(item) = logit(correct_id) - logit(distractor_id)
  head_attribution = full_logit_diff - ablated_logit_diff

Probes:
  end_of_sentence:   correct=46 ('.'), distractor=44 (',')
  modal_continuation: correct=465 (' will'), distractor=306 (' is')

Run at both peak and final checkpoints for both seeds.
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

# Per-probe: (correct_token_id, distractor_token_id, description)
PROBE_TOKENS = {
    "end_of_sentence":    (46,  44,  "period vs comma"),
    "modal_continuation": (465, 306, "'will' vs 'is'"),
}


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
    print(f"  Seed {seed}, probe={probe_name}: peak={peak_step} final={final_step}")
    return pm, peak_step, fm, final_step


def run_forward(model, seqs, device, ablate_block=None, ablate_head=None):
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
                x = ns["norm"](x)
                x0 = x.clone()
                for i, block in enumerate(model.transformer.h):
                    x = model.resid_lambdas[i] * x + model.x0_lambdas[i] * x0
                    ve = (model.value_embeds[str(i)](idx)
                          if str(i) in model.value_embeds else None)
                    x = block(x, ve, cos_sin, model.window_sizes[i])
                x = ns["norm"](x)
                softcap = 15
                logits = model.lm_head(x).float()
                logits = softcap * torch.tanh(logits / softcap)
                logits_batch.append(logits.squeeze(0))
    finally:
        if handle is not None:
            handle.remove()
    return logits_batch


def extract_logit_diffs(logits_batch, probe_items, correct_id, distractor_id):
    diffs = []
    for i, item in enumerate(probe_items):
        prefix_len = len(item["prefix_ids"])
        pos = prefix_len - 1
        logits = logits_batch[i]
        if pos >= logits.shape[0]:
            diffs.append(float("nan"))
            continue
        ld = (logits[pos, correct_id] - logits[pos, distractor_id]).item()
        diffs.append(ld)
    return np.array(diffs)


def run_analysis(seed, probe_name, device):
    print(f"\n=== Seed {seed} / Probe: {probe_name} ===")
    block_idx, head_idx = CRITICAL_HEADS[seed]
    head_name = f"B{block_idx}H{head_idx}"
    correct_id, distractor_id, token_desc = PROBE_TOKENS[probe_name]

    peak_model, peak_step, final_model, final_step = select_checkpoints(
        seed, probe_name, device)

    tok = Tokenizer.from_directory()
    ns = _get_train_ns()
    preps = ns["prepare_probes"](tok)
    probe_items = preps[probe_name]

    seqs = [item["prefix_ids"] + [correct_id] for item in probe_items]
    print(f"  {len(seqs)} probe items; correct={correct_id} distractor={distractor_id} ({token_desc})")

    lf_pk  = run_forward(peak_model,  seqs, device)
    la_pk  = run_forward(peak_model,  seqs, device, ablate_block=block_idx, ablate_head=head_idx)
    lf_fn  = run_forward(final_model, seqs, device)
    la_fn  = run_forward(final_model, seqs, device, ablate_block=block_idx, ablate_head=head_idx)

    ld_pk_full = extract_logit_diffs(lf_pk, probe_items, correct_id, distractor_id)
    ld_pk_abl  = extract_logit_diffs(la_pk, probe_items, correct_id, distractor_id)
    ld_fn_full = extract_logit_diffs(lf_fn, probe_items, correct_id, distractor_id)
    ld_fn_abl  = extract_logit_diffs(la_fn, probe_items, correct_id, distractor_id)

    attr_peak  = ld_pk_full - ld_pk_abl
    attr_final = ld_fn_full - ld_fn_abl
    other_pk   = ld_pk_full - attr_peak   # = ld_pk_abl
    other_fn   = ld_fn_full - attr_final  # = ld_fn_abl

    def stats(arr, label):
        valid = arr[~np.isnan(arr)]
        m = float(np.mean(valid))
        s = float(np.std(valid))
        pos_frac = float(np.mean(valid > 0))
        acc = float(np.mean(valid > 0))
        print(f"    {label}: mean={m:+.3f} ± {s:.3f}  frac_pos={pos_frac:.1%}")
        return m, s, pos_frac

    print(f"\n  Full model logit_diff ({token_desc}) at last prefix position:")
    stats(ld_pk_full,  f"peak (step {peak_step})")
    stats(ld_fn_full,  f"final (step {final_step})")

    print(f"\n  {head_name} attribution = full − ablated:")
    m_pk, s_pk, pos_pk = stats(attr_peak,  "peak  attribution")
    m_fn, s_fn, pos_fn = stats(attr_final, "final attribution")

    print(f"\n  Other-heads contribution (full − head_attr = ablated model logit_diff):")
    stats(other_pk, "peak  other-heads")
    stats(other_fn, "final other-heads")

    acc_pk_full = float(np.mean(ld_pk_full[~np.isnan(ld_pk_full)] > 0))
    acc_fn_full = float(np.mean(ld_fn_full[~np.isnan(ld_fn_full)] > 0))
    acc_pk_abl  = float(np.mean(ld_pk_abl[~np.isnan(ld_pk_abl)]   > 0))
    acc_fn_abl  = float(np.mean(ld_fn_abl[~np.isnan(ld_fn_abl)]   > 0))
    print(f"\n  Accuracy check:")
    print(f"    peak full={acc_pk_full:.3f}  ablated={acc_pk_abl:.3f}  ΔAcc={acc_pk_abl-acc_pk_full:+.3f}")
    print(f"    final full={acc_fn_full:.3f}  ablated={acc_fn_abl:.3f}  ΔAcc={acc_fn_abl-acc_fn_full:+.3f}")

    return dict(
        seed=seed, probe=probe_name, head=head_name,
        peak_step=peak_step, final_step=final_step,
        token_desc=token_desc,
        ld_pk_full=ld_pk_full, ld_fn_full=ld_fn_full,
        attr_peak=attr_peak, attr_final=attr_final,
        other_pk=np.mean(other_pk[~np.isnan(other_pk)]),
        other_fn=np.mean(other_fn[~np.isnan(other_fn)]),
        mean_attr_peak=m_pk, mean_attr_final=m_fn,
        pos_frac_peak=pos_pk, pos_frac_final=pos_fn,
        acc_pk=acc_pk_full, acc_fn=acc_fn_full,
        acc_pk_abl=acc_pk_abl, acc_fn_abl=acc_fn_abl,
    )


def write_summary_md(all_results, path="dla_multiprobe_results.md"):
    lines = ["# DLA Multi-Probe Results\n\n"]
    lines.append("Head attribution = full_logit_diff − ablated_logit_diff.\n"
                 "Positive = head increases the correct-token advantage.\n\n")
    for r in all_results:
        lines.append(f"## Seed {r['seed']} / {r['probe']} — {r['head']}\n\n")
        lines.append(f"- Token comparison: {r['token_desc']}\n")
        lines.append(f"- Peak step: {r['peak_step']}, Final step: {r['final_step']}\n\n")
        lines.append("| checkpoint | full logit_diff | head attribution | other-heads |\n")
        lines.append("|---|---|---|---|\n")
        pk_ld = float(np.mean(r['ld_pk_full'][~np.isnan(r['ld_pk_full'])]))
        fn_ld = float(np.mean(r['ld_fn_full'][~np.isnan(r['ld_fn_full'])]))
        lines.append(
            f"| peak (step {r['peak_step']}) | {pk_ld:+.3f} "
            f"({r['acc_pk']:.1%} correct) | "
            f"{r['mean_attr_peak']:+.3f} ({r['pos_frac_peak']:.1%} pos.) | "
            f"{r['other_pk']:+.3f} |\n"
        )
        lines.append(
            f"| final (step {r['final_step']}) | {fn_ld:+.3f} "
            f"({r['acc_fn']:.1%} correct) | "
            f"{r['mean_attr_final']:+.3f} ({r['pos_frac_final']:.1%} pos.) | "
            f"{r['other_fn']:+.3f} |\n"
        )
        lines.append("\n")
    with open(path, "w") as f:
        f.writelines(lines)
    print(f"Wrote {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 99])
    parser.add_argument("--probes", nargs="+",
                        default=["end_of_sentence", "modal_continuation"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    all_results = []
    for seed in args.seeds:
        for probe in args.probes:
            r = run_analysis(seed, probe, args.device)
            all_results.append(r)

    write_summary_md(all_results)

    print("\n=== Cross-probe summary ===")
    for r in all_results:
        print(f"Seed {r['seed']} {r['head']} / {r['probe']}:")
        print(f"  peak:  attr={r['mean_attr_peak']:+.3f} ({r['pos_frac_peak']:.1%} pos.)  "
              f"other={r['other_pk']:+.3f}  acc={r['acc_pk']:.1%}")
        print(f"  final: attr={r['mean_attr_final']:+.3f} ({r['pos_frac_final']:.1%} pos.)  "
              f"other={r['other_fn']:+.3f}  acc={r['acc_fn']:.1%}")
    print("Done.")
