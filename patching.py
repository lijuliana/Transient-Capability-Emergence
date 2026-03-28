"""
Residual-stream activation patching for transient-probe mechanistic analysis.

Protocol
--------
For each transient probe (modal_continuation, end_of_sentence, adjective_order):
  1. Load the peak checkpoint P and final checkpoint F for the same seed.
  2. Capture residual-stream activations (x_P[L], x_F[L]) at every layer
     boundary in both models.
  3. "Patch at depth L": run the final model but replace its residual-stream
     at the output of layer L with the peak model's value at that layer, then
     continue the final model from layer L+1 onwards.
  4. Measure probe accuracy (argmax_acc) at each patch depth L ∈ {0..n_layer}.
     L=0 means replacing at the embedding level (should recover ~peak accuracy).
     L=n_layer means no patching (final model accuracy).

The depth where accuracy drops as L increases is the "critical layer": the
layer whose computation (weights) changed most during the peak→final interval.

Output: figures/fig_patching.png — one row per probe, accuracy vs. patch depth,
plus a summary table written to patching_results.md.

Usage
-----
    python patching.py [--seed 99] [--probe-logs probe_log_seed99.tsv]
"""
import os
import sys
import re
import math
import glob
import argparse
import pathlib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Load model classes from train.py without executing the script body.
# We do this by compiling only the class/function definitions.
# ---------------------------------------------------------------------------

def _load_model_classes():
    """Return GPT, GPTConfig, norm by surgically importing train.py."""
    import importlib.util
    import types

    spec = importlib.util.spec_from_file_location(
        "_train_module", pathlib.Path(__file__).parent / "train.py"
    )
    # Create a module with a sentinel to stop execution at the config block
    mod = types.ModuleType("_train_module")
    mod.__spec__ = spec
    mod.__file__ = str(pathlib.Path(__file__).parent / "train.py")

    src = pathlib.Path("train.py").read_text()

    # Only execute lines up to (not including) the training-config block.
    # The sentinel is the first standalone assignment of ASPECT_RATIO which
    # is the first training-loop constant.
    stop_marker = "ASPECT_RATIO = "
    stop_idx = src.find(stop_marker)
    if stop_idx == -1:
        raise RuntimeError("Could not find stop marker in train.py")

    # Trim to just definitions
    src_defs = src[:stop_idx]

    # Some imports in train.py reference things that may fail (rustbpe, etc.)
    # Execute in a protected namespace.
    ns = {
        "__name__": "_train_module",
        "__file__": str(pathlib.Path("train.py").resolve()),
        "__builtins__": __builtins__,
    }
    exec(compile(src_defs, "train.py", "exec"), ns)
    return ns


_TRAIN_NS = None

def _get_train_ns():
    global _TRAIN_NS
    if _TRAIN_NS is None:
        _TRAIN_NS = _load_model_classes()
    return _TRAIN_NS


def build_model(config_dict):
    """Reconstruct GPT from a saved config dict."""
    ns = _get_train_ns()
    GPTConfig = ns["GPTConfig"]
    GPT = ns["GPT"]
    cfg = GPTConfig(**{k: v for k, v in config_dict.items()
                       if k in GPTConfig.__dataclass_fields__})
    model = GPT(cfg)
    return model


def load_checkpoint(ckpt_path, device="cpu"):
    """Load a checkpoint and return (model, step, config_dict)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(ckpt["config"])
    state = ckpt["model_state"]
    # torch.compile wraps keys with "_orig_mod." prefix — strip it if present
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    return model, ckpt["step"], ckpt["config"]


# ---------------------------------------------------------------------------
# Residual-stream capture + patching
# ---------------------------------------------------------------------------

def capture_residuals(model, idx, device):
    """
    Run a forward pass and return the residual-stream tensor at each layer
    boundary (after each Block).  Returns list of length n_layer+1:
      residuals[0] = x after embedding + norm (= x0)
      residuals[i+1] = x after block i
    Also returns final logits.
    """
    model.eval()
    idx = idx.to(device)
    residuals = []
    with torch.no_grad():
        T = idx.size(1)
        cos_sin = model.cos[:, :T], model.sin[:, :T]
        x = model.transformer.wte(idx)
        ns = _get_train_ns()
        norm_fn = ns["norm"]
        x = norm_fn(x)
        x0 = x.clone()
        residuals.append(x.clone())  # residuals[0] = after embedding
        for i, block in enumerate(model.transformer.h):
            x = model.resid_lambdas[i] * x + model.x0_lambdas[i] * x0
            ve = (model.value_embeds[str(i)](idx)
                  if str(i) in model.value_embeds else None)
            x = block(x, ve, cos_sin, model.window_sizes[i])
            residuals.append(x.clone())  # residuals[i+1] = after block i
    return residuals  # list of (B, T, n_embd) tensors, not including final norm/lm_head


def patched_logits(final_model, peak_residuals, patch_depth, idx, device):
    """
    Run the final model but inject peak_residuals[patch_depth] at depth
    patch_depth, then run all subsequent layers with the final model.

    patch_depth = 0: patch at the embedding level (use peak embedding as x0)
    patch_depth = k: patch after block k-1 (inject peak residual after block k-1)
    patch_depth = n_layer: no patch (run entirely with final model)
    """
    final_model.eval()
    idx = idx.to(device)
    ns = _get_train_ns()
    norm_fn = ns["norm"]
    n_layer = final_model.config.n_layer
    T = idx.size(1)
    cos_sin = final_model.cos[:, :T], final_model.sin[:, :T]

    with torch.no_grad():
        x = final_model.transformer.wte(idx)
        x = norm_fn(x)
        x0 = x.clone()

        if patch_depth == 0:
            # Replace embedding entirely with peak's embedding
            x = peak_residuals[0].to(device)
            x0 = peak_residuals[0].to(device)

        for i, block in enumerate(final_model.transformer.h):
            if patch_depth > 0 and i == 0:
                # Start layers with final x / x0
                pass  # already set above
            x = final_model.resid_lambdas[i] * x + final_model.x0_lambdas[i] * x0
            ve = (final_model.value_embeds[str(i)](idx)
                  if str(i) in final_model.value_embeds else None)
            x = block(x, ve, cos_sin, final_model.window_sizes[i])
            # Inject peak residual right after this block if we're at patch_depth
            if i + 1 == patch_depth:
                x = peak_residuals[patch_depth].to(device)

        x = norm_fn(x)
        softcap = 15
        logits = final_model.lm_head(x).float()
        logits = softcap * torch.tanh(logits / softcap)
    return logits


def probe_acc_from_logits_batch(logits_batch, metadata):
    """
    Compute argmax_acc from a list of (logits, meta) pairs where meta has
    keys: probe, idx, side, plen, slen, seq.
    Returns dict probe_name -> argmax_acc.
    """
    scores = {}
    for logits, m in zip(logits_batch, metadata):
        plen, slen, seq = m["plen"], m["slen"], m["seq"]
        logprobs = F.log_softmax(logits.float(), dim=-1)  # (T, V)
        lp_sum, n_counted = 0.0, 0
        for k in range(slen):
            pos = plen - 1 + k
            if pos + 1 > logprobs.size(0) or plen + k >= len(seq):
                break
            tok = seq[plen + k]
            lp_sum += logprobs[pos, tok].item()
            n_counted += 1
        lp = lp_sum / max(n_counted, 1)
        key = (m["probe"], m["idx"])
        if key not in scores:
            scores[key] = {}
        scores[key][m["side"]] = lp

    results = {}
    for (probe, idx), s in scores.items():
        if "correct" not in s or "distractor" not in s:
            continue
        results.setdefault(probe, {"correct": 0, "total": 0})
        results[probe]["total"] += 1
        if s["correct"] > s["distractor"]:
            results[probe]["correct"] += 1

    return {k: v["correct"] / v["total"] for k, v in results.items() if v["total"] > 0}


# ---------------------------------------------------------------------------
# Probe item preparation (standalone, no tokenizer needed for loading)
# ---------------------------------------------------------------------------

def load_prepared_probes(train_ns, tokenizer):
    """Call train.py's prepare_probes with the given tokenizer."""
    prepare_probes_fn = train_ns["prepare_probes"]
    return prepare_probes_fn(tokenizer)


def build_probe_batch(prepared_probes, target_probes, max_len=48):
    """Build a flat list of sequences + metadata for the target probes only."""
    all_seqs = []
    metadata = []
    for probe_name in target_probes:
        items = prepared_probes.get(probe_name, [])
        for item_idx, item in enumerate(items):
            prefix = item["prefix_ids"]
            for side, suffix in [("correct", item["correct_ids"]),
                                  ("distractor", item["distractor_ids"])]:
                full = (prefix + suffix)[:max_len]
                all_seqs.append(full)
                metadata.append({
                    "probe": probe_name, "idx": item_idx, "side": side,
                    "plen": len(prefix), "slen": len(suffix), "seq": full,
                })
    max_seqlen = max(len(s) for s in all_seqs) if all_seqs else 1
    padded = [s + [0] * (max_seqlen - len(s)) for s in all_seqs]
    return padded, metadata


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def find_checkpoints(seed, ckpt_dir="checkpoints"):
    """Return sorted list of checkpoint paths for this seed."""
    pattern = str(pathlib.Path(ckpt_dir) / f"seed{seed}_step*.pt")
    paths = sorted(glob.glob(pattern),
                   key=lambda p: int(re.search(r"step(\d+)", p).group(1)))
    return paths


def find_peak_and_final(probe_log_path, probe_name, min_step=10, smooth_window=11):
    """Return (peak_step, final_step) from a probe log (smoothed argmax_acc)."""
    df = pd.read_csv(probe_log_path, sep="\t")
    sub = df[df.probe_name == probe_name].sort_values("step")
    t, y = sub.step.values, sub.argmax_acc.values
    mask = t > min_step
    t2, y2 = t[mask], y[mask]
    sm = (np.convolve(y2, np.ones(smooth_window) / smooth_window, mode="same")
          if len(y2) >= smooth_window else y2)
    peak_step = int(t2[np.argmax(sm)])
    final_step = int(t[-1])
    return peak_step, final_step


def nearest_checkpoint(ckpt_paths, target_step):
    """Return the checkpoint path whose step is closest to target_step."""
    steps = [int(re.search(r"step(\d+)", p).group(1)) for p in ckpt_paths]
    idx = int(np.argmin(np.abs(np.array(steps) - target_step)))
    return ckpt_paths[idx], steps[idx]


def best_peak_checkpoint(ckpt_paths, probe_log_path, probe_name, min_step=10):
    """
    Pick the checkpoint whose recorded probe accuracy (in the log) is highest.
    Among the saved checkpoints, look up the argmax_acc at each checkpoint step
    (using the nearest log entry within 5 steps), then return the one with max acc.
    This is more robust than smoothed-peak estimation when checkpoints are sparse.
    """
    df = pd.read_csv(probe_log_path, sep="\t")
    sub = df[(df.probe_name == probe_name) & (df.step > min_step)].sort_values("step")
    log_steps = sub.step.values
    log_accs = sub.argmax_acc.values

    best_path, best_step, best_acc = None, None, -1.0
    for path in ckpt_paths:
        ckpt_step = int(re.search(r"step(\d+)", path).group(1))
        if ckpt_step <= min_step:
            continue
        # find nearest log entry
        dists = np.abs(log_steps - ckpt_step)
        nearest_idx = int(np.argmin(dists))
        if dists[nearest_idx] > 10:  # skip if no log entry within 10 steps
            continue
        acc = log_accs[nearest_idx]
        if acc > best_acc:
            best_acc = acc
            best_path = path
            best_step = ckpt_step

    if best_path is None:
        # fallback to smoothed peak
        peak_step, _ = find_peak_and_final(probe_log_path, probe_name, min_step)
        return nearest_checkpoint(ckpt_paths, peak_step)

    print(f"  Best checkpoint by probe acc: step {best_step}, acc={best_acc:.3f}")
    return best_path, best_step


def run_patching(seed, probe_names, ckpt_dir="checkpoints",
                 probe_log_pattern="probe_log_seed{seed}.tsv", device="cuda",
                 max_final_step=None):
    """
    For each probe, find peak & final checkpoints, capture residuals, then
    sweep patch_depth from 0 to n_layer and record accuracy.
    Returns dict: probe_name -> {patch_depth -> acc}
    """
    probe_log_path = probe_log_pattern.format(seed=seed)
    if not pathlib.Path(probe_log_path).exists():
        raise FileNotFoundError(f"Probe log not found: {probe_log_path}")

    ckpt_paths = find_checkpoints(seed, ckpt_dir)
    if not ckpt_paths:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir} for seed {seed}")
    print(f"Found {len(ckpt_paths)} checkpoints for seed {seed}")

    # Determine final checkpoint.  If max_final_step is given, use the
    # largest-step _final.pt whose step ≤ max_final_step; otherwise use
    # the globally largest-step _final.pt (or non-final checkpoint).
    final_paths = [p for p in ckpt_paths if "_final" in p]
    if max_final_step is not None:
        final_paths = [p for p in final_paths
                       if int(re.search(r"step(\d+)", p).group(1)) <= max_final_step]
    if final_paths:
        final_ckpt_path = final_paths[-1]
        final_step = int(re.search(r"step(\d+)", final_ckpt_path).group(1))
    else:
        ub = max_final_step if max_final_step is not None else 9999999
        final_ckpt_path, final_step = nearest_checkpoint(ckpt_paths, ub)

    print(f"Loading final checkpoint: {final_ckpt_path} (step {final_step})")
    final_model, _, config = load_checkpoint(final_ckpt_path, device)
    n_layer = final_model.config.n_layer

    # Load tokenizer and prepare probes
    ns = _get_train_ns()
    tokenizer_cls = ns["Tokenizer"]
    tokenizer = tokenizer_cls.from_directory()
    prepared_probes = load_prepared_probes(ns, tokenizer)

    results = {}
    for probe_name in probe_names:
        print(f"\n=== Patching probe: {probe_name} ===")
        # Use best-accuracy checkpoint rather than nearest-to-smoothed-peak
        peak_ckpt_path, actual_peak_step = best_peak_checkpoint(
            ckpt_paths, probe_log_path, probe_name)
        print(f"  Loading peak checkpoint: {peak_ckpt_path} (step {actual_peak_step})")
        peak_model, _, _ = load_checkpoint(peak_ckpt_path, device)

        # Build probe batch
        padded, metadata = build_probe_batch(prepared_probes, [probe_name])
        if not padded:
            print(f"  No items for {probe_name}, skipping")
            continue
        idx_tensor = torch.tensor(padded, dtype=torch.long, device=device)

        # Capture residuals from peak model
        print("  Capturing peak residuals...")
        peak_residuals = capture_residuals(peak_model, idx_tensor, device)

        # Sweep patch depth
        depth_acc = {}
        for depth in range(n_layer + 1):
            if depth == n_layer:
                # No patching — pure final model
                with torch.no_grad():
                    logits_all = final_model(idx_tensor).float()  # (B, T, V)
            else:
                logits_all = patched_logits(final_model, peak_residuals,
                                            depth, idx_tensor, device)
            # Compute accuracy
            logits_list = [logits_all[i] for i in range(logits_all.size(0))]
            acc_dict = probe_acc_from_logits_batch(logits_list, metadata)
            acc = acc_dict.get(probe_name, float("nan"))
            depth_acc[depth] = acc
            label = f"patch_depth={depth}" if depth < n_layer else "no_patch(final)"
            print(f"  {label:30s}  argmax_acc={acc:.3f}")

        results[probe_name] = depth_acc

        # Also get baseline peak accuracy (pure peak model)
        with torch.no_grad():
            peak_logits_all = peak_model(idx_tensor).float()
        peak_logits_list = [peak_logits_all[i] for i in range(peak_logits_all.size(0))]
        peak_acc = probe_acc_from_logits_batch(peak_logits_list, metadata).get(probe_name, float("nan"))
        print(f"  pure_peak_model              argmax_acc={peak_acc:.3f}")
        results[probe_name]["peak_baseline"] = peak_acc
        results[probe_name]["peak_step"] = actual_peak_step
        results[probe_name]["final_step"] = final_step

    return results


def make_patching_figure(results, n_layer, out_path="figures/fig_patching.png"):
    """
    One subplot per probe. X axis = patch depth (0..n_layer).
    Horizontal dashed lines at peak and final baselines.
    """
    probe_names = [k for k in results if isinstance(results[k], dict) and 0 in results[k]]
    n = len(probe_names)
    if n == 0:
        print("No results to plot.")
        return

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), squeeze=False)
    layer_labels = [f"L{i}" for i in range(n_layer)] + ["none\n(final)"]

    for col, probe in enumerate(probe_names):
        ax = axes[0][col]
        d = results[probe]
        depths = list(range(n_layer + 1))
        accs = [d.get(depth, float("nan")) for depth in depths]
        final_acc = d.get(n_layer, float("nan"))
        peak_acc = d.get("peak_baseline", float("nan"))

        ax.plot(depths, accs, "o-", color="C0", linewidth=2, markersize=7,
                label="patched final model")
        ax.axhline(peak_acc, color="green", linestyle="--", linewidth=1.2,
                   label=f"pure peak (step {d.get('peak_step','?')}): {peak_acc:.2f}")
        ax.axhline(final_acc, color="red", linestyle="--", linewidth=1.2,
                   label=f"pure final (step {d.get('final_step','?')}): {final_acc:.2f}")
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

        ax.set_xticks(depths)
        ax.set_xticklabels(layer_labels, fontsize=9)
        ax.set_xlabel("Patch depth\n(peak residual injected after this layer)")
        ax.set_ylabel("argmax_acc" if col == 0 else "")
        ax.set_ylim(0.0, 1.05)
        ax.set_title(probe.replace("_", " "), fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.3, axis="y")

    fig.suptitle(
        "Activation patching: peak checkpoint residuals injected into final model\n"
        "Accuracy ↑ at depth L = peak's representation at L is sufficient for the final model to recover the capability",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nWrote {out_path}")


def write_patching_results(results, n_layer, out_path="patching_results.md"):
    with open(out_path, "w") as f:
        f.write("# Activation Patching Results\n\n")
        f.write("Protocol: inject peak-checkpoint residual stream at each layer boundary into "
                "the final-checkpoint model. Accuracy is argmax_acc on the probe.\n\n")
        for probe, d in results.items():
            if not isinstance(d, dict) or 0 not in d:
                continue
            f.write(f"## {probe}\n\n")
            f.write(f"- Peak checkpoint step: {d.get('peak_step', '?')}, "
                    f"acc = {d.get('peak_baseline', float('nan')):.3f}\n")
            f.write(f"- Final checkpoint step: {d.get('final_step', '?')}, "
                    f"acc = {d.get(n_layer, float('nan')):.3f}\n\n")
            f.write("| patch_depth | acc |\n|---|---|\n")
            for depth in range(n_layer + 1):
                label = f"after layer {depth-1}" if depth > 0 else "at embedding"
                acc = d.get(depth, float("nan"))
                f.write(f"| {depth} ({label}) | {acc:.3f} |\n")
            # Find critical layer (biggest drop going from depth L to L+1)
            drops = [(d.get(i, float("nan")) - d.get(i+1, float("nan")), i)
                     for i in range(n_layer)]
            drops = [(drop, i) for drop, i in drops
                     if not math.isnan(drop)]
            if drops:
                max_drop, crit_layer = max(drops)
                f.write(f"\n**Critical layer** (largest accuracy drop): "
                        f"layer {crit_layer}→{crit_layer+1} (Δ={max_drop:.3f})\n\n")
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--ckpt-dir", default="checkpoints")
    parser.add_argument("--probe-logs", default="probe_log_seed{seed}.tsv")
    parser.add_argument("--probes", nargs="+",
                        default=["modal_continuation", "end_of_sentence", "adjective_order"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-final-step", type=int, default=None,
                        help="Only consider _final checkpoints up to this step (use "
                             "with fine-checkpoint runs that share a seed with earlier runs)")
    args = parser.parse_args()

    print(f"Running activation patching: seed={args.seed}, device={args.device}")
    print(f"Probes: {args.probes}")

    results = run_patching(
        seed=args.seed,
        probe_names=args.probes,
        ckpt_dir=args.ckpt_dir,
        probe_log_pattern=args.probe_logs,
        device=args.device,
        max_final_step=args.max_final_step,
    )

    # Determine n_layer from any result
    n_layer = 4  # default for this architecture
    for d in results.values():
        if isinstance(d, dict) and n_layer in d:
            break

    make_patching_figure(results, n_layer)
    write_patching_results(results, n_layer)
    print("\nDone.")


if __name__ == "__main__":
    main()
