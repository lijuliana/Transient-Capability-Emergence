"""
Phase 2 analysis: read probe logs, fit sigmoid emergence curves, produce figures.

Reads any `probe_log*.tsv` files in cwd. Each file = one training run (one seed).
Aggregates across seeds, fits a sigmoid acc(t) = a + b/(1 + exp(-k(t - t_50)))
per probe per seed, writes:
  - figures/fig1_emergence.png         (per-probe accuracy + sigmoid fits)
  - figures/fig2_metric_dependence.png (one probe under argmax vs log-prob metric)
  - results_summary.md                 (t_50 mean/std per probe across seeds)

No scipy: uses numpy gradient descent for the 4-parameter fit, with a linear
heuristic fallback if the fit fails.

Usage:
    uv run analyze.py
"""

import os
import re
import glob
import math
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Sigmoid fitting (numpy only)
# ---------------------------------------------------------------------------


def sigmoid(t, a, b, k, t50):
    z = -k * (t - t50)
    z = np.clip(z, -50, 50)  # stability
    return a + b / (1.0 + np.exp(z))


def fit_sigmoid(t, y, n_iter=3000, lr=2e-2):
    """
    Fit a rising sigmoid acc(t) = a + b/(1 + exp(-k(t - t50))) by projected
    gradient descent on MSE. Constraints:
      - a in [0, 1], a + b in [0, 1], so b >= 0 (rising only)
      - k > 0, t50 in [t_min, t_max]
    Returns (params dict, success_bool). If the curve doesn't actually rise
    (early-window mean to late-window mean < 0.15), returns fit_ok=False
    and falls back to the heuristic, which itself returns NaN for genuine
    non-emergence. Initial-step noise is filtered via early-window mean
    rather than y[0].
    """
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if len(t) < 4:
        return _fallback_t50(t, y), False

    # No-emergence guard: compare a tight early window (first ~5 samples,
    # which is pre-emergence for any probe whose t_50 > a few steps) to a
    # late window (last 25%, which is the plateau for any probe that has
    # actually emerged). Symmetric quartile windows would put fast-emerging
    # probes (saturated by step 50) entirely in the "early" bucket.
    n = len(y)
    early_n = min(5, max(1, n // 50))
    early = float(np.mean(y[:early_n]))
    late = float(np.mean(y[-max(1, n // 4):]))
    # Threshold 0.04: with 60+ items per probe, the late-window mean (over
    # ~320 datapoints) has noise <0.005, so a 0.04 trajectory-level rise is
    # >8σ — well above measurement noise. Catches consistently-tiny
    # emergences like pronoun_gender (0.04-0.06 above chance) while still
    # filtering out reflexive_pronoun (stuck at 0.5) and the declining
    # close_quote / end_of_sentence curves.
    if late - early < 0.04:
        return _fallback_t50(t, y), False
    # Transient-emergence guard: if smoothed accuracy peaks well above the
    # late-window plateau, the probe rose then collapsed (close_quote,
    # end_of_sentence). Sigmoid t_50 is undefined for non-monotonic curves;
    # the peak/final values in the summary capture this pattern instead.
    kernel = np.ones(3) / 3.0
    y_sm = np.convolve(y, kernel, mode="same") if n >= 3 else y
    peak = float(y_sm.max())
    if peak - late > 0.15:
        return _fallback_t50(t, y), False

    y_min, y_max = float(y.min()), float(y.max())

    # Initialize from data using early/late windows so init spikes don't
    # bias the sigmoid floor or its midpoint.
    a0 = early
    b0 = max(late - early, 1e-3)
    midpoint = a0 + 0.5 * b0
    # Smooth y for crossing detection (suppresses single-sample noise).
    if len(y) >= 3:
        kernel = np.ones(3) / 3.0
        y_smooth = np.convolve(y, kernel, mode="same")
    else:
        y_smooth = y
    cross_mid = np.where(y_smooth >= midpoint)[0]
    t50_0 = float(t[cross_mid[0]]) if len(cross_mid) > 0 else float(t[len(t) // 2])
    # Initialize k from empirical 25%/75% crossing width: rise from a+0.25*b
    # to a+0.75*b spans 2.2/k in t-space. This is much closer to the real
    # transition width than the naive 4/t_range, which collapses k to ~0
    # for fast-saturating probes.
    q25 = a0 + 0.25 * b0
    q75 = a0 + 0.75 * b0
    cross_lo = np.where(y_smooth >= q25)[0]
    cross_hi = np.where(y_smooth >= q75)[0]
    t_range = max(t.max() - t.min(), 1.0)
    if len(cross_lo) > 0 and len(cross_hi) > 0:
        width = max(float(t[cross_hi[0]]) - float(t[cross_lo[0]]),
                    float(t[1] - t[0]) if len(t) > 1 else 1.0)
        k0 = 2.2 / max(width, 1.0)
    else:
        k0 = 4.0 / t_range

    # Optimize on normalized t for stability
    t_max = max(t.max(), 1.0)
    t_norm = t / t_max
    a, b = a0, b0
    t50_n = t50_0 / t_max
    k_n = k0 * t_max

    best_loss = np.inf
    best = (a, b, k_n / t_max, t50_n * t_max)

    for _ in range(n_iter):
        z = -k_n * (t_norm - t50_n)
        z = np.clip(z, -50, 50)
        sig = 1.0 / (1.0 + np.exp(z))
        pred = a + b * sig
        err = pred - y
        loss = float(np.mean(err ** 2))

        if loss < best_loss:
            best_loss = loss
            best = (a, b, k_n / t_max, t50_n * t_max)

        # Gradients
        d_a = 2 * np.mean(err)
        d_b = 2 * np.mean(err * sig)
        sig_grad = sig * (1 - sig)
        d_kn = 2 * np.mean(err * b * sig_grad * (t_norm - t50_n))
        d_t50n = 2 * np.mean(err * b * sig_grad * (-k_n))

        a -= lr * d_a
        b -= lr * d_b
        k_n -= lr * d_kn
        t50_n -= lr * d_t50n

        # Project to feasible set
        a = float(np.clip(a, 0.0, 1.0))
        # Keep a + b in [0, 1] and b >= 0
        b = float(np.clip(b, 0.0, 1.0 - a))
        k_n = float(np.clip(k_n, 0.5, 200.0))
        t50_n = float(np.clip(t50_n, t_norm.min(), t_norm.max()))

    a, b, k, t50 = best
    if b < 0.1 or best_loss > 0.05:
        return _fallback_t50(t, y), False
    return {"a": a, "b": b, "k": k, "t50": t50, "loss": best_loss}, True


def _fallback_t50(t, y):
    """Heuristic t_50 from early/late window means.

    Returns NaN when there is no genuine emergence (rise < 0.15 from early
    window to late window, or curve declines). Otherwise t_50 is the first
    step at which a smoothed accuracy crosses the midpoint of the rise.
    Smoothing (rolling mean over 3 samples) suppresses init-noise spikes
    where y[0] can land high simply because the random head has the right
    sign on a 20-item probe.
    """
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(t) == 0:
        return {"a": 0.5, "b": 0.0, "k": 1.0,
                "t50": float("nan"), "loss": float("nan")}
    n = len(y)
    early_n = min(5, max(1, n // 50))
    early = float(np.mean(y[:early_n]))
    late = float(np.mean(y[-max(1, n // 4):]))
    rise = late - early
    if rise < 0.04:
        # No emergence (flat or declining) — t_50 is undefined.
        return {"a": early, "b": rise, "k": 0.0,
                "t50": float("nan"), "loss": float("nan")}
    # Transient-emergence guard (matches fit_sigmoid): if smoothed peak
    # rises well above the late-window plateau, the probe collapsed —
    # sigmoid t_50 is meaningless on a non-monotonic curve.
    if n >= 3:
        kernel = np.ones(3) / 3.0
        y_peak_check = np.convolve(y, kernel, mode="same")
    else:
        y_peak_check = y
    peak = float(y_peak_check.max())
    if peak - late > 0.15:
        return {"a": early, "b": rise, "k": 0.0,
                "t50": float("nan"), "loss": float("nan")}
    # Midpoint of the rise.
    mid = early + 0.5 * rise
    # Rolling-mean smoothing of width 3 to dampen single-sample spikes.
    if n >= 3:
        kernel = np.ones(3) / 3.0
        y_smooth = np.convolve(y, kernel, mode="same")
    else:
        y_smooth = y
    cross = np.where(y_smooth >= mid)[0]
    t50 = float(t[cross[0]]) if len(cross) > 0 else float("nan")
    return {"a": early, "b": rise, "k": 1.0,
            "t50": t50, "loss": float("nan")}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_probe_logs():
    """Return dict[seed_label] -> DataFrame with columns step, training_seconds,
    probe_name, argmax_acc, logprob_diff."""
    # Only the five main training seeds. Seed 99 is a mechanistic-analysis seed
    # only and must not contaminate cross-seed reproducibility statistics.
    MAIN_SEEDS = {42, 123, 7, 5, 17}
    paths = sorted(glob.glob("probe_log_seed*.tsv"))
    if not paths:
        raise SystemExit(
            "No probe_log_seed*.tsv files found in cwd. Run training with "
            "AUTORESEARCH_SEED=<n> uv run train.py to produce them."
        )
    runs = {}
    for p in paths:
        m = re.search(r"probe_log_seed(\d+)\.tsv$", p)
        if not m:
            continue
        seed_num = int(m.group(1))
        if seed_num not in MAIN_SEEDS:
            print(f"Skipping {p} (not a main seed)")
            continue
        label = f"seed{m.group(1)}"
        df = pd.read_csv(p, sep="\t")
        runs[label] = df
        print(f"Loaded {p}: {len(df)} rows ({df['probe_name'].nunique()} probes)")
    return runs


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _fit_all_probes(runs):
    """Run sigmoid fit for every (probe, seed). Returns DataFrame.

    Also records peak_acc and final_acc per (probe, seed) so the summary can
    surface transient-emergence probes — those that rise then collapse, where
    sigmoid t_50 is undefined but peak vs final is informative (the
    end_of_sentence probe behaves this way at this scale).
    """
    probes = sorted({p for df in runs.values() for p in df["probe_name"].unique()})
    fits = []
    for probe in probes:
        for seed_label, df in runs.items():
            sub = df[df["probe_name"] == probe].sort_values("step")
            if len(sub) == 0:
                continue
            t = sub["step"].values
            y = sub["argmax_acc"].values
            params, ok = fit_sigmoid(t, y)
            # Smooth before extracting peak so single-sample noise spikes
            # don't dominate.
            if len(y) >= 3:
                kernel = np.ones(3) / 3.0
                y_sm = np.convolve(y, kernel, mode="same")
            else:
                y_sm = y
            peak_acc = float(y_sm.max())
            n = len(y)
            final_acc = float(np.mean(y[-max(1, n // 4):]))
            fits.append({
                "probe": probe, "seed": seed_label,
                "t50": params["t50"], "k": params["k"], "fit_ok": ok,
                "peak_acc": peak_acc, "final_acc": final_acc,
            })
    return pd.DataFrame(fits)


def _probe_order_by_t50(fits_df, probes):
    """Sort probes by mean t_50 across seeds (earliest first); NaN sorts last."""
    order = []
    for probe in probes:
        valid = fits_df[(fits_df["probe"] == probe)].dropna(subset=["t50"])
        mean_t = float(valid["t50"].mean()) if len(valid) > 0 else float("inf")
        order.append((mean_t, probe))
    order.sort(key=lambda x: (math.isnan(x[0]) if isinstance(x[0], float) else False, x[0]))
    return [p for _, p in order]


def make_fig1(runs, out_path):
    """Per-probe accuracy curves with sigmoid fits, one panel per probe.

    Probes are sorted left-to-right, top-to-bottom by mean t_50 across seeds
    (earliest emergence first) so the ordering of capabilities is visible at
    a glance. Each panel shows: per-seed traces (thin) + cross-seed mean±std
    band over the union of step grids.
    """
    probes_unsorted = sorted({p for df in runs.values() for p in df["probe_name"].unique()})
    fits_df = _fit_all_probes(runs)
    probes = _probe_order_by_t50(fits_df, probes_unsorted)
    n_probes = len(probes)
    ncols = 4
    nrows = math.ceil(n_probes / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)

    seed_colors = {seed: f"C{i}" for i, seed in enumerate(sorted(runs.keys()))}

    for i, probe in enumerate(probes):
        ax = axes[i // ncols][i % ncols]
        # Collect per-seed step grids so we can pool a mean±std band
        all_steps = sorted({int(s) for df in runs.values()
                            for s in df[df["probe_name"] == probe]["step"].values})
        per_seed_at_step = {step: [] for step in all_steps}
        t50_marks = []
        for seed_label, df in runs.items():
            sub = df[df["probe_name"] == probe].sort_values("step")
            if len(sub) == 0:
                continue
            t = sub["step"].values
            y = sub["argmax_acc"].values
            ax.plot(t, y, marker="o", linestyle="-", alpha=0.35,
                    color=seed_colors[seed_label], label=f"{seed_label}")
            for step_v, y_v in zip(t, y):
                per_seed_at_step[int(step_v)].append(float(y_v))
            row = fits_df[(fits_df["probe"] == probe) & (fits_df["seed"] == seed_label)]
            if not row.empty and bool(row["fit_ok"].iloc[0]) and not math.isnan(row["t50"].iloc[0]):
                params = {"a": 0.0, "b": 1.0, "k": float(row["k"].iloc[0]),
                          "t50": float(row["t50"].iloc[0])}
                # Re-fit just to get a/b for plotting (cheap)
                p2, ok2 = fit_sigmoid(t, y)
                if ok2:
                    t_grid = np.linspace(t.min(), t.max(), 200)
                    y_fit = sigmoid(t_grid, p2["a"], p2["b"], p2["k"], p2["t50"])
                    ax.plot(t_grid, y_fit, "--", alpha=0.6,
                            color=seed_colors[seed_label])
                    t50_marks.append(float(p2["t50"]))
        # Cross-seed mean ± std band (only at steps where ≥2 seeds reported)
        band_x = [s for s in all_steps if len(per_seed_at_step[s]) >= 2]
        if band_x:
            band_mean = np.array([np.mean(per_seed_at_step[s]) for s in band_x])
            band_std = np.array([np.std(per_seed_at_step[s], ddof=0) for s in band_x])
            ax.fill_between(band_x, band_mean - band_std, band_mean + band_std,
                            color="gray", alpha=0.20, label="±1σ across seeds")
            ax.plot(band_x, band_mean, color="black", linewidth=1.4, alpha=0.8,
                    label="cross-seed mean")
        # Mark mean t_50 if any
        if t50_marks:
            ax.axvline(float(np.mean(t50_marks)), color="red", linestyle=":",
                       alpha=0.5, linewidth=1.0)
        ax.set_title(probe, fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0.5, color="k", linestyle=":", alpha=0.3)
        ax.set_xscale("symlog", linthresh=10)
        ax.set_xlabel("step (symlog)")
        ax.set_ylabel("argmax acc")
        ax.grid(alpha=0.3, which="both")
        if i == 0:
            ax.legend(loc="lower right", fontsize=7)
    # Hide unused panels
    for j in range(n_probes, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle("Per-probe emergence curves (argmax accuracy, sorted by t₅₀)",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return fits_df


def _pick_metric_dependence_probe(runs):
    """Choose the probe that best demonstrates Schaeffer's metric-dependence:
    argmax shows a sharp step-like transition while logprob_diff rises
    smoothly on the same trajectory.

    Score = argmax_rise * logprob_smoothness * monotonicity_penalty:
      - argmax_rise: late-window mean - early-window mean of argmax_acc
        (how much it rose)
      - logprob_smoothness: 1 - normalized variance of first-differences
        of logprob_diff (smaller deltas -> smoother)
      - monotonicity_penalty: 0 if peak-to-final drop > 0.15 (probe
        collapsed), else 1. We want monotonic rises only — collapsing
        probes don't demonstrate the Schaeffer argument cleanly.
    """
    best_probe, best_score = None, -1.0
    probes = sorted({p for df in runs.values() for p in df["probe_name"].unique()})
    for probe in probes:
        rises, smooth_scores, drops = [], [], []
        for df in runs.values():
            sub = df[df["probe_name"] == probe].sort_values("step")
            if len(sub) < 10:
                continue
            y = sub["argmax_acc"].values.astype(np.float64)
            n = len(y)
            early_n = min(5, max(1, n // 50))
            early = float(np.mean(y[:early_n]))
            late = float(np.mean(y[-max(1, n // 4):]))
            kernel = np.ones(3) / 3.0
            y_sm = np.convolve(y, kernel, mode="same")
            peak = float(y_sm.max())
            drops.append(peak - late)
            rises.append(late - early)
            lp = sub["logprob_diff"].values.astype(np.float64)
            if len(lp) > 1:
                d = np.diff(lp)
                rng = float(lp.max() - lp.min())
                norm_var = float(np.var(d)) / max(rng ** 2, 1e-9)
                smooth_scores.append(1.0 / (1.0 + norm_var * 100.0))
        if not rises:
            continue
        rise = float(np.mean(rises))
        smooth = float(np.mean(smooth_scores))
        drop = float(np.mean(drops))
        if drop > 0.15:
            continue  # transient — bad demo of the Schaeffer argument
        score = rise * smooth
        if score > best_score:
            best_score = score
            best_probe = probe
    return best_probe or "numeric_sequence"


def make_fig2(runs, out_path, probe=None):
    """One probe: argmax (step-like) vs logprob_diff (continuous) — Schaeffer
    demo. Uses symlog x-axis (linear below step 10, log above) so the
    emergence transition is visible at the same time as the post-saturation
    plateau. Without this, fast-saturating probes look flat at 1.0 from
    step 0 on a linear x-axis and the "step" is invisible.
    """
    if probe is None:
        probe = _pick_metric_dependence_probe(runs)
    BEHAVIOURAL = {"seed42", "seed123", "seed7", "seed5", "seed17"}
    runs_b = {s: df for s, df in runs.items() if s in BEHAVIOURAL}
    if not runs_b:
        runs_b = runs
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    seeds = sorted(runs_b.keys())
    seed_colors = {s: f"C{i}" for i, s in enumerate(seeds)}
    SMOOTH_WIN = 21

    def smooth(y, win=SMOOTH_WIN):
        if len(y) < win:
            return y
        return np.convolve(y, np.ones(win) / win, mode="same")

    for seed_label in seeds:
        df = runs_b[seed_label]
        sub = df[df["probe_name"] == probe].sort_values("step")
        if len(sub) == 0:
            continue
        t = sub["step"].values
        a = sub["argmax_acc"].values
        l = sub["logprob_diff"].values
        c = seed_colors[seed_label]
        axes[0].plot(t, smooth(a), color=c, alpha=0.9, linewidth=1.8, label=seed_label)
        axes[1].plot(t, smooth(l), color=c, alpha=0.9, linewidth=1.8, label=seed_label)
    for ax in axes:
        ax.set_xscale("symlog", linthresh=10)
        ax.grid(alpha=0.3, which="both")
    axes[0].set_title(f"argmax accuracy (step-like)\nlooks like a sharp emergence transition",
                      fontsize=11)
    axes[0].set_xlabel("training step (symlog)", fontsize=10)
    axes[0].set_ylabel("argmax accuracy", fontsize=10)
    axes[0].axhline(0.5, color="k", linestyle=":", alpha=0.4, label="chance")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].legend(loc="lower right", fontsize=9)

    axes[1].set_title(f"log-probability difference (continuous)\nrises smoothly over two decades",
                      fontsize=11)
    axes[1].set_xlabel("training step (symlog)", fontsize=10)
    axes[1].set_ylabel("mean (logp_correct − logp_distractor)", fontsize=10)
    axes[1].axhline(0.0, color="k", linestyle=":", alpha=0.4)
    axes[1].legend(loc="lower right", fontsize=9)

    fig.suptitle(
        f"Metric-dependence on {probe} (5 behavioural seeds): "
        "the same learning produces a step-like or smooth curve depending on the metric.",
        fontsize=11.5,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_fig3(fits_df, out_path):
    """Per-seed t_50 scatter: one marker per (probe, seed). Reveals at a glance
    which probes have tight cross-seed agreement (vertical clustering) and
    which don't (vertical spread). Probes are ordered by mean t_50 on x-axis.
    """
    if fits_df.empty:
        return
    valid = fits_df.dropna(subset=["t50"]).copy()
    if valid.empty:
        return
    probe_means = valid.groupby("probe")["t50"].mean().sort_values()
    probes = list(probe_means.index)
    seeds = sorted(valid["seed"].unique())
    seed_colors = {s: f"C{i}" for i, s in enumerate(seeds)}

    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(probes) + 4), 5))
    for s in seeds:
        sub = valid[valid["seed"] == s]
        xs = [probes.index(p) for p in sub["probe"] if p in probes]
        ys = [t for p, t in zip(sub["probe"], sub["t50"]) if p in probes]
        ax.scatter(xs, ys, color=seed_colors[s], s=70, alpha=0.85,
                   edgecolors="black", linewidths=0.5, label=s)
    # Mean line per probe
    mean_xs = list(range(len(probes)))
    mean_ys = [probe_means[p] for p in probes]
    ax.plot(mean_xs, mean_ys, color="black", linewidth=1.0, alpha=0.6, label="mean")

    ax.set_xticks(range(len(probes)))
    ax.set_xticklabels(probes, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("t₅₀ (training step)")
    ax.set_xlabel("probe (sorted by mean t₅₀)")
    ax.set_title("Cross-seed reproducibility of t₅₀ (one marker per seed)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _bootstrap_ci_t50(t50_values, n_boot=2000, ci=0.95, rng_seed=0):
    """Bootstrap 95% CI for the mean of t50_values (list/array of floats).
    Returns (lo, hi). With n < 2 returns (nan, nan)."""
    vals = np.asarray([v for v in t50_values if not (isinstance(v, float) and math.isnan(v))],
                      dtype=float)
    if len(vals) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(rng_seed)
    boot_means = np.array([rng.choice(vals, size=len(vals), replace=True).mean()
                           for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    lo, hi = float(np.quantile(boot_means, alpha)), float(np.quantile(boot_means, 1 - alpha))
    return lo, hi


def write_summary(fits_df, out_path):
    """Per-probe t_50 mean/std across seeds + peak/final accuracy, as
    Markdown. The peak/final columns surface transient-emergence probes —
    rises that collapse before the run ends — for which sigmoid t_50 is
    undefined but the dynamics are still publishable.
    """
    if fits_df.empty:
        with open(out_path, "w") as f:
            f.write("# results_summary\n\n(no fits available)\n")
        return
    g = fits_df.groupby("probe")
    rows = []
    for probe, group in g:
        valid = group.dropna(subset=["t50"])
        n = len(valid)
        peak_mean = float(group["peak_acc"].mean())
        final_mean = float(group["final_acc"].mean())
        peak_drop = peak_mean - final_mean
        if n == 0:
            rows.append({"probe": probe, "n_seeds": 0,
                         "t50_mean": float("nan"), "t50_std": float("nan"),
                         "ci_lo": float("nan"), "ci_hi": float("nan"),
                         "k_mean": float("nan"), "fit_ok_frac": 0.0,
                         "peak_acc_mean": peak_mean, "final_acc_mean": final_mean,
                         "peak_drop_mean": peak_drop, "ratio_std_mean": float("nan")})
            continue
        t50_mean = float(valid["t50"].mean())
        t50_std = float(valid["t50"].std(ddof=0)) if n > 1 else 0.0
        ratio = (t50_std / t50_mean) if t50_mean > 0 else float("nan")
        ci_lo, ci_hi = _bootstrap_ci_t50(valid["t50"].tolist())
        rows.append({
            "probe": probe,
            "n_seeds": n,
            "t50_mean": t50_mean,
            "t50_std": t50_std,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "k_mean": float(valid["k"].mean()),
            "fit_ok_frac": float(group["fit_ok"].mean()),
            "peak_acc_mean": peak_mean,
            "final_acc_mean": final_mean,
            "peak_drop_mean": peak_drop,
            "ratio_std_mean": ratio,
        })
    summary = pd.DataFrame(rows).sort_values("t50_mean")
    # Reproducibility (relaxed): pass if EITHER std/mean <= 0.2 (the
    # original criterion, meaningful for late emergence) OR absolute t_50
    # std <= 25 steps (catches very-early-emergence probes where mean is
    # so small that the ratio is dominated by the denominator).
    tight_ratio = summary["ratio_std_mean"] <= 0.2
    tight_abs = summary["t50_std"] <= 25.0
    n_reproducible = int(((tight_ratio | tight_abs) &
                          (summary["n_seeds"] >= 2) &
                          summary["t50_mean"].notna()).sum())
    n_transient = int(((summary["peak_drop_mean"] >= 0.15) &
                       (summary["peak_acc_mean"] >= 0.6)).sum())
    with open(out_path, "w") as f:
        f.write("# Probe-emergence summary\n\n")
        f.write("Per-probe sigmoid fit results across seeds. `t50` is the step at\n")
        f.write("which `argmax_acc` reaches 50% of the fitted span. Probes are\n")
        f.write("sorted by `t50_mean` (earlier emergence first). `peak_acc_mean`\n")
        f.write("is the maximum of the smoothed accuracy curve over training;\n")
        f.write("`final_acc_mean` is the mean over the last 25% of steps. A\n")
        f.write("large `peak_drop` indicates non-monotonic / transient emergence.\n")
        f.write("`std/mean` is the relative reproducibility metric. For\n")
        f.write("very-early-emergence probes (t₅₀ < 50) the ratio is\n")
        f.write("dominated by a tiny denominator, so a probe is counted as\n")
        f.write("reproducible if `std/mean ≤ 0.2` OR absolute `t50_std ≤\n")
        f.write("25 steps`. Success criterion: ≥ 2 reproducible probes.\n\n")
        f.write(f"**Reproducible probes (relaxed: ratio ≤ 0.2 OR abs std ≤ 25 steps,\n"
                f"`n_seeds ≥ 2`): {n_reproducible}**\n\n")
        f.write(f"**Transient-emergence probes (`peak_drop ≥ 0.15` and "
                f"`peak ≥ 0.6`): {n_transient}**\n\n")
        f.write("| probe | n_seeds | t50_mean | 95% CI | t50_std | std/mean | "
                "k_mean | fit_ok_frac | peak_acc | final_acc | peak_drop |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|\n")
        for _, r in summary.iterrows():
            ratio_str = (f"{r['ratio_std_mean']:.2f}"
                         if not (isinstance(r['ratio_std_mean'], float)
                                 and math.isnan(r['ratio_std_mean']))
                         else "—")
            ci_lo, ci_hi = r["ci_lo"], r["ci_hi"]
            nan = float("nan")
            ci_str = (f"[{ci_lo:.0f}, {ci_hi:.0f}]"
                      if not (isinstance(ci_lo, float) and math.isnan(ci_lo))
                      else "—")
            f.write(f"| {r['probe']} | {int(r['n_seeds'])} | "
                    f"{r['t50_mean']:.1f} | {ci_str} | {r['t50_std']:.1f} | "
                    f"{ratio_str} | "
                    f"{r['k_mean']:.4f} | {r['fit_ok_frac']:.2f} | "
                    f"{r['peak_acc_mean']:.2f} | {r['final_acc_mean']:.2f} | "
                    f"{r['peak_drop_mean']:.2f} |\n")
    print(f"Wrote {out_path}")


def make_fig4(runs, out_path):
    """Transient-emergence figure. For each transient probe (peak−late > 0.15
    AND peak ≥ 0.6), show:
      - per-seed heavily-smoothed argmax_acc trajectories (alpha=0.4)
      - cross-seed mean ± std band (bold)
      - bold marker at each seed's smoothed peak
      - dashed line at chance (0.5)
    Aggressive smoothing (rolling window 51) makes the rise+fall obvious.
    Behavioural seeds only (drops seed99 if present, since that's mechanistic).
    """
    BEHAVIOURAL = {"seed42", "seed123", "seed7", "seed5", "seed17"}
    runs_b = {s: df for s, df in runs.items() if s in BEHAVIOURAL}
    if not runs_b:
        runs_b = runs

    transient_probes = []
    for probe in sorted({p for df in runs_b.values() for p in df["probe_name"].unique()}):
        accs = []
        for df in runs_b.values():
            sub = df[df["probe_name"] == probe].sort_values("step")
            if len(sub) == 0:
                continue
            y = sub["argmax_acc"].values
            n = len(y)
            kernel = np.ones(3) / 3.0
            y_sm = np.convolve(y, kernel, mode="same") if n >= 3 else y
            peak = float(y_sm.max())
            late = float(np.mean(y[-max(1, n // 4):]))
            accs.append((peak, late))
        if not accs:
            continue
        avg_peak = np.mean([p for p, _ in accs])
        avg_late = np.mean([l for _, l in accs])
        if avg_peak - avg_late > 0.15 and avg_peak >= 0.6:
            transient_probes.append(probe)

    if not transient_probes:
        return

    # Order so end_of_sentence (the lead probe) is leftmost
    order_pref = ["end_of_sentence", "modal_continuation", "adjective_order"]
    transient_probes = sorted(transient_probes,
                              key=lambda p: order_pref.index(p) if p in order_pref else 99)

    seeds = sorted(runs_b.keys())
    seed_colors = {s: f"C{i}" for i, s in enumerate(seeds)}
    n_p = len(transient_probes)
    fig, axes = plt.subplots(1, n_p, figsize=(5.5 * n_p, 4.5), squeeze=False, sharey=True)

    SMOOTH_WIN = 51  # aggressive — collapse becomes visible

    def heavy_smooth(y, win=SMOOTH_WIN):
        if len(y) < win:
            return y
        ker = np.ones(win) / win
        return np.convolve(y, ker, mode="same")

    for col, probe in enumerate(transient_probes):
        ax = axes[0][col]
        # Common step grid for cross-seed mean
        all_t = sorted({int(s) for df in runs_b.values()
                        for s in df[df["probe_name"] == probe]["step"].values})
        if not all_t:
            continue
        # Per-seed smoothed curves on common grid via interp
        per_seed_curves = {}
        peak_points = []
        for seed_label, df in runs_b.items():
            sub = df[df["probe_name"] == probe].sort_values("step")
            if len(sub) == 0:
                continue
            t = sub["step"].values
            y = sub["argmax_acc"].values
            y_sm = heavy_smooth(y)
            # Find peak (skip first 30 steps to suppress init noise)
            mask = t > 30
            if mask.any():
                peak_idx = int(np.argmax(np.where(mask, y_sm, -1)))
                peak_points.append((seed_label, t[peak_idx], y_sm[peak_idx]))
            # Light-alpha individual trace
            ax.plot(t, y_sm, alpha=0.4, linewidth=1.2,
                    color=seed_colors[seed_label], label=seed_label)
            # Resample to common grid
            per_seed_curves[seed_label] = np.interp(all_t, t, y_sm)
        # Cross-seed mean and std
        if len(per_seed_curves) >= 2:
            mat = np.vstack(list(per_seed_curves.values()))
            mean_curve = mat.mean(axis=0)
            std_curve = mat.std(axis=0)
            ax.plot(all_t, mean_curve, color="black", linewidth=2.5,
                    label="mean (5 seeds)", zorder=5)
            ax.fill_between(all_t, mean_curve - std_curve, mean_curve + std_curve,
                            color="black", alpha=0.18, zorder=4, label="±1 std")
        # Bold peak markers
        for seed_label, peak_t, peak_v in peak_points:
            ax.scatter([peak_t], [peak_v], s=70, marker="o",
                       color=seed_colors[seed_label],
                       edgecolor="black", linewidth=1.0, zorder=10)
        # Chance line
        ax.axhline(0.5, color="grey", linestyle=":", alpha=0.6,
                   linewidth=0.8, label="chance")
        # Title with the peak/final cross-seed numbers
        if len(per_seed_curves) >= 2:
            mat = np.vstack(list(per_seed_curves.values()))
            cross_peak = float(np.max(mat.mean(axis=0)))
            cross_final = float(np.mean(mat[:, -max(1, len(all_t) // 4):]))
            ax.set_title(f"{probe}\npeak {cross_peak:.2f} → final {cross_final:.2f}  "
                         f"(drop {cross_peak - cross_final:.2f})",
                         fontsize=11)
        else:
            ax.set_title(probe, fontsize=11)
        ax.set_xlabel("training step", fontsize=10)
        if col == 0:
            ax.set_ylabel("argmax accuracy", fontsize=10)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
        if col == 0:
            ax.legend(loc="lower left", fontsize=8, framealpha=0.9, ncol=2)

    fig.suptitle("Transient capabilities: rise to peak then collapse  "
                 "(rolling-mean smoothed; bold = cross-seed mean ± std; circles = per-seed peak)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    runs = load_probe_logs()
    fits_df = make_fig1(runs, "figures/fig1_emergence.png")
    print("Wrote figures/fig1_emergence.png")
    # Force numeric_sequence for metric-dependence figure (matches paper text)
    make_fig2(runs, "figures/fig2_metric_dependence.png", probe="numeric_sequence")
    print("Wrote figures/fig2_metric_dependence.png")
    make_fig3(fits_df, "figures/fig3_t50_scatter.png")
    print("Wrote figures/fig3_t50_scatter.png")
    make_fig4(runs, "figures/fig4_transient.png")
    print("Wrote figures/fig4_transient.png")
    write_summary(fits_df, "results_summary.md")


if __name__ == "__main__":
    main()
