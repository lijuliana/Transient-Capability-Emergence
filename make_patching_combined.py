"""
Generate a combined 2-seed patching figure (fig_patching_combined.png).
Shows seed 99 and seed 17 patching curves for each transient probe.
Uses fine-checkpoint runs so that peak and final checkpoints are from
the same training trajectory for each seed.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Hard-coded results from running patching.py for seeds 99 and 17.
# Format: {seed: {probe: {depth: acc, 'peak_baseline': acc, 'final_step': N, 'peak_step': N}}}
# Fine-checkpoint runs (peak and final from the same training trajectory).
# seed 99: fine ckpts 150-180 + regular ckpts to step 2231 (final).
# seed 17: fine ckpts 1100-1160 + regular ckpts to step 2256 (final).
RESULTS = {
    99: {
        "end_of_sentence": {
            0: 0.333, 1: 0.300, 2: 0.533, 3: 0.533, 4: 0.300,
            "peak_baseline": 0.567, "peak_step": 1600, "final_step": 2231,
        },
        "modal_continuation": {
            0: 0.771, 1: 0.857, 2: 0.686, 3: 0.771, 4: 0.686,
            "peak_baseline": 0.771, "peak_step": 600, "final_step": 2231,
        },
        "adjective_order": {
            0: 0.700, 1: 0.767, 2: 0.733, 3: 0.667, 4: 0.700,
            "peak_baseline": 0.800, "peak_step": 1200, "final_step": 2231,
        },
    },
    17: {
        "end_of_sentence": {
            0: 0.383, 1: 0.183, 2: 0.383, 3: 0.650, 4: 0.267,
            "peak_baseline": 0.700, "peak_step": 1130, "final_step": 2256,
        },
        "modal_continuation": {
            0: 0.714, 1: 0.714, 2: 0.886, 3: 0.943, 4: 0.771,
            "peak_baseline": 0.714, "peak_step": 600, "final_step": 2256,
        },
        "adjective_order": {
            0: 0.800, 1: 0.767, 2: 0.967, 3: 0.867, 4: 0.800,
            "peak_baseline": 0.867, "peak_step": 1800, "final_step": 2256,
        },
    },
}

PROBES = ["end_of_sentence", "modal_continuation", "adjective_order"]
SEEDS = [99, 17]
N_LAYER = 4
SEED_COLORS = {99: "C0", 17: "C1"}
SEED_MARKERS = {99: "o", 17: "s"}

fig, axes = plt.subplots(1, len(PROBES), figsize=(5.5 * len(PROBES), 4.5), squeeze=False)
layer_labels = [f"L{i}" for i in range(N_LAYER)] + ["none\n(final)"]

for col, probe in enumerate(PROBES):
    ax = axes[0][col]
    depths = list(range(N_LAYER + 1))

    for seed in SEEDS:
        d = RESULTS[seed][probe]
        accs = [d.get(dep, float("nan")) for dep in depths]
        final_acc = d[N_LAYER]
        peak_acc = d["peak_baseline"]
        color = SEED_COLORS[seed]
        marker = SEED_MARKERS[seed]

        ax.plot(depths, accs, marker + "-", color=color, linewidth=1.8,
                markersize=7, label=f"seed {seed} (patched)", zorder=3)
        ax.axhline(peak_acc, color=color, linestyle="--", linewidth=1.0, alpha=0.6,
                   label=f"seed {seed} peak (step {d['peak_step']}): {peak_acc:.2f}")
        ax.axhline(final_acc, color=color, linestyle=":", linewidth=1.0, alpha=0.8,
                   label=f"seed {seed} final (step {d['final_step']}): {final_acc:.2f}")

    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.4)
    ax.set_xticks(depths)
    ax.set_xticklabels(layer_labels, fontsize=9)
    ax.set_xlabel("Patch depth (peak residual injected after this layer)", fontsize=9)
    ax.set_ylabel("argmax_acc" if col == 0 else "", fontsize=9)
    ax.set_ylim(0.0, 1.05)
    ax.set_title(probe.replace("_", " "), fontsize=11, fontweight="bold")
    ax.legend(fontsize=7, loc="best", ncol=1)
    ax.grid(alpha=0.3, axis="y")

fig.suptitle(
    "Activation patching (two seeds): peak-checkpoint residuals injected into final model\n"
    "Solid = patched final model accuracy; dashed = pure peak; dotted = pure final",
    fontsize=10,
)
fig.tight_layout(rect=[0, 0, 1, 0.91])
os.makedirs("figures", exist_ok=True)
out = "figures/fig_patching_combined.png"
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"Wrote {out}")
