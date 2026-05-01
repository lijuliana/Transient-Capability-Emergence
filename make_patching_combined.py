"""
Generate a combined 2-seed patching figure (fig_patching_combined.png).
For each transient probe, show patching recovery as a function of injection depth
for both mechanistic seeds. The recovery is the difference between the patched
accuracy and the pure-final-model baseline. Y-axis is zoomed to the relevant
range and the recovery is annotated on the figure for the eos best-depth result.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Hard-coded results from running patching.py for seeds 99 and 17.
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
SEEDS = [17, 99]
N_LAYER = 4
SEED_COLORS = {99: "#2E86AB", 17: "#E63946"}
SEED_MARKERS = {99: "o", 17: "s"}

# Per-probe y-axis range to zoom in on the interesting region
Y_RANGES = {
    "end_of_sentence": (0.10, 0.85),
    "modal_continuation": (0.55, 1.00),
    "adjective_order": (0.55, 1.00),
}

fig, axes = plt.subplots(1, len(PROBES), figsize=(6.0 * len(PROBES), 4.8), squeeze=False)
layer_labels = ["L0", "L1", "L2", "L3", "none\n(=final)"]

for col, probe in enumerate(PROBES):
    ax = axes[0][col]
    depths = list(range(N_LAYER + 1))

    for seed in SEEDS:
        d = RESULTS[seed][probe]
        accs = [d[dep] for dep in depths]
        final_acc = d[N_LAYER]
        peak_acc = d["peak_baseline"]
        color = SEED_COLORS[seed]
        marker = SEED_MARKERS[seed]

        # Patched accuracy curve (the headline)
        ax.plot(depths, accs, marker + "-", color=color, linewidth=2.4,
                markersize=10, label=f"seed {seed} patched", zorder=4,
                markeredgecolor="white", markeredgewidth=1.0)

        # Final-model baseline as a dashed horizontal (the reference for "recovery")
        ax.axhline(final_acc, color=color, linestyle="--", linewidth=1.4,
                   alpha=0.65, zorder=2,
                   label=f"seed {seed} final baseline ({final_acc:.2f})")

        # Annotate the best-depth recovery for end_of_sentence (the headline)
        if probe == "end_of_sentence":
            best_depth = max(range(N_LAYER), key=lambda i: accs[i])
            best_acc = accs[best_depth]
            recovery = best_acc - final_acc
            ax.annotate(
                f"+{recovery * 100:.1f} pp",
                xy=(best_depth, best_acc),
                xytext=(best_depth + 0.15, best_acc + 0.04),
                fontsize=11, fontweight="bold", color=color,
                arrowprops=dict(arrowstyle="-", color=color, lw=1, alpha=0.6),
            )

    # Pretty x-axis
    ax.set_xticks(depths)
    ax.set_xticklabels(layer_labels, fontsize=10)
    ax.set_xlabel("Inject peak residual after this layer", fontsize=10)
    if col == 0:
        ax.set_ylabel("argmax accuracy on probe", fontsize=10)
    ax.set_ylim(*Y_RANGES[probe])
    ax.set_title(probe.replace("_", " "), fontsize=12, fontweight="bold")
    # Tight legend, only one column
    ax.legend(fontsize=8, loc="lower left", framealpha=0.9, ncol=1)
    ax.grid(alpha=0.3, axis="y")
    # Highlight the depth-3 column on the eos panel since that's the headline
    if probe == "end_of_sentence":
        ax.axvspan(2.5, 3.5, alpha=0.10, color="gold", zorder=0)

fig.suptitle(
    "Activation patching: depth-3 injection (peak residual after block 2 → final block) "
    "recovers most of the lost end_of_sentence accuracy in both mechanistic seeds",
    fontsize=11.5, y=0.99,
)
fig.tight_layout(rect=[0, 0, 1, 0.93])
os.makedirs("figures", exist_ok=True)
out = "figures/fig_patching_combined.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Wrote {out}")
