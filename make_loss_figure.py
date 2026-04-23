"""
Extract smoothed training loss from run logs and plot alongside
transient-probe accuracy curves. Produces figures/fig5_loss_and_transient.png.
"""
import os
import re
import glob
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def extract_loss(log_path):
    """Return {step: loss} from a run log file."""
    data = {}
    with open(log_path, "r", errors="replace") as f:
        content = f.read()
    for part in re.split(r"[\r\n]", content):
        m = re.search(r"step (\d+) \(\d+\.\d+%\) \| loss: ([0-9.]+)", part)
        if m:
            data[int(m.group(1))] = float(m.group(2))
    return data


def smooth(values, window=50):
    """Simple moving average with reflect padding."""
    arr = np.array(values, dtype=float)
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")


def make_fig5(out_path="figures/fig5_loss_and_transient.png"):
    # Load loss data from all seed logs
    seed_logs = {
        "seed42": "run_seed42.log",
        "seed123": "run_seed123.log",
        "seed7": "run_seed7.log",
        "seed5": "run_seed5.log",
    }
    # Only include logs that exist
    seed_logs = {k: v for k, v in seed_logs.items() if os.path.exists(v)}

    transient_probes = ["end_of_sentence", "modal_continuation", "adjective_order"]
    probe_logs = {
        path: label
        for path, label in [
            ("probe_log_seed42.tsv", "seed42"),
            ("probe_log_seed123.tsv", "seed123"),
            ("probe_log_seed7.tsv", "seed7"),
            ("probe_log_seed5.tsv", "seed5"),
        ]
        if os.path.exists(path)
    }

    seed_colors = {
        "seed42": "C0",
        "seed123": "C1",
        "seed7": "C2",
        "seed5": "C3",
        "seed17": "C4",
    }

    n_rows = 2
    n_cols = 1 + len(transient_probes)  # loss + one per transient probe
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows),
                             squeeze=False)

    # --- Row 0: Training loss curves ---
    ax_loss = axes[0][0]
    for seed_label, log_path in seed_logs.items():
        loss_dict = extract_loss(log_path)
        if not loss_dict:
            continue
        steps = sorted(loss_dict)
        losses = [loss_dict[s] for s in steps]
        # Smooth with window 30
        sm = smooth(losses, 30)
        ax_loss.plot(steps, sm, alpha=0.8, linewidth=1.2,
                     color=seed_colors.get(seed_label, "gray"), label=seed_label)
    ax_loss.set_xlabel("training step")
    ax_loss.set_ylabel("EMA training loss")
    ax_loss.set_title("Training loss (smoothed)")
    ax_loss.grid(alpha=0.3)
    ax_loss.legend(fontsize=8)
    # Hide unused row-0 panels for transient probes
    for col in range(1, n_cols):
        axes[0][col].set_visible(False)

    # --- Row 1: Per-transient-probe accuracy with initialization-spike note ---
    dfs = {}
    for path, label in probe_logs.items():
        dfs[label] = pd.read_csv(path, sep="\t")

    for col_i, probe in enumerate(transient_probes):
        ax = axes[1][col_i + 1]
        for seed_label, df in dfs.items():
            sub = df[df["probe_name"] == probe].sort_values("step")
            if len(sub) == 0:
                continue
            t = sub["step"].values
            y = sub["argmax_acc"].values
            # Smooth with window 5 (only for early-step noise)
            y_sm = np.convolve(y, np.ones(5) / 5, mode="same") if len(y) >= 5 else y
            # Find peak on smoothed (exclude first 10 steps — init noise)
            mask_no_init = t > 10
            if mask_no_init.sum() > 0:
                peak_idx = np.argmax(y_sm[mask_no_init])
                # Map back to full index
                full_idxs = np.where(mask_no_init)[0]
                peak_step_val = t[full_idxs[peak_idx]]
            else:
                peak_step_val = t[np.argmax(y_sm)]
            ax.plot(t, y, alpha=0.2, linewidth=0.8,
                    color=seed_colors.get(seed_label, "gray"))
            ax.plot(t, y_sm, alpha=0.85, linewidth=1.3,
                    color=seed_colors.get(seed_label, "gray"), label=seed_label)
            ax.axvline(peak_step_val, color=seed_colors.get(seed_label, "gray"),
                       linestyle="--", alpha=0.5, linewidth=0.8)
        ax.set_title(probe.replace("_", "\n"), fontsize=9)
        ax.set_xlabel("training step")
        ax.set_ylabel("argmax acc" if col_i == 0 else "")
        ax.axhline(0.5, color="k", linestyle=":", alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
        if col_i == 0:
            ax.legend(fontsize=7, loc="upper right")

    # Hide loss-row placeholder for probe panel
    axes[1][0].set_visible(False)

    fig.suptitle(
        "Training dynamics: smoothed loss (top-left) and transient-probe accuracy\n"
        "(smoothed, per seed; dashed = peak step excluding init noise)",
        fontsize=10
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    make_fig5()
