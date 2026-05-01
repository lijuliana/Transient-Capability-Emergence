"""
Extract smoothed training loss from run logs and plot alongside
transient-probe accuracy curves. Produces figures/fig5_loss_and_transient.png.

Layout: 1x4 row — training loss + 3 transient probes side by side.
No wasted whitespace.
"""
import os
import re
import glob
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
    arr = np.array(values, dtype=float)
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")


def make_fig5(out_path="figures/fig5_loss_and_transient.png"):
    seed_logs = {
        "seed42": "run_seed42.log",
        "seed123": "run_seed123.log",
        "seed7": "run_seed7.log",
        "seed5": "run_seed5.log",
        "seed17": "run_seed17.log",
    }
    seed_logs = {k: v for k, v in seed_logs.items() if os.path.exists(v)}

    transient_probes = ["end_of_sentence", "modal_continuation", "adjective_order"]
    probe_logs = {
        path: label
        for path, label in [
            ("probe_log_seed42.tsv", "seed42"),
            ("probe_log_seed123.tsv", "seed123"),
            ("probe_log_seed7.tsv", "seed7"),
            ("probe_log_seed5.tsv", "seed5"),
            ("probe_log_seed17.tsv", "seed17"),
        ]
        if os.path.exists(path)
    }

    seed_colors = {
        "seed42": "C0", "seed123": "C1", "seed7": "C2",
        "seed5": "C3", "seed17": "C4",
    }

    # 1 x 4 layout: loss + three transient probes
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.6), sharey=False)

    # Panel 0: loss
    ax_loss = axes[0]
    for seed_label, log_path in seed_logs.items():
        loss_dict = extract_loss(log_path)
        if not loss_dict:
            continue
        steps = sorted(loss_dict)
        losses = [loss_dict[s] for s in steps]
        sm = smooth(losses, 30)
        ax_loss.plot(steps, sm, alpha=0.85, linewidth=1.5,
                     color=seed_colors.get(seed_label, "gray"), label=seed_label)
    ax_loss.set_xlabel("training step", fontsize=10)
    ax_loss.set_ylabel("EMA training loss", fontsize=10)
    ax_loss.set_title("Training loss (smoothed)", fontsize=11, fontweight="bold")
    ax_loss.grid(alpha=0.3)
    ax_loss.legend(fontsize=8, loc="upper right")

    # Panels 1-3: transient probes
    dfs = {label: pd.read_csv(path, sep="\t") for path, label in probe_logs.items()}
    SMOOTH_WIN = 51

    for col_i, probe in enumerate(transient_probes):
        ax = axes[col_i + 1]
        for seed_label, df in dfs.items():
            sub = df[df["probe_name"] == probe].sort_values("step")
            if len(sub) == 0:
                continue
            t = sub["step"].values
            y = sub["argmax_acc"].values
            y_sm = (np.convolve(y, np.ones(SMOOTH_WIN) / SMOOTH_WIN, mode="same")
                    if len(y) >= SMOOTH_WIN else y)
            ax.plot(t, y_sm, alpha=0.85, linewidth=1.5,
                    color=seed_colors.get(seed_label, "gray"), label=seed_label)
        ax.set_title(probe.replace("_", " "), fontsize=11, fontweight="bold")
        ax.set_xlabel("training step", fontsize=10)
        if col_i == 0:
            ax.set_ylabel("argmax accuracy", fontsize=10)
        ax.axhline(0.5, color="grey", linestyle=":", alpha=0.5, linewidth=0.8)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)

    fig.suptitle(
        "Training loss and transient-probe accuracy across five seeds  "
        "(rolling-mean smoothed; chance = 0.5)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    make_fig5()
