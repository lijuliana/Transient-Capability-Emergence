# Transient Capability Interp

Mechanistic-interpretability experiments on a small (11.5M parameter)
transformer trained on a TinyStories-like corpus, focused on a
sentence-final-period prediction capability that the model briefly
acquires (peak ~74% accuracy) and then loses (final ~24%) across all
five seeds we ran.

The repo contains the training pipeline, fourteen narrowly scoped
behavioural probes, sigmoid-fitting machinery for cross-seed
emergence-time analysis, and a set of mechanistic analyses
(activation patching, head ablation, OV cosine, direct logit
attribution, attention entropy) that localise the collapse.

## Setup

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run prepare.py    # downloads training data + trains BPE tokenizer (~2 min)
```

A single training run takes roughly five minutes on an RTX 4090.

## Training

```bash
AUTORESEARCH_SEED=42 uv run train.py > run_seed42.log 2>&1
```

The `AUTORESEARCH_SEED` environment variable selects the seed.
Probe accuracy is logged to `probe_log_seed{SEED}.tsv` every five
optimizer steps. The five behavioural seeds reported in the analysis
below are 42, 123, 7, 5, and 17.

## Behavioural analysis

`analyze.py` reads all `probe_log_seed*.tsv` files in the working
directory, fits per-probe sigmoids across seeds, and produces
emergence-time tables and figures.

```bash
uv run analyze.py
```

Outputs:

- `figures/fig1_emergence.png` — per-probe accuracy trajectories
  with sigmoid fits, sorted by mean t₅₀
- `figures/fig2_metric_dependence.png` — argmax vs. logprob_diff on
  `numeric_sequence`
- `figures/fig3_t50_scatter.png` — per-seed t₅₀ scatter
- `figures/fig4_transient.png` — peak-then-collapse trajectories on
  the three transient probes
- `figures/fig5_loss_and_transient.png` — training loss overlaid
  with the transient curves
- `results_summary.md` — per-probe mean / std / 95% CI for t₅₀

## Mechanistic analysis

The mechanistic experiments require fine-grained checkpoint saving
around the probe-log peak. Two seeds (17, 99) are used. Each script
operates on the saved checkpoints and produces a numerical results
file plus a figure under `figures/`.

| Script | Question answered |
| --- | --- |
| `patching.py` | Where in the forward pass is the recoverable information lost? |
| `head_ablation.py` | Which attention head, ablated, recovers the most accuracy? |
| `head_patching.py` | Is the critical head's output sufficient on its own? |
| `ov_analysis.py` | Which head's OV circuit changed most peak→final? |
| `dla_ablation.py` | What is the per-item logit contribution shift? |
| `dla_multiprobe.py` | Does the same head shift its contribution across probes? |
| `adjorder_dla.py` | First-token DLA on `adjective_order` |
| `attn_entropy.py` | Did the critical head's QK routing change? |
| `make_loss_figure.py` | Generates `fig5_loss_and_transient.png` |
| `make_patching_combined.py` | Generates `fig_patching_combined.png` |

## Probe set

Fourteen probes, six lexical and eight syntactic. Each probe is a
list of (prefix, correct, distractor) triples; accuracy is the
fraction of items for which `logp(correct) > logp(distractor)`.

| Type | Probes |
| --- | --- |
| Lexical / associative | `determiner_a_an`, `comparative_than`, `proper_noun_completion`, `pronoun_gender`, `numeric_sequence`, `common_idiom` |
| Syntactic / structural | `end_of_sentence`, `modal_continuation`, `adjective_order`, `past_tense_consistency`, `subj_verb_agreement`, `relative_clause_agreement`, `close_quote`, `reflexive_pronoun` |

Probe definitions live at the top of `train.py` in the `PROBES`
dictionary; each probe has 30–86 hand-curated and templated items.

## Headline result

Three syntactic probes show reproducible peak-then-collapse
trajectories across all five seeds (`end_of_sentence`,
`modal_continuation`, `adjective_order`). For `end_of_sentence`,
residual-stream patching localises the change to blocks 0–2, head
ablation isolates a single critical head per seed (B0H0 in seed 17,
B0H1 in seed 99 — the head index does not replicate), and a
probe-specificity control rules out general head degradation: the
same critical head, at the same final checkpoint, helps the stable
`numeric_sequence` probe while harming `end_of_sentence`.

## Layout

```
.
├── prepare.py                  data download, BPE tokenizer, dataloader
├── train.py                    model, optimizer, training loop, probes
├── analyze.py                  behavioural analysis, sigmoid fitting
├── patching.py                 residual-stream activation patching
├── head_ablation.py            per-head zero-ablation
├── head_patching.py            head-output patching
├── ov_analysis.py              OV cosine and trajectory
├── dla_ablation.py             direct logit attribution
├── dla_multiprobe.py           cross-probe DLA
├── adjorder_dla.py             first-token DLA on adjective_order
├── attn_entropy.py             attention entropy / QK stability
├── make_loss_figure.py         figure helper
├── make_patching_combined.py   figure helper
├── analysis.ipynb              exploratory notebook
├── figures/                    generated figures (PNG)
├── pyproject.toml              deps
└── uv.lock
```

## Hardware

The training pipeline runs on Apple Silicon (MPS) and on NVIDIA GPUs.
On an RTX 4090, a single 5-minute training run reaches roughly
3,946–4,869 optimizer steps. Set `AUTORESEARCH_SKIP_MACOS_CHECK=1`
on Linux GPU pods to skip the macOS gate at the top of `train.py`
and `prepare.py`.

## License

MIT.
