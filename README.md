# Transient Capability Interp

Mechanistic interpretability experiments on a small transformer
trained on TinyStories-like text. Work in progress.

## Run

```bash
uv sync
uv run prepare.py
AUTORESEARCH_SEED=42 uv run train.py
uv run analyze.py
```

The `AUTORESEARCH_SKIP_MACOS_CHECK=1` env var bypasses the macOS
gate when running on a Linux GPU pod.
