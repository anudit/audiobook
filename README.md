# Audiobook

Converts EPUB to Audiobooks. Fully local.

## Setup
Make sure you have [uv](https://docs.astral.sh/uv/getting-started/installation/) installed.
```
uv venv
uv sync
```

## Create Audiobook
Edit the `input_epub` and `output_dir` in `convert.py` according to your file.

Run it.
```
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run audiobook.py
```
