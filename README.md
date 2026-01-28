# token-filtering

Code for running token-level data filtering experiments.

## Setup

We recommend using [`uv`](https://docs.astral.sh/uv/) for setup. We use the `safety-tooling` module for API tasks (e.g. getting Claude ground-truth labels for probe training). To get started, after installing, run
```
git submodule update --init --recursive
uv sync
```

You can then run any script using `uv run <script_name.py>`.

## Acknowledgements

Transformer training/model code is mostly forked from [nanoGPT](https://github.com/karpathy/nanoGPT) and [modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt/tree/master), with minor modifications.