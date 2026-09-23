# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

LeJEPA self-supervised learning (invariance + SIGReg loss, Balestriero & LeCun 2025) with a 3D ViT on microscopy volumes.

## Commands

Uses `uv` for everything; `mise.toml` loads `.env` (sets `LMD_DATA_ROOT`).

```sh
uv run pytest                                    # all tests (testpaths = lib/tests)
uv run pytest lib/tests/test_lejepa.py::test_lejepa_config   # single test
uv run python e00_basic.py                       # fzf picker over top-level fns
uv run python e00_basic.py run 3                 # call run(3) directly
```

Remote workflow (experiments run on the Janelia cluster, `login1.int.janelia.org:~/proj/ssl-lejepa`):
- VCS is **jj** (colocated with git). Remotes: `origin` (GitHub) and `janelia` (the cluster checkout).
- `./push.sh` pushes `main` to GitHub and has the cluster fetch it; `./push-j.sh` pushes straight to the cluster.
- `./run.sh e00_basic.py runmany` runs `uv run python ...` on the login node over ssh and appends the command to `experiment.log`.
- `./pull.sh` rsyncs the small result files (`*.json`, `*.log`, `profile.out`) from the cluster's `outdir/` into local `outdir/` with `--delete`. Large `profile.json` traces are not pulled.

## Experiment scripts (`eNN_*.py`)

A script is a flat module of top-level functions. `lib.util.call_entrypoint` / `pick_entrypoint` expose them as CLI subcommands, so every top-level function is an entrypoint. Args are positional and cast to int when possible.

The pattern in `e00_basic.py`:
- `Params` dataclass holds one run's config. `allparams()` returns the sweep as a list of `Params`, each with its own `savedir` (`outdir/e00/<exp>/d{i}/`). Each sweep is recorded in a commit message prefixed `exp:` that names the outdir. Results are recorded in commits prefixed `result:`.
- `run(n)` trains `allparams()[n]`. `runlsf(n)` wipes the savedir and `bsub`s `run n` to the `gpu_b300` queue (project `miaai`). `runmany()` submits the whole sweep.
- Outputs go to the savedir as append-only JSON-lines: `metrics.json` (loss per step), `performance.json` (unprofiled throughput window), `runs.json` (git commit + diff hash), plus `profile.json`/`profile.out` from `torch.profiler`. Each run's full working-copy diff is appended to `_diffs/diffs.json`, keyed by diff hash (`git_provenance`).
- `loadJsonTable(filename)` joins every savedir's rows with that run's `Params` into a pandas DataFrame. It asserts that any saved `params` still match `allparams()`, so the sweep definition has to stay in sync with the results on disk. `plotN()` functions plot the DataFrame with plotly.
- Training steps are wrapped in named `record_function` phases (`01_DATA_IO`, `03_H2D_TRANSFER`, `04_FORWARD_AND_LOSS`, …), but only while the profiler is active. The warmup → benchmark → profile step windows are set by `warmup_steps`/`benchmark_steps`/`profile_steps`.

Data comes from `lmd_catalog` (the volume catalog) → `.to_miao()` → `miao.VolumeDataset`, which yields dicts with an `"img"` tensor of shape `Batch C Z Y X`. Both packages are git dependencies (see `[tool.uv.sources]`).

## lib/

- `models/lejepa.py`: `Lejepa(config)` combines `ViT3DEncoder` → `ProjectorMLP` → `ViewMaker` → `lejepa_loss`. `forward(x)` makes global/local crops, encodes and projects each one, and returns `LejepaOutput` (a dict with attribute access: `loss`, `inv`, `sigreg`, `weighted_sigreg`). With `return_loss=False` or `views="none"` it returns embeddings instead. `LejepaConfig` has a hand-written `__init__` with alias properties (`dim`/`width`, `num_layers`/`n_layers`, …).
- `losses/sigreg.py`: `SIGReg` compares the empirical characteristic function along random 1D slices against the N(0,1) CF. The invariance term pulls every view toward the mean of the global views.
- `views/maker.py`: random-scale 3D crops with random flips, done per sample in Python loops.
- `profiler.py`: parses `profile.out`/chrome traces into bottleneck reports (`python -m lib.profiler`).
- `lib/__init__.py` monkeypatches `torch.Tensor.backwards` and a no-arg `torch.rand()`.

`lib/tests/test_training_profile.py` runs `e00_basic.run` end to end on a synthetic dataset by monkeypatching the module's globals (`allparams`, `lmd`, `VolumeDataset`, `Lejepa`, `LejepaConfig`, `repo_root`, `git_provenance`). Renaming those names in the experiment script breaks it.
