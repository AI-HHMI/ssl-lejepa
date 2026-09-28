# ssl-lejepa

LeJEPA self-supervised learning (invariance + SIGReg loss, Balestriero & LeCun 2025) with a 3D ViT on microscopy volumes.

## Commands

Uses `uv` for everything; `mise.toml` loads `.env` (sets `LMD_DATA_ROOT`).

```sh
uv run pytest                                    # all tests (testpaths = lib/tests)
uv run pytest lib/tests/test_lejepa.py::test_lejepa_config   # single test
uv run python e00_basic.py                       # fzf picker over top-level fns
uv run python e00_basic.py run 3                 # call run(3) directly (single process, no DDP)
uv run python e00_analysis.py loss_curves e00/nanhunt_flash   # analysis of any pulled sweep
```

Remote workflow (experiments run on the Janelia cluster, `login1.int.janelia.org:~/proj/ssl-lejepa`):
- VCS is **jj** (colocated with git). Remotes: `origin` (GitHub) and `janelia` (the cluster checkout).
- `./push.sh` pushes `main` to the `janelia` remote and runs `jj new main` on the cluster. It does not touch `origin`: moving `main@origin` makes those commits immutable in jj.
- `sh jrun.sh <bookmark> e00_basic.py runmany` pushes the bookmark to `janelia`, runs `jj new <bookmark>` on the cluster, then `uv run python e00_basic.py runmany` on the login node.
- `./pull.sh` rsyncs the small result files (`*.json`, `*.jsonl`, `*.log`, `*.png`, `profile.out`) from the cluster's `outdir/` into local `outdir/` with `--delete`. It skips large `profile.json` traces and `outdir/.trash/`.

## How this repo works

- **An experiment is one jj change**, described `exp: e00/<name>. <question>`. Its `allparams()` writes only to `outdir/e00/<name>/d{i}/`. The change ID stays the same while you fix it: cancel jobs, amend, resubmit. Each run records the commit hash it actually ran.
- **Keep experiment changes thin**: `allparams()` plus comments. Model, training and lib changes go in their own commits underneath. Concurrent experiments are sibling changes, merged into `main` soon after. On merge, `allparams()` is the only conflict: keep the newest experiment's.
- **Only committed code runs on the cluster.** `runlsf` asserts there are no uncommitted changes to tracked files, and snapshots the code plus its commit into `.tmpcode/<sweep>/` (`provenance.json`). Each job then records that commit in its `runs.json`, along with argv and LSF job ID. Repro for any run: *commit X, `e00_basic.py run n`*.
- **Resubmitting never destroys results.** `runlsf` moves an existing savedir to `outdir/.trash/<path>/<time>/` (`lib.util.trash`). Empty `.trash` by hand. Never edit or `rm` experiment dirs otherwise.
- **Every entrypoint call is logged** by `call_entrypoint`, to `outdir/_log/commands-<host>.jsonl`: time, host, argv, LSF job, commit, subject, dirty. That's one file per host, so cluster logs arrive with `./pull.sh`, and this machine's own file is excluded from `--delete`.
- **Two artifact places**:
  - `outdir/` is an exact mirror of the cluster's, written only by runs and never edited locally, so `--delete` is safe.
  - `results/` is local analysis output (figures, tables, summaries, screenshots) and is not committed. `results/perf_journey.html` is the one tracked file: the hand-written page that `perf_journey()` fills in.
- **Artifacts describe themselves, so analysis runs at HEAD.** `e00_analysis.py` reads only pulled artifacts, each run's saved `params`, never the current `allparams()`. Any analysis works on any past sweep, and fixing a figure means rerunning it at HEAD. Name analysis functions after what they study (`nanhunt_plot`, `flash_perf`). Old checkpoints must keep loading in `lib`, e.g. `PatchEmbed3d`'s Conv3d hook. Where that's impossible, branch off the old commit.
- **Replaying an experiment**: point a bookmark at its commit (`jj bookmark create replay -r <commit>`), then `sh jrun.sh replay e00_basic.py runmany`. It writes to the same savedirs, so the original results move to `.trash/`.

## Experiment scripts (`eNN_*.py`)

A script is a flat module of top-level functions. `lib.util.call_entrypoint` / `pick_entrypoint` expose them as CLI subcommands, so every top-level function is an entrypoint. Args are positional and cast to int when possible.

The pattern in `e00_basic.py`:
- `Params` dataclass holds one run's config. `allparams()` returns the sweep as a list of `Params`, each with its own `savedir` (`outdir/e00/<exp>/d{i}/`). Each sweep is recorded in a commit message prefixed `exp:` that names the outdir. Results are recorded in commits prefixed `result:`.
- `run(n)` trains `allparams()[n]`. `runlsf(n)` moves any old savedir to `outdir/.trash/` and `bsub`s it to the `gpu_b300` queue (project `miaai`) as `torchrun --standalone --nproc_per_node={n_gpus} e00_basic.py run n`, requesting `n_gpus` GPUs and `n_gpus * (n_workers + 1)` CPU slots. `runmany()` submits the whole sweep.
- Optimization knobs in `Params`:
  - `amp`: bf16 autocast for forward + loss; SIGReg stays fp32.
  - `compile`: `torch.compile(dynamic=True)` on the encoder.
  - `n_workers`: DataLoader workers per GPU.
  - `n_gpus`: DDP ranks on one node. `batch_size` and `n_workers` are per GPU.
  - `views`: `basic` or `displace` (see `lib/`).
- Under DDP, only rank 0 writes results and profiles. Throughput numbers are node totals. SIGReg and projector BatchNorm statistics are still per rank (local batch).
- Each run does `warmup_steps` (10), then an unprofiled benchmark window of `benchmark_steps` (50), then one profiler-warmup step and `profile_steps` (10) profiled steps, and then stops (`steps_per_epoch` = 71). Training steps are wrapped in named `record_function` phases (`01_DATA_IO`, `03_H2D_TRANSFER`, `04_FORWARD_AND_LOSS`, `05_BACKWARD`, `06_OPTIMIZER`, `07_LOGGING`), but only while the profiler is active.
- Outputs go to the savedir as append-only JSON-lines:
  - `metrics.json`: loss every 10 steps.
  - `performance.json`: from the benchmark window. `samples_per_second`, `tokens_per_second`, `tokens_per_sample`, `input_mvox_per_second`, `world_size`.
  - `runs.json`: commit, subject, dirty flag and diff hash (`code_provenance`), argv and LSF job ID.
  - `profile.json` / `profile.out`: from `torch.profiler`.
  - `trace_summary.json`: from `lib.util.trace_summary`. GPU busy fraction (union of kernel intervals) and per-phase host ms over the profiled steps.

- `e00_analysis.load_table(sweep, filename)` joins every run dir's rows with that run's saved `Params` (current defaults for fields added since) into a pandas DataFrame.
- Compare runs by `tokens_per_second`, not Mvox/s. Mvox/s counts the input patch, but tokens per sample change with the view scheme and patch-embedding rules. Profile-derived columns (GPU busy %, profiled step ms) depend on `profile_steps`; only compare them within a sweep.

Data comes from `lmd_catalog` (the volume catalog) → `.to_miao()` → `miao.VolumeDataset`, which yields dicts with an `"img"` tensor of shape `Batch C Z Y X`. Both packages are git dependencies (see `[tool.uv.sources]`). `lib/data.py:hemibrain_eb_config(split)` builds the MiaoConfig for the FlyEM hemibrain Ellipsoid Body train/val/test splits used in gary_comparison.

## lib/

- `models/lejepa.py`: `Lejepa(config)` combines `ViT3DEncoder` → `ProjectorMLP` → `ViewMaker` → `lejepa_loss`.
  - `forward(x)` makes global/local views, encodes and projects each one, and returns `LejepaOutput`. That's a dict with attribute access: `loss`, `inv`, `sigreg`, `weighted_sigreg`, plus `n_tokens` (encoder tokens this step).
  - With `return_loss=False` or `views="none"` it returns *projector* embeddings. For downstream use, take `model.encode(x)`, the backbone embedding.
  - `LejepaConfig` has a hand-written `__init__` with alias properties (`dim`/`width`, `num_layers`/`n_layers`, …).
- `encoders/vit3d.py`: `PatchEmbed3d` asserts the input is a multiple of the patch size, so there's no padding. Sin/cos position embeddings are built for whatever grid comes in.
- `losses/sigreg.py`: `SIGReg` compares the empirical characteristic function along random 1D slices against the N(0,1) CF. The invariance term pulls every view toward the mean of the global views.
- `views/maker.py`: `ViewMaker` makes crops without resizing, each flipped per sample and per axis. Each view is one batched advanced-indexing gather.
  - `basic`: each view draws a volume fraction from `global_scale` / `local_scale`. Sides are rounded to multiples of the patch size, which gives 9 distinct shapes for a 48×144×144 patch.
  - `displace`: fixed `global_size` / `local_size`. Each local sits inside a randomly chosen global of the same sample, at a uniformly random displacement. Every step has the same shapes and token count.
- `util.py`: entrypoint CLI and command log, `snapshot` / `code_provenance` / `trash`, `trace_summary`, `profile_device_ms`.
- `benchmark.py`: `Benchmark`, which owns the training loop's timed window (`performance.json`) and profiled window (`profile.json`/`.out`, `trace_summary.json`). `run()` calls `begin`/`phase`/`count`/`end` each step. Also holds `PEAK_BF16_TFLOPS`.
- `lib/__init__.py` monkeypatches a no-arg `torch.rand()`.

`lib/tests/test_training_profile.py` runs `e00_basic.run` end to end on a synthetic dataset by monkeypatching the module's globals (`allparams`, `lmd.get`, `MiaoConfig`, `VolumeDataset`, `Lejepa`, `LejepaConfig`, `repo_root`, `git_provenance`). Renaming those names in the experiment script breaks it.

## Throughput so far (B300, 12-layer width-512 ViT, 48×144×144 patches, `basic` views)

| change | ktok/s per GPU | notes |
|---|---|---|
| fp32 | ~270 | attention on fp32 mem-efficient kernel, ~64% of GPU time |
| + bf16 autocast | ~820 | cuDNN flash attention |
| + compile (dynamic) | ~930 | data-loader bound at ~170 samples/s with 4 workers |
| + 8 workers | ~1330 | 88% GPU busy; 16 workers no better |
| DDP, 8 GPUs | ~1010 (8.1M total) | 76% scaling efficiency; ~37 ms/step all-reduce plus per-rank view-size stragglers |

`outdir/e00/{amp,compile,compile-amp-tok_s,viewvec,workers,ddp}` hold these sweeps. Vectorizing `ViewMaker` didn't change throughput, because eager mode was already GPU-bound. `displace` views are meant to remove the DDP stragglers, but they're untested at scale.
