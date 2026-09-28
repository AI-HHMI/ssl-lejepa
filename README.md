# ssl-lejepa

LeJEPA self-supervised learning (invariance + SIGReg loss, Balestriero & LeCun 2025) with a 3D ViT on microscopy volumes.

## Commands

Uses `uv` for everything; `mise.toml` loads `.env` (sets `LMD_DATA_ROOT`).

```sh
uv run pytest                                    # all tests (testpaths = lib/tests)
uv run pytest lib/tests/test_lejepa.py::test_lejepa_config   # single test
uv run python experiment.py                       # fzf picker over top-level fns (experiment code)
uv run python experiment.py run 3                 # call run(3) directly (single process, no DDP)
uv run python analysis.py loss_curves e00/nanhunt_flash   # analysis of any pulled sweep
```

Remote workflow (experiments run on the Janelia cluster, `login1.int.janelia.org:~/proj/ssl-lejepa`):
- VCS is **jj** (colocated with git). Remotes: `origin` (GitHub) and `janelia` (the cluster checkout).
- `./push.sh` pushes `main` to the `janelia` remote and runs `jj new main` on the cluster. It does not touch `origin`: moving `main@origin` makes those commits immutable in jj.
- `sh jrun.sh <bookmark> experiment.py runmany` pushes the bookmark to `janelia`, runs `jj new <bookmark>` on the cluster, then `uv run python experiment.py runmany` on the login node. For experiments the bookmark is `exp` (see below).
- `./pull.sh` rsyncs the small result files (`*.json`, `*.jsonl`, `*.log`, `*.png`, `profile.out`) from the cluster's `outdir/` into local `outdir/` with `--delete`. It skips large `profile.json` traces and `outdir/.trash/`.

## How this repo works

Every piece of code is either **remote** or **local**, and depends either on **mutable `lib/`** or **only on `outdir/`** data.

| class | runs | depends on | examples | rule |
|---|---|---|---|---|
| **experiment** | remote (LSF) | `lib/` (treat all of it as mutable, `util.py` included) | `experiment.py`: `run`, `pca`, `replay_bad_batch`; submitted by `runlsf`/`pcalsf`/`replaylsf` | Only for **this commit's own `allparams()`**, run from its committed snapshot. Training, inference (`pca`) and replays of an old sweep need time travel: `jj new <its commit>`. |
| **analysis** | local | only `outdir/` (saved artifacts); never builds a `Lejepa`: imports neither `experiment.py` nor `lib/` model code, only `lib.util`'s entrypoint CLI | `analysis.py`: `loss_curves`, `nanhunt_plot`, `flash_perf`, `perf_journey`, `check_runs` | Runs at HEAD on **any** past sweep. It depends on `lib/` only transitively, through `outdir/`, which is an append-only log. |
| **glue** | either | neither | `jrun.sh`, `pull.sh`, `gpufree.sh` | |

- **An experiment is one jj change**, described `exp: e00/<name>. <question>`. Its `allparams()` writes only to `outdir/e00/<name>/d{i}/`. The change ID stays the same while you fix it: cancel jobs, amend, resubmit. Each run records the commit hash it actually ran.
- **Keep experiment changes thin, and don't merge them.** An `exp:` change holds `allparams()` plus comments. Model, training, lib and analysis changes go in their own commits on `main`, underneath. `main` carries only those, never experiments.
  - Each experiment stays an **unmerged leaf** on the `main` commit it branched from. Concurrent experiments are sibling leaves. Before leaving one, move any other code it picked up onto `main` (`jj split`, then rebase).
  - **Find experiments by description**, not bookmarks: `jj log -r 'description(glob:"exp: e00/b300*")'`.
  - **Never `jj abandon` an experiment that has runs.** Leaves stay visible heads, which is what keeps description search working. An abandoned commit still resolves by ID, but revsets no longer see it. If finished leaves clutter `jj log`, hide them from the default view in `.jj/repo/config.toml`, e.g. `[revsets] log = "@ | ancestors(immutable_heads().., 2) | trunk() | ~description(glob:'exp:*')"` (untested).
  - **Don't amend an `exp:` change after its jobs ran.** Fixes before submission are fine. After that, make a new change on top, so the commit `runs.json` records is the one you find.
- **Remote code gets the concurrency guards; local code gets none.**
  - Only committed code is submitted: `bsub` jobs come from `runlsf` (train), `pcalsf` (redo PCA maps) and `replaylsf` (replay a bad batch). Each first calls `assert_committed()`.
  - Each job runs a snapshot of the code plus its commit (`.tmpcode/<sweep>/`, `provenance.json`), and records that commit, argv and LSF job ID in `runs.json`. Repro for any run: *commit X, `experiment.py run n`*.
  - Every CLI call of an experiment script (`experiment.py <fn> ...`, via `log_command` in its `__main__`) is logged to `outdir/_log/commands-<host>.jsonl`, one file per host. This covers the login-node `runmany`/`pcalsf`/`replaylsf` calls and each job's `run`/`pca`/`replay_bad_batch`. Analysis doesn't log.
  - Resubmitting moves the old savedir to `outdir/.trash/<path>/<time>/` (`lib.util.trash`) instead of deleting it. Empty `.trash` by hand.
- **`pca` and `replay_bad_batch` assert they're working on this commit's own runs.** `pca(n)` checks the checkpoint's saved params equal `allparams()[n]`. `replay_bad_batch` checks the dump sits in one of this commit's savedirs.
- **Two artifact places**:
  - `outdir/` is an exact mirror of the cluster's append-only run dirs, written only by remote runs, never locally. That's why `./pull.sh --delete` is safe. Never edit or `rm` experiment dirs.
  - `results/` is local analysis output (figures, tables, summaries, screenshots) and is not committed. `results/perf_journey.html` is the one tracked file: the hand-written page that `perf_journey()` fills in.
- **Analysis reads each run's saved `params`, never the current `allparams()`**, so fixing a figure means rerunning it at HEAD. Each experiment gets one entrypoint named after its sweep (`e00/b300-compile` → `analysis.py e00_b300_compile`) that makes all its figures and tables (`results/<sweep>/`). It's built from generic pieces: `bench`, `loss_curves`, `load_table`. `lib/tests/test_analysis.py` checks `analysis.py` imports neither `experiment.py` nor anything from `lib/` except `lib.util`.
- **Launching and replaying**: one reusable bookmark, `exp`, marks what's being launched: `jj bookmark set exp -r <change>`, then `sh jrun.sh exp experiment.py runmany`. No per-experiment bookmarks. Once pushed, a commit exists both locally and in the cluster's repo, even after `exp` moves on. Replaying an old experiment is the same with its commit, e.g. `jj bookmark set exp -r <commit id from runs.json>`. It writes to the same savedirs, so the original results move to `.trash/`.

## experiment.py

A script is a flat module of top-level functions. `lib.util.call_entrypoint` / `pick_entrypoint` expose them as CLI subcommands, so every top-level function is an entrypoint. Args are positional and cast to int when possible.

The pattern in `experiment.py`:
- `Params` dataclass holds one run's config. `allparams()` returns the sweep as a list of `Params`, each with its own `savedir` (`outdir/e00/<exp>/d{i}/`). Each sweep is recorded in a commit message prefixed `exp:` that names the outdir. Results are recorded in commits prefixed `result:`.
- `run(n)` trains `allparams()[n]`. `runlsf(n)` moves any old savedir to `outdir/.trash/` and `bsub`s it to the `gpu_b300` queue (project `miaai`) as `torchrun --standalone --nproc_per_node={n_gpus} experiment.py run n`, requesting `n_gpus` GPUs and `n_gpus * (n_workers + 1)` CPU slots. `runmany()` submits the whole sweep.
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
  - `runs.json`: one row per experiment function that ran on this dir (`record`: `fn` = `run` / `pca` / `replay_bad_batch`). Each row has commit, subject, dirty flag and diff hash (`code_provenance`), argv and LSF job ID. This overlaps with `outdir/_log/commands-<host>.jsonl` on purpose: the run dir is self-contained, and the global log covers calls with no run dir (`runmany`, `pcalsf`).
    Rows from before 2026-09-28 have only `commit_id` + `diff_hash`. Their uncommitted diffs are in `outdir/_log/diffs-legacy.json`, keyed by `diff_hash`: the frozen old `_diffs/diffs.json` from the cluster.
  - `profile.json` / `profile.out`: from `torch.profiler`.
  - `trace_summary.json`: from `lib.util.trace_summary`. GPU busy fraction (union of kernel intervals) and per-phase host ms over the profiled steps.

- `analysis.load_table(sweep, filename)` joins every run dir's rows with that run's saved `Params` (current defaults for fields added since) into a pandas DataFrame.
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
- `util.py`: entrypoint CLI and command log, `snapshot` / `code_provenance` / `assert_committed` / `trash`, `trace_summary`.
- `benchmark.py`: `Benchmark`, which owns the training loop's timed window (`performance.json`) and profiled window (`profile.json`/`.out`, `trace_summary.json`). `run()` calls `begin`/`phase`/`count`/`end` each step. Also holds `PEAK_BF16_TFLOPS`.
- `lib/__init__.py` monkeypatches a no-arg `torch.rand()`.

`lib/tests/test_training_profile.py` runs `experiment.run` end to end on a synthetic dataset by monkeypatching the module's globals (`allparams`, `lmd.get`, `MiaoConfig`, `VolumeDataset`, `Lejepa`, `LejepaConfig`, `repo_root`, `git_provenance`). Renaming those names in the experiment script breaks it.

## Throughput so far (B300, 12-layer width-512 ViT, 48×144×144 patches, `basic` views)

| change | ktok/s per GPU | notes |
|---|---|---|
| fp32 | ~270 | attention on fp32 mem-efficient kernel, ~64% of GPU time |
| + bf16 autocast | ~820 | cuDNN flash attention |
| + compile (dynamic) | ~930 | data-loader bound at ~170 samples/s with 4 workers |
| + 8 workers | ~1330 | 88% GPU busy; 16 workers no better |
| DDP, 8 GPUs | ~1010 (8.1M total) | 76% scaling efficiency; ~37 ms/step all-reduce plus per-rank view-size stragglers |

`outdir/e00/{amp,compile,compile-amp-tok_s,viewvec,workers,ddp}` hold these sweeps. Vectorizing `ViewMaker` didn't change throughput, because eager mode was already GPU-bound. `displace` views are meant to remove the DDP stragglers, but they're untested at scale.
