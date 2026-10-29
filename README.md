# Setup

Clone the repo then sync and activate the virtual env.

```sh
# [Any machine local or cluster.]
git clone git@github.com:AI-HHMI/ssl-lejepa.git
cd ssl-lejepa/
uv sync --all-extras # reports require extras
source .venv/bin/activate
```

Pull all results into outdir/ and serve the overview.

```sh
# [On a Local machine with browser for viewing.]
sh pull.sh
sh serve.sh
# Improve and rerender an experiment report, e.g.
python reports/viewsizes_v2.py
```
Every report of training runs has a "Loss trajectories" section: `analysis_plots.trunk_loss_curves(...)` with `["loss", "inv", "sigreg"]`
(a run without a term, e.g. an MAE's inv, is left out of that card), plus any other logged term the experiment varies.

Repro an experiment e.g.
```sh
# [On the Janelia cluster.]
# Check out an old experiment commit, i.e. one with `exp: e00/<experiment-name>` in the description.
git checkout 16ca5785 ## e00/viewsizes-v2/
python experiment.py runmany ## sometimes it's "runall" or "submitall"
git checkout main

# Watch the spawned jobs.
bjobs -w
```

Design an experiment: `experiment.py` is declarative: `BASE` (a `Train`: model, encoder, data, optim, speed, profile) and a
table of runs, one `Trunk` per line (`replace(BASE, ...)`, named). `tasks.py` is what a run does: plain
functions with JSON-able arguments, `trunk(t)`, `anneal(t, fork_point)`, `finetune(t, branch_dir)` and `score(t, branch_dir)`; the hardware and wall time of each is one block at the top of `tasks.py`. A task is a job's entrypoint: it reads
and writes files and checkpoints, submits the next task with `go(task, savedir, hardware, minutes, **args)` and calls the library
(`lib/`: straight-line code on loaded models and a `Train`, no callbacks) in its own process. A trunk pretrains in one job and forks a
branch at its start and after every segment; a branch is `anneal` (the trunk's GPUs), then `finetune` (pca and tiled inference on one RTX 6000), then `score`, a CPU job after
which the trunk keeps `best.pt`. `go` writes the arguments and resources to `<savedir>/<task>.json`, the job file that
`python experiment.py <task> <job file>` loads by the task's type hints. `lib/task_utils.py` is the LSF helpers, `lib/task_entry.py` the command line (`main`, `submitall`, `resubmit_task`, ...) and `lib/scoring.py` the mia-evals scoring of a branch and the trunk's best-keeping. Words: *experiment* is
one `experiment.py`; a *run* a line of its table; a *task* a function of `tasks.py`; a *job* a task submitted to LSF.
`python experiment.py -h` lists the tasks.
```sh
# [On the Janelia cluster, from the experiment's commit.]
sh jrun.sh exp experiment.py submitall        # every run not submitted yet
sh jrun.sh exp experiment.py resubmit_task start mask7   # restart one run fresh (or redo one task from its job file, e.g. a branch's score.json; a run is restarted with the experiment's `start`)
```
