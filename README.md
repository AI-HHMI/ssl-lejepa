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

Design an experiment: `experiment.py` is declarative: `BASE` (a `Train`: model, encoder, data, optim, hardware, speed, profile) and a
table of runs, one `Trunk` per line (`BASE.but(...)`, named; `H200` and `B300` are whole nodes). `tasks.py` is what a run does: three
tasks, `trunk`, `branch` and `score`, each beside the spec dataclass of its name that it takes (JSON-able; each holds its parent's:
`Score(branch)`, `Branch(trunk)`, `Trunk(train)`). A task is a job's entrypoint: it loads its spec, reads and writes files and
checkpoints, submits the next tasks with `go()` and calls the library (`lib/`: straight-line code on loaded models and specs, no
callbacks) in its own process. A trunk pretrains in one job and forks a branch at its start and after every segment; a branch anneals,
runs pca and tiled inference, and goes to `score`, a CPU job after which the trunk keeps `best.pt`. `lib/task_utils.py` is the LSF and
registry helpers. Words: *experiment* is one `experiment.py`; a *run* a line of its table; a *task* a function of `tasks.py`; a *job* a
task submitted to LSF. `python experiment.py -h` lists the tasks with their hardware.
```sh
# [On the Janelia cluster, from the experiment's commit.]
sh jrun.sh exp experiment.py submitall        # every run not submitted yet
sh jrun.sh exp experiment.py lsf benchmark mask7   # one task of one run (or of a spec json, e.g. a branch's score.json)
```
