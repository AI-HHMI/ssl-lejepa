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

Design an experiment: `experiment.py` is only its table of runs (`BASE.but(...)`, one line per run, named; `H200` and `B300` are whole
nodes) and `lib/jobs.py` the rest. `python experiment.py -h` lists the tasks of the pipeline with the hardware each needs and the data
it reads (by role: train, val, fit, test, tune, the parts every dataset of `lib/data.py` has).
```sh
# [On the Janelia cluster, from the experiment's commit.]
sh jrun.sh exp experiment.py submitall        # every run not submitted yet
sh jrun.sh exp experiment.py lsf benchmark mask7   # one task of one run (or of a spec json)
```
