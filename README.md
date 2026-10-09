# Setup

Install on the Janelia cluster, re-run a recent anaylsis, then a full blown old experiment sweep.

```sh
# 1. Clone and build the .venv that every job runs from.
ssh login1.int.janelia.org
cd ~/my_projects/
git clone git@github.com:AI-HHMI/ssl-lejepa.git
cd ssl-lejepa/
uv sync --all-extras
source .venv/bin/activate
## Scoring (experiment.score) runs mia-evals from its own checkout at ~/proj/mia-evals (MIA_EVALS):
## git clone git@github.com:AI-HHMI/mia-evals.git ~/proj/mia-evals && (cd ~/proj/mia-evals && uv sync)

# 2. Run a recent experiment analysis
python -m reports.viewsizes_pca    ## -> results/e00/viewsizes-pca/report.html

# 3. Run an old experiment
## Check out an old experiment commit, i.e. one with `exp: e00/<experiment-name>` in the description.
git checkout 16ca5785 ## e00/viewsizes-v2/

# 3. Run it. This will spawn multiple LSF/bsub jobs.
python experiment.py runall
## more recently we renamed `runall -> submitall`

# 4. Wait and watch the jobs
bjobs -w
## experiments write to outdir/
## analysis reads from outdir/ and writes to results/
```
