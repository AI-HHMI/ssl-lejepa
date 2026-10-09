# Setup

Install on the Janelia cluster, re-run an old experiment, then do analysis.

```sh
# Clone and build the .venv that every job runs from.
ssh login1.int.janelia.org
cd ~/my_projects/
git clone git@github.com:AI-HHMI/ssl-lejepa.git
cd ssl-lejepa/
uv sync --all-extras
source .venv/bin/activate

# Run an old experiment
# Check out an old experiment commit, i.e. one with `exp: e00/<experiment-name>` in the description.
git checkout 16ca5785 ## e00/viewsizes-v2/
python experiment.py runmany
git checkout main
# This will spawn multiple LSF/bsub jobs.
# Experiments write to outdir/ .
# Wait and watch the jobs.
bjobs -w
# analysis reads from outdir/ and writes to results/
python reports/viewsizes_v2.py  ## -> results/e00/viewsizes-v2/report.html

```
