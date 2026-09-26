# Push a jj bookmark to the cluster and check it out there; optionally run a python command on the login node.
#   sh jrun.sh <branch>                          push only
#   sh jrun.sh <branch> e00_basic.py runmany     push, then `uv run python e00_basic.py runmany` on the cluster
set -e
BRANCH=${1:?usage: sh jrun.sh <branch> [script.py args...]}
shift

jj git push --remote janelia -b "$BRANCH"
ssh -o ConnectTimeout=15 login1.int.janelia.org \
    "cd ~/proj/ssl-lejepa/ && jj new $(printf '%q' "$BRANCH")"

[ $# -eq 0 ] && exit 0

CMD=$(printf '%q ' "$@")
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$BRANCH] $CMD" >> experiment.log
ssh -o ConnectTimeout=15 login1.int.janelia.org "cd ~/proj/ssl-lejepa/ && uv run python $CMD"
