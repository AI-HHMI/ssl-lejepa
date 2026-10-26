set -eo pipefail
mkdir -p outdir/
# -i itemizes changes (--no-group: cluster groups can't be set locally, so every file looked changed); drop directory lines (cd = new dir, .d = dir timestamp) so only file changes print.
# .trash/ holds resubmitted runs' old results (lib.util.trash), cluster-only. _log/ holds the cluster's
# commands.jsonl (lib.util.log_command; older per-host commands-<host>.jsonl). Nothing local writes outdir/,
# so --delete is safe. *.zarr/ (probe affinity artifacts, GBs each) stay on the cluster.
# The cluster's outdir lives on /nrs (home quota); the checkout's outdir/ must be a symlink to it.
# oc-rsync 0.6.4 sometimes deadlocks at startup (client and server both idle, before any file list): a re-run gets past it.
# --timeout=15 makes it give up after 15 s without data, and we retry 3 times.
pull() {
  oc-rsync -azi --no-group --delete \
    --exclude='.trash/' \
    --exclude='*.zarr/' \
    --include='*/' \
    --include='**/profile.out' \
    --exclude='**/profile.json' \
    --include='*.json' \
    --include='*.jsonl' \
    --include='*.png' \
    --include='*.log' \
    --exclude='*' \
    --prune-empty-dirs \
    --rsync-path=oc-rsync \
    --timeout=15 \
    -e "ssh -o ConnectTimeout=15" \
    'login1.int.janelia.org:/nrs/miaai/broaddusc/proj/ssl-lejepa/outdir/' outdir/ \
    | { grep -v '^[.c]d' || true; }
}
for try in 1 2 3; do
  pull && exit 0
  echo "pull.sh: attempt $try of 3 failed, retrying" >&2
done
exit 1
