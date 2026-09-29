set -eo pipefail
mkdir -p outdir/
# -i itemizes changes (--no-group: cluster groups can't be set locally, so every file looked changed); drop directory lines (cd = new dir, .d = dir timestamp) so only file changes print.
# .trash/ holds resubmitted runs' old results (lib.util.trash), cluster-only. _log/ holds the cluster's
# commands.jsonl (lib.util.log_command; older per-host commands-<host>.jsonl). Nothing local writes outdir/,
# so --delete is safe.
oc-rsync -azi --no-group --delete \
  --exclude='.trash/' \
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
  -e "ssh -o ConnectTimeout=15" \
  'login1.int.janelia.org:~/proj/ssl-lejepa/outdir/' outdir/ \
  | { grep -v '^[.c]d' || true; }
