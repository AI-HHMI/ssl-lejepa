set -eo pipefail
mkdir -p outdir/
# -i itemizes changes (--no-group: cluster groups can't be set locally, so every file looked changed); drop directory lines (cd = new dir, .d = dir timestamp) so only file changes print.
# .trash/ holds resubmitted runs' old results (lib.util.trash), cluster-only. _log/ has one commands-<host>.jsonl
# per host (lib.util.log_command); excluding this machine's own keeps --delete off it (oc-rsync ignores 'P').
oc-rsync -azi --no-group --delete \
  --exclude="_log/commands-$(hostname -s).jsonl" \
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
