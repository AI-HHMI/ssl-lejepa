set -e
mkdir -p outdir/
oc-rsync -avz --include='*/' --include='*.jsonl' --exclude='*' --prune-empty-dirs --rsync-path=oc-rsync -e "ssh -o ConnectTimeout=15" 'login1.int.janelia.org:~/proj/ssl-lejepa/outdir/' outdir/
