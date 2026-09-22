set -e
jj git push --remote janelia -b main
ssh -o ConnectTimeout=15 login1.int.janelia.org \
    "cd ~/proj/ssl-lejepa/ && jj new main"
