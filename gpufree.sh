# Free GPUs per LSF GPU queue on the Janelia cluster, from your laptop: sh gpufree.sh
# FREE_GPUS: unallocated GPUs on hosts accepting jobs / all GPUs in the queue's host groups.
# EMPTY_NODES: hosts with no jobs at all (what an 8-GPU full-node job needs).
# GPU_HOSTS: hosts with at least one free GPU. FREE_SLOTS: free CPU slots on open hosts.
# PEND_SLOTS: slots requested by pending jobs in the queue.
ssh -o ConnectTimeout=15 login1.int.janelia.org 'bash -s' <<'EOF'
printf "%-12s %9s %11s %11s %10s %9s\n" QUEUE FREE_GPUS EMPTY_NODES GPU_HOSTS FREE_SLOTS PEND_SLOTS
for q in $(bqueues -w | awk '$1 ~ /^gpu_/ && $1 !~ /_parallel$/ {print $1}'); do
  groups=$(bqueues -l "$q" | awk '/^HOSTS:/ {for (i = 2; i <= NF; i++) {g = $i; sub(/\/.*$/, "", g); print g}}')
  pend=$(bqueues -w "$q" | awk 'NR == 2 {print $9}')
  bhosts -o "host_name status ngpus ngpus_alloc max njobs" -noheader $groups | awk -v q="$q" -v pend="$pend" '
    { total += $3 }
    $2 == "ok" {
      free += $3 - $4; slots += $5 - $6
      if ($4 < $3) hosts++
      if ($4 == 0 && $6 == 0 && $3 > 0) empty++
    }
    END { printf "%-12s %4d/%-4d %11d %11d %10d %9s\n", q, free, total, empty, hosts, slots, pend }'
done
EOF
