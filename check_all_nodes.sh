#!/usr/bin/env bash
# Usage:
#   ./slurm_free.sh
#   scontrol show node -o | ./slurm_free.sh

if [ -t 0 ]; then
  # No stdin: fetch from scontrol
  SRC_CMD="scontrol show node -o"
else
  # Read from stdin
  SRC_CMD="cat -"
fi

$SRC_CMD | awk '
function tres_val(s, k,   m,v) {
  # Extract value for key k (e.g., "gres/gpu") from a TRES string
  if (match(s, k"=[^,]+", m)) {
    v = m[0]
    sub(k"=", "", v)
    gsub(/[^0-9]/, "", v)   # keep digits only
    return v+0
  }
  return 0
}
function to_mib(x,   n,u) {
  # Convert mem strings like 256G, 515095M to MiB; plain numbers are MiB
  if (x ~ /[KkMmGgTt]$/) {
    n = substr(x, 1, length(x)-1) + 0
    u = substr(x, length(x), 1)
    if (u ~ /[Kk]/) return n/1024.0
    if (u ~ /[Mm]/) return n
    if (u ~ /[Gg]/) return n*1024.0
    if (u ~ /[Tt]/) return n*1024.0*1024.0
  }
  return x + 0
}
{
  node=""; cpualloc=0; cpueff=0; cputot=0; realmem=0; allocmem=0; cfgtres=""; alloctres=""
  for (i=1; i<=NF; i++) {
    if (index($i,"=")) {
      pos = index($i,"=")
      key = substr($i,1,pos-1)
      val = substr($i,pos+1)
      if (key=="NodeName")     node=val
      else if (key=="CPUAlloc") cpualloc=val+0
      else if (key=="CPUEfctv") cpueff=val+0
      else if (key=="CPUTot")   cputot=val+0
      else if (key=="RealMemory") realmem=val+0
      else if (key=="AllocMem")   allocmem=val+0
      else if (key=="CfgTRES")    cfgtres=val
      else if (key=="AllocTRES")  alloctres=val
    }
  }

  total_cpus = (cpueff>0 ? cpueff : cputot)
  free_cpus  = total_cpus - cpualloc
  if (free_cpus < 0) free_cpus = 0

  # Memory: prefer AllocMem/RealMemory (MiB). Fallback to AllocTRES mem if needed.
  if (allocmem <= 0 && alloctres != "") {
    if (match(alloctres, /mem=[^,]+/)) {
      allocmem = to_mib(substr(alloctres, RSTART+4, RLENGTH-4))
    }
  }
  free_mib = realmem - allocmem
  if (free_mib < 0) free_mib = 0
  total_gib = realmem/1024.0
  free_gib  = free_mib/1024.0

  # GPUs from TRES
  total_gpus = tres_val(cfgtres, "gres/gpu")
  alloc_gpus = tres_val(alloctres, "gres/gpu")
  free_gpus  = total_gpus - alloc_gpus
  if (free_gpus < 0) free_gpus = 0

  printf "%-8s  FreeCPUs=%3d/%d   FreeCPUMem=%.1fGiB/%.1fGiB   FreeGPUs=%d/%d\n",
         node, free_cpus, total_cpus, free_gib, total_gib, free_gpus, total_gpus
}'
