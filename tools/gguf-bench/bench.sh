#!/usr/bin/env bash
# Benchmark unsloth/DeepSeek-V4.1-Flash-1017-GGUF UD-IQ1_S (748B-A16B, 40 layers, 384 experts / 6 used, 217 GiB)
# on zeta: RTX PRO 6000 Blackwell 97.9 GB VRAM, Ryzen 9 9950X (16 cores), 123 GB RAM, model on NVMe.
# The model is larger than VRAM + RAM, so it is always mmap'd: whatever does not fit is read from NVMe per token.
# Per configuration: llama-bench prefill (pp512) and decode (tg128), plus peak VRAM of our own process, peak RSS
# and the page cache high-water mark.
set -o pipefail
cd ~/bench-gguf/llama.cpp || exit 1
M=$HOME/models/gguf/UD-IQ1_S/DeepSeek-V4.1-Flash-1017-UD-IQ1_S-00001-of-00005.gguf
B=./build/bin/llama-bench
OUT=$HOME/bench-gguf/results.log
T=16

sample() {            # sample <pid> <tag>: peak VRAM of that pid, peak RSS, page-cache high-water, every 2 s
  local pid=$1 tag=$2 vmax=0 rmax=0 cmax=0 v r c
  while kill -0 "$pid" 2>/dev/null; do
    v=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | awk -F', ' -v p="$pid" '$1==p{s+=$2} END{print s+0}')
    r=$(awk '/VmRSS/{print int($2/1024)}' /proc/"$pid"/status 2>/dev/null)
    c=$(awk '/^Cached:/{print int($2/1024)}' /proc/meminfo)
    [ "${v:-0}" -gt "$vmax" ] 2>/dev/null && vmax=$v
    [ "${r:-0}" -gt "$rmax" ] 2>/dev/null && rmax=$r
    [ "${c:-0}" -gt "$cmax" ] 2>/dev/null && cmax=$c
    sleep 2
  done
  echo "$tag peak_vram_mib=$vmax peak_rss_mib=$rmax peak_pagecache_mib=$cmax" | tee -a "$OUT"
}

run() {               # run <tag> <llama-bench args...>
  local tag=$1; shift
  echo "=== $(date +%F_%H:%M:%S) $tag :: $*" | tee -a "$OUT"
  "$B" -m "$M" -t "$T" -p 512 -n 128 -r 2 -o md "$@" > "$HOME/bench-gguf/bench_$tag.log" 2>&1 &
  local bp=$!
  sleep 5
  sample "$bp" "$tag" &
  local sp=$!
  wait "$bp"; local rc=$?
  wait "$sp" 2>/dev/null
  echo "$tag rc=$rc  load_s=$(grep -oE 'load time *= *[0-9.]+' "$HOME/bench-gguf/bench_$tag.log" | tail -1)" | tee -a "$OUT"
  grep -E '^\|' "$HOME/bench-gguf/bench_$tag.log" | tail -4 | tee -a "$OUT"
  grep -iE 'error|failed|cannot|unsupported' "$HOME/bench-gguf/bench_$tag.log" | head -3 | tee -a "$OUT"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader | tee -a "$OUT"
}

echo "=== $(date +%F_%H:%M:%S) bench start" | tee -a "$OUT"
free -g | sed -n 2p | tee -a "$OUT"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader | tee -a "$OUT"
ls -la "$M" | awk '{print "first shard", $5, $9}' | tee -a "$OUT"
du -sh "$(dirname "$M")" | tee -a "$OUT"

# 1) every expert tensor on the CPU side (unsloth's recipe for small VRAM): attention + dense layers on the GPU
run cmoe -ngl 99 -ncmoe 40
# 2) experts of the first K layers on CPU, the rest on GPU: fill the 98 GB of VRAM
for K in 24 18 14 10; do
  run "ncmoe$K" -ngl 99 -ncmoe "$K"
done
# 3) no GPU at all, decode only and short: reference for how much the GPU is worth here
echo "=== $(date +%F_%H:%M:%S) cpu_only :: -ngl 0 -n 32" | tee -a "$OUT"
"$B" -m "$M" -t "$T" -p 0 -n 32 -r 1 -ngl 0 -o md > "$HOME/bench-gguf/bench_cpu.log" 2>&1 &
bp=$!; sleep 5; sample "$bp" cpu_only & sp=$!
wait "$bp"; echo "cpu_only rc=$?" | tee -a "$OUT"; wait "$sp" 2>/dev/null
grep -E '^\|' "$HOME/bench-gguf/bench_cpu.log" | tail -3 | tee -a "$OUT"
echo "=== $(date +%F_%H:%M:%S) bench done" | tee -a "$OUT"
