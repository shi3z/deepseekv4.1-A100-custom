#!/usr/bin/env bash
# Follow-up measurements for UD-IQ1_S on zeta: the VRAM edge (-ncmoe 9/8), prefill scaling (pp4096), and one real
# generation through llama-cli (sanity of the output, time to first token, interactive decode rate).
set -o pipefail
cd ~/bench-gguf/llama.cpp || exit 1
M=$HOME/models/gguf/UD-IQ1_S/DeepSeek-V4.1-Flash-1017-UD-IQ1_S-00001-of-00005.gguf
OUT=$HOME/bench-gguf/results2.log
T=16

sample() {
  local pid=$1 tag=$2 vmax=0 rmax=0 v r
  while kill -0 "$pid" 2>/dev/null; do
    v=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | awk -F', ' -v p="$pid" '$1==p{s+=$2} END{print s+0}')
    r=$(awk '/VmRSS/{print int($2/1024)}' /proc/"$pid"/status 2>/dev/null)
    [ "${v:-0}" -gt "$vmax" ] 2>/dev/null && vmax=$v
    [ "${r:-0}" -gt "$rmax" ] 2>/dev/null && rmax=$r
    sleep 2
  done
  echo "$tag peak_vram_mib=$vmax peak_rss_mib=$rmax" | tee -a "$OUT"
}

echo "=== $(date +%F_%H:%M:%S) follow-up start" | tee -a "$OUT"

# 1) how far the experts fit: 9 then 8 CPU layers (10 peaked at 92.7 of 97.9 GB)
for K in 9 8; do
  echo "=== $(date +%F_%H:%M:%S) ncmoe$K" | tee -a "$OUT"
  ./build/bin/llama-bench -m "$M" -t $T -p 512 -n 128 -r 2 -o md -ngl 99 -ncmoe $K > "$HOME/bench-gguf/bench_ncmoe$K.log" 2>&1 &
  bp=$!; sleep 5; sample "$bp" "ncmoe$K" & sp=$!
  wait "$bp"; echo "ncmoe$K rc=$?" | tee -a "$OUT"; wait "$sp" 2>/dev/null
  grep -E '^\|' "$HOME/bench-gguf/bench_ncmoe$K.log" | tail -2 | tee -a "$OUT"
  grep -iE 'out of memory|cuda error|failed' "$HOME/bench-gguf/bench_ncmoe$K.log" | head -2 | tee -a "$OUT"
done

# 2) prefill scaling at the best fitting split
echo "=== $(date +%F_%H:%M:%S) prefill scaling (ncmoe 10)" | tee -a "$OUT"
./build/bin/llama-bench -m "$M" -t $T -p 512,2048,8192 -n 0 -r 1 -o md -ngl 99 -ncmoe 10 > "$HOME/bench-gguf/bench_pp.log" 2>&1
grep -E '^\|' "$HOME/bench-gguf/bench_pp.log" | tail -4 | tee -a "$OUT"

# 3) one real generation: output sanity, time to first token, interactive rate
echo "=== $(date +%F_%H:%M:%S) llama-cli generation (ncmoe 10, 256 tokens)" | tee -a "$OUT"
./build/bin/llama-cli -m "$M" -t $T -ngl 99 -ncmoe 10 -c 4096 -n 256 --temp 0.3 -no-cnv \
  -p "Explain in three sentences why a mixture-of-experts model can have 748 billion parameters but only activate 16 billion per token." \
  > "$HOME/bench-gguf/cli.log" 2>&1 &
bp=$!; sleep 5; sample "$bp" cli & sp=$!
wait "$bp"; echo "cli rc=$?" | tee -a "$OUT"; wait "$sp" 2>/dev/null
echo "--- output ---" | tee -a "$OUT"
sed -n '/why a mixture-of-experts/,$p' "$HOME/bench-gguf/cli.log" | head -25 | tee -a "$OUT"
echo "--- timings ---" | tee -a "$OUT"
grep -E 'load time|prompt eval time|eval time|total time' "$HOME/bench-gguf/cli.log" | tee -a "$OUT"
echo "=== $(date +%F_%H:%M:%S) follow-up done" | tee -a "$OUT"
