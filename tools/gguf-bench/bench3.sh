#!/usr/bin/env bash
# One real generation through llama-cli at the best fitting split (-ncmoe 9): output sanity, prompt-eval and decode
# rates as llama.cpp reports them, peak VRAM / RSS of the process.
set -o pipefail
cd ~/bench-gguf/llama.cpp || exit 1
M=$HOME/models/gguf/UD-IQ1_S/DeepSeek-V4.1-Flash-1017-UD-IQ1_S-00001-of-00005.gguf
OUT=$HOME/bench-gguf/results3.log
echo "=== $(date +%F_%H:%M:%S) llama-cli generation (ncmoe 9, 256 tokens)" | tee -a "$OUT"
./build/bin/llama-cli -m "$M" -t 16 -ngl 99 -ncmoe 9 -c 4096 -n 256 --temp 0.3 -st --no-warmup \
  -p "Explain in three sentences why a mixture-of-experts model can have 748 billion parameters but only activate 16 billion per token." \
  > "$HOME/bench-gguf/cli.log" 2>&1 &
bp=$!
vmax=0; rmax=0
while kill -0 "$bp" 2>/dev/null; do
  v=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | awk -F', ' -v p="$bp" '$1==p{s+=$2} END{print s+0}')
  r=$(awk '/VmRSS/{print int($2/1024)}' /proc/"$bp"/status 2>/dev/null)
  [ "${v:-0}" -gt "$vmax" ] 2>/dev/null && vmax=$v
  [ "${r:-0}" -gt "$rmax" ] 2>/dev/null && rmax=$r
  sleep 2
done
wait "$bp"; rc=$?
echo "cli rc=$rc peak_vram_mib=$vmax peak_rss_mib=$rmax" | tee -a "$OUT"
echo "--- model output ---" | tee -a "$OUT"
sed -n '/activate 16 billion per token/,$p' "$HOME/bench-gguf/cli.log" | grep -vE '^(llama_|load_|print_info|init:|main:|system_info)' | head -20 | tee -a "$OUT"
echo "--- timings ---" | tee -a "$OUT"
grep -E 'load time|prompt eval time|^ *eval time|total time' "$HOME/bench-gguf/cli.log" | tee -a "$OUT"
echo "=== $(date +%F_%H:%M:%S) done" | tee -a "$OUT"
