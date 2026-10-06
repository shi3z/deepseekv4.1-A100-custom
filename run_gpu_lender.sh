#!/usr/bin/env bash
# run_gpu_lender.sh: lend GPUs 0/1 to the H3 ComfyUI fleet while the DeepSeek LLM (run_128k_4slots.sh) is idle.
# Runs in tmux session "gpu-lender". See scripts/gpu_lender.py for the rules and env knobs.
#   tmux new-session -d -s gpu-lender "./run_gpu_lender.sh"
cd /mnt/ssdraid/git/deepseekv4.1
export LEND_IDLE_MIN=${LEND_IDLE_MIN:-20}        # minutes without any LLM request before lending
export LEND_MIN_QUEUE=${LEND_MIN_QUEUE:-1}       # fleet queue/ depth needed to lend
export LEND_GPUS=${LEND_GPUS:-0,1}               # at most 2 of the LLM's GPUs
export LEND_EMPTY_QUEUE_MIN=${LEND_EMPTY_QUEUE_MIN:-10}
exec python3 scripts/gpu_lender.py 2>&1 | tee -a results/gpu_lender_stdout.log
