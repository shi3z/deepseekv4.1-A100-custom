#!/usr/bin/env bash
# =============================================================================
# Claude Code runner backed by DeepSeek-V4.1 via LiteLLM
#
# Usage:
#   ./claude-dsv41.sh [Claude Code options...]
#
# Examples:
#   ./claude-dsv41.sh                  # Interactive Claude Code session
#   ./claude-dsv41.sh -p "Fix the bug" # Non-interactive prompt
#
# Environment variables you can override:
#   MODEL_NAME      Model identifier in LiteLLM (default: deepseek-v4.1-flash)
#   PROXY_PORT      LiteLLM proxy port (default: 8101)
#   LITELLM_CONFIG  Path to dsv41-litellm.yaml (default: same directory)
#   DSV41_PORT      DeepSeek-V4.1 engine port (default: 8000)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${LITELLM_CONFIG:-$SCRIPT_DIR/dsv41-litellm.yaml}"
PROXY_PORT="${PROXY_PORT:-8101}"
DSV41_PORT="${DSV41_PORT:-8000}"
MODEL_NAME="${MODEL_NAME:-deepseek-v4.1-flash}"
LOG="${LITELLM_LOG:-$HOME/.local/state/claude-dsv41/litellm-$PROXY_PORT.log}"

export PATH="$HOME/.local/bin:$PATH"

for tool in python3 curl litellm claude flock; do
    if ! command -v "$tool" >/dev/null; then
        echo "ERROR: Required tool '$tool' is not installed or not in PATH." >&2
        exit 1
    fi
done

# Check if DeepSeek-V4.1 backend is responding
if ! curl --noproxy '*' -fsS --max-time 3 "http://127.0.0.1:$DSV41_PORT/health" >/dev/null 2>&1; then
    echo "ERROR: DeepSeek-V4.1 server is not responding at http://127.0.0.1:$DSV41_PORT." >&2
    echo "Please start the inference server first (e.g., ./run_server_batched.sh)." >&2
    exit 1
fi

mkdir -p -- "$(dirname -- "$LOG")"
exec 9>"$LOG.lock"
flock -w 100 9

proxy_alive() {
    curl --noproxy '*' -fsS --max-time 2 "http://127.0.0.1:$PROXY_PORT/health/liveliness" >/dev/null 2>&1
}

# Start LiteLLM proxy if not already running
if ! proxy_alive; then
    echo "Starting LiteLLM proxy at http://127.0.0.1:$PROXY_PORT (log: $LOG)..." >&2
    nohup litellm --config "$CONFIG" --host 127.0.0.1 --port "$PROXY_PORT" >"$LOG" 2>&1 </dev/null 9>&- &
    proxy_pid=$!
    ready=0
    for ((i=0; i<60; i++)); do
        if proxy_alive; then ready=1; break; fi
        kill -0 "$proxy_pid" 2>/dev/null || break
        sleep 1
    done
    if (( ! ready )); then
        echo "ERROR: LiteLLM failed to start within 60s. Check log at $LOG" >&2
        exit 1
    fi
fi

flock -u 9
exec 9>&-

# Configure Anthropic API emulation for Claude Code
export ANTHROPIC_BASE_URL="http://127.0.0.1:$PROXY_PORT"
export ANTHROPIC_AUTH_TOKEN="sk-local"
unset ANTHROPIC_API_KEY CLAUDE_CODE_OAUTH_TOKEN
unset CLAUDE_CODE_USE_BEDROCK CLAUDE_CODE_USE_VERTEX CLAUDE_CODE_USE_FOUNDRY

export ANTHROPIC_MODEL="$MODEL_NAME"
export ANTHROPIC_DEFAULT_OPUS_MODEL="$MODEL_NAME"
export ANTHROPIC_DEFAULT_SONNET_MODEL="$MODEL_NAME"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="$MODEL_NAME"
export ANTHROPIC_SMALL_FAST_MODEL="$MODEL_NAME"
export CLAUDE_CODE_SUBAGENT_MODEL="$MODEL_NAME"

export CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1
export CLAUDE_CODE_DISABLE_UNKNOWN_MODEL_WINDOW_ENFORCEMENT=1
export CLAUDE_CODE_MAX_CONTEXT_TOKENS="${CLAUDE_CODE_MAX_CONTEXT_TOKENS:-786432}"
export API_TIMEOUT_MS="${API_TIMEOUT_MS:-600000}"

echo "Claude Code -> LiteLLM (http://127.0.0.1:$PROXY_PORT) -> DeepSeek-V4.1 (http://127.0.0.1:$DSV41_PORT)" >&2

if [ $# -eq 0 ]; then
    exec claude --model "$MODEL_NAME" -r
else
    exec claude --model "$MODEL_NAME" "$@"
fi
