#!/usr/bin/env bash
# Build llama.cpp with CUDA for this machine: RTX PRO 6000 Blackwell = compute 12.0 (sm_120), CUDA 12.9 toolkit.
# cmake comes from a pip venv (no passwordless sudo here, so no apt).
set -eo pipefail
export PATH=/usr/local/cuda-12.9/bin:$PATH
export CUDACXX=/usr/local/cuda-12.9/bin/nvcc
cd ~/bench-gguf
if [ ! -x venv/bin/cmake ]; then
  python3 -m venv venv
  venv/bin/pip -q install --upgrade pip
  venv/bin/pip -q install cmake ninja
fi
export PATH=$PWD/venv/bin:$PATH
[ -d llama.cpp ] || git clone -q --depth 1 https://github.com/ggml-org/llama.cpp
cd llama.cpp
git log -1 --format="commit %h %ci %s"
cmake -B build -G Ninja -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 32 --target llama-bench llama-cli llama-server
ls -la build/bin/llama-bench build/bin/llama-cli
echo "=== build done $(date +%F_%H:%M:%S)"
#### download
#!/usr/bin/env bash
# Download the smallest full-model quantization of unsloth/DeepSeek-V4.1-Flash-1017-GGUF (UD-IQ1_S, 5 shards, 217 GiB)
# plus the DSpark draft model, into ~/models/gguf. hf CLI from the SHIGERU training venv (huggingface_hub 2.2.0).
set -o pipefail
export HF_HOME=~/.cache/huggingface
HF=~/shigeru/open-mercury/.venv-train/bin/hf
cd ~/models/gguf
echo "=== $(date +%F_%H:%M:%S) start UD-IQ1_S"
$HF download unsloth/DeepSeek-V4.1-Flash-1017-GGUF --include "UD-IQ1_S/*" --local-dir ~/models/gguf 2>&1 | tail -5
echo "=== $(date +%F_%H:%M:%S) UD-IQ1_S rc=$?"
du -sh ~/models/gguf/UD-IQ1_S 2>/dev/null
echo "=== $(date +%F_%H:%M:%S) start dspark"
$HF download unsloth/DeepSeek-V4.1-Flash-1017-GGUF --include "dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf" --local-dir ~/models/gguf 2>&1 | tail -3
echo "=== $(date +%F_%H:%M:%S) dspark rc=$?"
echo "=== $(date +%F_%H:%M:%S) download done"
df -h ~ | tail -1
