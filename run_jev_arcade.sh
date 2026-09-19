#!/bin/bash
# Launch Jev Arcade 20 Mini-Games Web UI Server
set -e

PORT=${1:-8888}
HOST="0.0.0.0"

export DSV41_SERVER_URL="http://127.0.0.1:8000"
export TYPESAFE_API_KEY="apikey_213088f1322b5885496b999b159a452ee161_0920d4a50e53f268ee01015aac9f679af6d74bb8d467312bc1bbaeb51ad85cad"

echo "=========================================================="
echo " 🎮 Launching Jev Arcade 20 Server on http://${HOST}:${PORT}"
echo "    Tailnet URL for iPad: http://100.126.237.55:${PORT}"
echo "    Local DeepSeek Backend: ${DSV41_SERVER_URL}"
echo "=========================================================="

exec uvicorn apps.jev_arcade.main:app --host "${HOST}" --port "${PORT}"
