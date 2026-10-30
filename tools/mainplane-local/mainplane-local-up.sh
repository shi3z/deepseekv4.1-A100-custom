#!/usr/bin/env bash
# Loopback-only mainplane harness for the dsv41 server (no Cloudflare tunnel, no pointer.mainplane.ai registration):
# the pointer calls go to a dead proxy and fail fast; loopback traffic (the harness API on 8080, dsv41 on 8000) bypasses
# the proxy.  TS_DEBUG_USE_DERP_HTTP: the mesh relay is the harness's own /derp on http://localhost:8080 (local build
# patch in pkg/coordinator relayMap).  Logs: /var/lib/mainplane-server/harness.log
export HTTPS_PROXY=http://127.0.0.1:9 HTTP_PROXY=http://127.0.0.1:9 NO_PROXY=127.0.0.1,localhost
export TS_DEBUG_USE_DERP_HTTP=1
exec "$HOME/bin/mainplane-server" up /var/lib/mainplane-server/config.json
