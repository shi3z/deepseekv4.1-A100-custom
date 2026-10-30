#!/usr/bin/env bash
# The admin worker of the loopback-only harness, as the official install runs it: root for the mesh TUN, code as the
# operator named in /var/lib/mainplane/operator (shi3z), join token in /var/lib/mainplane/join.
# Logs: /var/lib/mainplane-server/worker.log
exec sudo -n env HTTPS_PROXY=http://127.0.0.1:9 HTTP_PROXY=http://127.0.0.1:9 NO_PROXY=127.0.0.1,localhost \
     TS_DEBUG_USE_DERP_HTTP=1 "$HOME/bin/mainplane" worker
