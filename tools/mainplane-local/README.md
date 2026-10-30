# mainplane on the dsv41 machine, loopback only (2026-10-30)

[mainplane](https://github.com/mainplane-ai/mainplane) is an agent harness (Go): sessions, tools (`run`/`read`/`write`/`edit`
on "workers"), a mesh between harness and workers, LLM providers behind it. Its `openai-chat` provider talks to the dsv41
server (`/v1/chat/completions`, see the README's "OpenAI-compatible agent clients" section).

The product's install (`mainplane-server install`) opens a Cloudflare quick tunnel and registers a network name on
`pointer.mainplane.ai`. This directory is the recipe for running harness + admin worker + CLI on **one machine, over
loopback, with neither**: no tunnel, no public registration, nothing listening outside 127.0.0.1. The dsv41 server stays
as it is.

## What runs

| process | runs as | how | why |
|---|---|---|---|
| `mainplane-server up /var/lib/mainplane-server/config.json` | the user | `mainplane-local-up.sh` (nohup) | the harness: API on 127.0.0.1:8080, its own mesh node in userspace, `/derp` relay on the same port |
| `mainplane worker` | root, code as the operator (`shi3z`) | `mainplane-local-worker.sh` (sudo -n) | the admin worker: the mesh needs a TUN (`mainplane0`) and `ip link`, so root as in the official install; tool calls run as the operator via `Credential{uid,gid}` |
| `mainplane ...` | the user | `~/bin/mainplane` | CLI; finds the harness through `pointer.Local`, i.e. `/var/lib/mainplane-server/config.json` on this machine, and proves its key at `/id` |

Environment on the harness and worker processes:

- `HTTPS_PROXY=HTTP_PROXY=http://127.0.0.1:9`, `NO_PROXY=127.0.0.1,localhost`: every call to `pointer.mainplane.ai`
  (name claim, URL publish) dies on a dead proxy in milliseconds and is logged `again in 1m0s`; loopback (harness API,
  dsv41 on 8000, relay) bypasses the proxy. STUN probes (UDP to stun.cloudflare.com / stun.l.google.com) still go out:
  they are plain UDP, not proxied; harmless, and the nodes pick a direct path anyway.
- `TS_DEBUG_USE_DERP_HTTP=1`: tailscale's DERP client dials the relay as `http://localhost:8080/derp` instead of
  `https://localhost/derp` (nothing serves 443 here).

Config (`url` makes the harness start no cloudflared; `way()` insists on exactly `https://<host>`):

```json
{
  "http": "127.0.0.1:8080",
  "url": "https://localhost",
  "providers": {"dsv41": {"shape": "openai-chat", "url": "http://127.0.0.1:8000/v1", "key": "local"}}
}
```

## Build (from source, Go 1.27)

```bash
git clone --depth 1 https://github.com/mainplane-ai/mainplane   # tested at 690ac7f (v0.5.9 era)
cd mainplane
git apply /mnt/ssdraid/git/deepseekv4.1/tools/mainplane-local/coordinator-loopback-relay.patch
mkdir -p cmd/dsv41local cmd/dsv41probe
cp /mnt/ssdraid/git/deepseekv4.1/tools/mainplane-local/dsv41local.go cmd/dsv41local/main.go
cp /mnt/ssdraid/git/deepseekv4.1/tools/mainplane-local/dsv41probe.go cmd/dsv41probe/main.go
go build -o ~/bin/ ./cmd/mainplane ./cmd/mainplane-server ./cmd/dsv41local ./cmd/dsv41probe
```

`coordinator-loopback-relay.patch`: when the harness URL's host is `localhost` (or a loopback IP), the relay node in the
mesh map is port 8080 (the harness's own HTTP port, where `/derp` is served) instead of 443. Without it both nodes retry
`https://localhost/derp` once a second. One `if` in `relayMap()`.

`dsv41local <dir> <network name>`: makes the harness key, `network.json` (name + device code, so the harness claims
nothing at start), an api key named `default`, and prints the CLI api key and the admin worker's join token
(`auth.Secret(k, "admin " + code)`, what `mainplane-server install` computes).

`dsv41probe <base url> <model> <cache key>`: drives a dsv41 server with mainplane's own `pkg/provider` code (compile →
stream → records) for two turns, the second with a tool result; used with `fake_dsv41_server.py` (dsv41.serve on a fake
engine) to verify the wire format without a GPU.

## Set up and start

```bash
sudo mkdir -p /var/lib/mainplane-server && sudo chown $USER /var/lib/mainplane-server && chmod 700 /var/lib/mainplane-server
# config.json as above
~/bin/dsv41local /var/lib/mainplane-server dsv41-local-001 > /var/lib/mainplane-server/local-tokens.txt   # keep private
# admin worker state, as the official install leaves it
sudo mkdir -p /var/lib/mainplane && sudo chmod 700 /var/lib/mainplane
awk '/^admin_token/{printf "%s",$2}' /var/lib/mainplane-server/local-tokens.txt | sudo tee /var/lib/mainplane/join >/dev/null
printf '%s' "$USER" | sudo tee /var/lib/mainplane/operator >/dev/null; sudo chmod 600 /var/lib/mainplane/join
cp mainplane-local-up.sh mainplane-local-worker.sh ~/bin/
nohup ~/bin/mainplane-local-up.sh     >> /var/lib/mainplane-server/harness.log 2>&1 &
nohup ~/bin/mainplane-local-worker.sh >> /var/lib/mainplane-server/worker.log  2>&1 &
~/bin/mainplane login "$(awk '/^api_key/{print $2}' /var/lib/mainplane-server/local-tokens.txt)"
~/bin/mainplane workers          # admin  linux/amd64  bash,js  connected
echo '{"model":"dsv41/deepseek-v4.1-flash","tools":"default","context_limit":480000,"workers":[{"name":"admin"}],"params":{"max_tokens":4096,"temperature":0.3}}' | ~/bin/mainplane new
~/bin/mainplane message <id> "..."; ~/bin/mainplane tail <id>; ~/bin/mainplane chat <id>
```

Neither process is a service: after a reboot start the two scripts again (or add them to the user's crontab `@reboot`).
`mainplane-server install` would replace all of this with systemd units, a tunnel and a public name.

## Notes

- The dsv41 server has no auth; it is reached on 127.0.0.1 only here. The harness API has api keys but listens on
  loopback only, so nothing of mainplane is reachable from outside this machine.
- The CLI (`mainplane` with no verb, the TUI) fetches the model catalog from `dl.mainplane.ai` at start; the local
  provider `dsv41` is not in that catalog, so sessions are made with `mainplane new` and a JSON body, or the TUI's
  `/config` JSON.
- A session step waits in the dsv41 queue like any other request; while another client runs a long cold prefill
  (190K tokens, ~15 min) the first step waits for a decode slot.
