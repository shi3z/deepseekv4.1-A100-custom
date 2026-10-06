#!/usr/bin/env python3
"""GPU lender: when the DeepSeek LLM (GPUs 0-3, run_128k_4slots.sh) has been idle for a while and the H3 fleet has
work queued, stop the LLM, hand the lent GPUs (LEND_GPUS, at most 2) to the fleet as ComfyUI members c<gpu> on port
8190+<gpu>, and keep a wake-up proxy on the LLM port: GET /v1/models and /health keep answering, and the first
POST /v1/chat/completions (or /v1/completions) triggers the reclaim -- the lent members are stopped (their claims
requeued), the 4-GPU LLM is restarted (~2-3 min) and the held requests are then forwarded to it. Connections that
arrive while the LLM is loading (after the proxy released the port) are refused and must be retried. The fleet queue
draining (LEND_EMPTY_QUEUE_MIN) also reclaims. The lent GPUs' ComfyUI backend/worker are supervised here (the fleet's
own watchdog only knows c4/c6/c7), with the fleet's thermal rules. State: results/gpu_lender_state.json; log:
results/gpu_lender.log.

(The single-GPU expert-offload LLM, run_offload_lend.sh, was meant to keep serving during a lend but needs 20+ minutes
to pin 272 GB of experts into host RAM, so it is not used.)

Env: LEND_IDLE_MIN (20) minutes of LLM idleness before lending; LEND_MIN_QUEUE (1) fleet queue depth needed;
LEND_GPUS ("0,1"); LEND_POLL (30) s; LEND_EMPTY_QUEUE_MIN (10); LEND_PORT (8000); LEND_DRY_RUN=1 logs decisions
without acting.
"""
from __future__ import annotations

import http.client
import json
import os
import re
import socket
import subprocess
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path("/mnt/ssdraid/git/deepseekv4.1")
FLEET = Path("/mnt/ssdraid/project/h3-fleet")
COMFY = Path("/mnt/ssdraid/project/comfy-h3")
COMFY_PY = COMFY / ".venv-exp/bin/python"
TMUX_TARGET = "dsv41-server"
PORT = int(os.environ.get("LEND_PORT", "8000"))
API = f"http://127.0.0.1:{PORT}"
STATE = REPO / "results/gpu_lender_state.json"
LOG = REPO / "results/gpu_lender.log"
SERVE_RE = r"^/home/shi3z/.local/bin/python -u -m dsv41.serve"

IDLE_MIN = float(os.environ.get("LEND_IDLE_MIN", "20"))
MIN_QUEUE = int(os.environ.get("LEND_MIN_QUEUE", "1"))
LEND_GPUS = [int(g) for g in os.environ.get("LEND_GPUS", "0,1").split(",") if g.strip()][:2]
POLL = float(os.environ.get("LEND_POLL", "30"))
EMPTY_QUEUE_MIN = float(os.environ.get("LEND_EMPTY_QUEUE_MIN", "10"))
DRY = os.environ.get("LEND_DRY_RUN", "0") == "1"
# thermal limits, same as the fleet's thermal_guard.sh (it only covers c4/c6/c7): inlet >= WARN stops new jobs
# (SIGSTOP the lent workers), inlet >= CRIT also freezes the lent backends; release at <= CLR. Per-GPU: >= GPU_DANGER
# pauses that GPU's worker until <= GPU_RESUME. BMC inlet hard-crit is 50 C and the box has rebooted from it.
INLET_WARN = float(os.environ.get("THERMAL_INLET_WARN", "48"))
INLET_CRIT = float(os.environ.get("THERMAL_INLET_CRIT", "49"))
INLET_CLR = float(os.environ.get("THERMAL_INLET_WARN_CLR", "47"))
GPU_DANGER = float(os.environ.get("THERMAL_GPU_DANGER", "83"))
GPU_RESUME = float(os.environ.get("THERMAL_GPU_RESUME", "73"))
assert 1 <= len(LEND_GPUS) <= 2 and all(g in (0, 1, 2, 3) for g in LEND_GPUS), "lend 1-2 of the LLM's GPUs 0-3"


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def sh(cmd: str, check: bool = False) -> str:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"{cmd!r} failed: {r.stderr.strip()[:300]}")
    return r.stdout


# ------------------------------------------------------------------------------------------- observation
def api(path: str, timeout: float = 5.0):
    try:
        with urllib.request.urlopen(API + path, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return None


def llm_pid() -> int | None:
    out = sh(f"pgrep -f '{SERVE_RE}'").split()
    return int(out[0]) if out else None




def fleet_queue_depth() -> int:
    try:
        return sum(1 for p in (FLEET / "queue").iterdir() if p.suffix == ".json")
    except OSError:
        return 0


def gpu_mem_used_mib(gpu: int) -> int:
    out = sh(f"nvidia-smi --id={gpu} --query-gpu=memory.used --format=csv,noheader,nounits").strip()
    return int(out) if out.isdigit() else 10**9


# ------------------------------------------------------------------------------------------- LLM control
def stop_llm(timeout: float = 120.0) -> None:
    pid = llm_pid()
    if pid is None:
        return
    log(f"stopping LLM server pid {pid}")
    if DRY:
        return
    sh(f"kill {pid}")
    t0 = time.time()
    while time.time() - t0 < timeout and llm_pid() is not None:
        time.sleep(2)
    if llm_pid() is not None:
        sh(f"kill -9 {pid}")
        time.sleep(3)
    # wait for the GPUs to drain (the EP server frees ~75 GiB per GPU)
    t0 = time.time()
    while time.time() - t0 < 120 and any(gpu_mem_used_mib(g) > 4000 for g in range(4)):
        time.sleep(3)


def wait_gpus_free(gpus, timeout: float = 1800.0, limit_mib: int = 4000) -> bool:
    """Other services may grab a GPU while the LLM is down (seen: a TTS server took 11 GiB of GPU 1 and the restart
    OOMed). Wait until every LLM GPU is back under limit_mib, logging the occupants."""
    t0 = time.time()
    while True:
        busy = {g: gpu_mem_used_mib(g) for g in gpus if gpu_mem_used_mib(g) > limit_mib}
        if not busy:
            return True
        if time.time() - t0 > timeout:
            log(f"GPUs still occupied after {timeout / 60:.0f} min: {busy}; starting the LLM anyway")
            return False
        if int(time.time() - t0) % 300 < POLL:
            who = sh("nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader").strip().replace("\n", "; ")
            log(f"waiting for GPUs {sorted(busy)} to be freed by other processes ({busy} MiB used): {who[:300]}")
        time.sleep(POLL)


def start_llm(mode: str = "full") -> bool:
    script, logfile = "./run_128k_4slots.sh", "results/server_128k_4slots.log"
    wait_gpus_free(range(4))
    log(f"starting LLM ({script})")
    if DRY:
        return True
    with open(REPO / logfile, "a") as f:
        f.write(f"=== restart marker {time.strftime('%H:%M:%S')} gpu_lender -> {mode} ===\n")
    subprocess.run(["tmux", "send-keys", "-t", TMUX_TARGET, f"cd {REPO} && {script} 2>&1 | tee -a {logfile}", "Enter"], check=True)
    t0 = time.time()
    while time.time() - t0 < 900:
        if api("/v1/models") is not None:
            log(f"LLM {mode} mode up after {time.time() - t0:.0f} s")
            return True
        time.sleep(10)
    log(f"LLM {mode} mode did NOT come up within 15 min")
    return False


# ------------------------------------------------------------------------------------------- wake-up proxy on the LLM port
class WakeProxy:
    """Holds the LLM port while the LLM is stopped. Health/model-list GETs are answered locally; generation POSTs set
    `wake` and are held until `ready` (the real server is back on the same port), then forwarded and streamed."""

    def __init__(self, port: int, models_json: bytes):
        self.port, self.models_json = port, models_json
        self.wake, self.ready = threading.Event(), threading.Event()
        self.held = 0
        self.server: ThreadingHTTPServer | None = None
        proxy = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):  # quiet
                pass

            def _json(self, code: int, body: bytes):
                self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

            def do_GET(self):
                if self.path.startswith("/v1/models"):
                    self._json(200, proxy.models_json)
                elif self.path in ("/health",):
                    self._json(200, b'{"status": "ok", "note": "LLM lent to the fleet; first generation request wakes it (2-3 min)"}')
                else:
                    self._json(200, json.dumps({"engine_phase": "lent", "slots": [], "total_requests": 0, "lender": "GPUs lent to the fleet"}).encode())

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(n) if n else b""
                if not (self.path.startswith("/v1/chat/completions") or self.path.startswith("/v1/completions")):
                    self._json(503, b'{"error": "LLM is lent to the fleet; only /v1/chat/completions and /v1/completions wake it"}'); return
                proxy.held += 1
                if not proxy.wake.is_set():
                    log(f"wake request on {self.path} from {self.client_address[0]}: reclaiming the LLM")
                proxy.wake.set()
                if not proxy.ready.wait(timeout=1200):
                    self._json(503, b'{"error": "LLM did not come back within 20 min"}'); proxy.held -= 1; return
                try:  # forward to the real server, which now owns the port
                    c = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=3600)
                    hdrs = {k: v for k, v in self.headers.items() if k.lower() not in ("host", "content-length", "transfer-encoding")}
                    hdrs["Content-Length"] = str(len(body))
                    c.request("POST", self.path, body=body, headers=hdrs)
                    r = c.getresponse()
                    self.send_response(r.status)
                    chunked = False
                    for k, v in r.getheaders():
                        if k.lower() == "transfer-encoding" and "chunked" in v.lower():
                            chunked = True
                        if k.lower() in ("connection", "transfer-encoding"):
                            continue
                        self.send_header(k, v)
                    if chunked:
                        self.send_header("Transfer-Encoding", "chunked")
                    self.send_header("Connection", "close"); self.end_headers()
                    while True:
                        chunk = r.read(65536)
                        if not chunk:
                            break
                        if chunked:
                            self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
                        else:
                            self.wfile.write(chunk)
                        self.wfile.flush()
                    if chunked:
                        self.wfile.write(b"0\r\n\r\n")
                    c.close()
                except Exception as e:
                    log(f"proxy forward failed: {e!r}")
                finally:
                    proxy.held -= 1
                    self.close_connection = True

        self.handler = H

    def start(self) -> None:
        ThreadingHTTPServer.allow_reuse_address = True
        self.server = ThreadingHTTPServer(("0.0.0.0", self.port), self.handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        log(f"wake proxy listening on :{self.port}")

    def release_port(self) -> None:
        """Stop accepting (the real LLM is about to bind the port); held handler threads keep running."""
        if self.server is not None:
            self.server.shutdown(); self.server.server_close(); self.server = None
            log("wake proxy released the port")


def fetch_models_json() -> bytes:
    try:
        with urllib.request.urlopen(API + "/v1/models", timeout=5) as r:
            return r.read()
    except Exception:
        return json.dumps({"object": "list", "data": [{"id": "deepseek-v4.1-flash-abliterated", "object": "model", "owned_by": "local"}]}).encode()


# ------------------------------------------------------------------------------------------- fleet members on lent GPUs
def member(gpu: int) -> tuple[str, int]:
    return f"c{gpu}", 8190 + gpu


def worker_pids(port: int) -> list[int]:
    pids = []
    for pid in sh("pgrep -f 'comfy_worker\\.py$'").split():
        try:
            env = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
        except OSError:
            continue
        if f"COMFY_PORT={port}".encode() in env:
            pids.append(int(pid))
    return pids


def backend_pid(port: int) -> int | None:
    m = re.search(r"pid=(\d+)", sh(f"ss -tlnp | grep '127.0.0.1:{port} '"))
    return int(m.group(1)) if m else None


def backend_up(port: int) -> bool:
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/system_stats", timeout=5).read()
        return True
    except Exception:
        return False


def start_member(gpu: int) -> None:
    wid, port = member(gpu)
    if not backend_up(port):
        log(f"starting ComfyUI {wid} :{port} on GPU {gpu}")
        if not DRY:
            subprocess.Popen(
                f"cd {COMFY} && setsid nohup env CUDA_VISIBLE_DEVICES={gpu} PYTORCH_TRANSFORMERS_CACHE=/mnt/ssd/cache {COMFY_PY} main.py "
                f"--listen 127.0.0.1 --port {port} --extra-model-paths-config extra_model_paths.yaml --output-directory {COMFY}/output "
                f">> {COMFY}/comfy_gpu{gpu}.log 2>&1 < /dev/null &", shell=True)
            t0 = time.time()
            while time.time() - t0 < 600 and not backend_up(port):
                time.sleep(5)
            if not backend_up(port):
                log(f"ComfyUI {wid} :{port} did not come up")
                return
    if not worker_pids(port):
        log(f"starting worker {wid} (:{port})")
        if not DRY:
            subprocess.Popen(f"cd {FLEET} && setsid nohup env H3_WORKER_ID={wid} COMFY_PORT={port} python3 comfy_worker.py "
                             f">> worker_p{port}.log 2>&1 < /dev/null &", shell=True)


def requeue_claims(wid: str) -> None:
    for c in sorted(FLEET.glob(f"claimed/{wid}__*.json")):
        try:
            d = json.loads(c.read_text())
            j = d.get("job", d)
            tgt = "hold_norefs" if not j.get("refs") else "queue"
            (FLEET / tgt / f"cut{int(j['cut']):04d}.json").write_text(json.dumps(j, ensure_ascii=False))
            c.unlink()
            log(f"requeued {c.name} -> {tgt}")
        except Exception as e:
            log(f"could not requeue {c.name}: {e}")


def stop_member(gpu: int) -> None:
    wid, port = member(gpu)
    pids = worker_pids(port)
    bpid = backend_pid(port)
    if not pids and bpid is None:
        return
    log(f"stopping fleet member {wid} (:{port}, GPU {gpu}): worker {pids}, backend {bpid}")
    if DRY:
        return
    for pid in pids:
        sh(f"kill -CONT {pid}; kill -9 {pid}")
    requeue_claims(wid)
    if bpid is not None:
        sh(f"kill -CONT {bpid}; kill {bpid}")
        t0 = time.time()
        while time.time() - t0 < 30 and backend_pid(port) is not None:
            time.sleep(2)
        if (bpid := backend_pid(port)) is not None:
            sh(f"kill -9 {bpid}")
    t0 = time.time()
    while time.time() - t0 < 120 and gpu_mem_used_mib(gpu) > 4000:
        time.sleep(3)


# ------------------------------------------------------------------------------------------- thermal (lent GPUs only)
def inlet_temp() -> float | None:
    out = sh("timeout 15 ipmitool sdr type Temperature 2>/dev/null")
    m = re.search(r"Inlet Temp.*\|\s*(\d+) degrees C", out)
    if m:
        return float(m.group(1))
    try:  # fallback: thermal_guard's last TEMPS line
        tail = sh(f"tail -1 {FLEET}/thermal_guard.log")
        m = re.search(r"inlet=(\d+)C", tail)
        return float(m.group(1)) if m else None
    except Exception:
        return None


def gpu_temp(gpu: int) -> float | None:
    out = sh(f"nvidia-smi --id={gpu} --query-gpu=temperature.gpu --format=csv,noheader,nounits").strip()
    return float(out) if out.isdigit() else None


def signal_member(gpu: int, sig: str, backend_too: bool) -> None:
    wid, port = member(gpu)
    for pid in worker_pids(port):
        sh(f"kill -{sig} {pid}")
    if backend_too and (bpid := backend_pid(port)) is not None:
        sh(f"kill -{sig} {bpid}")


def thermal_step(st: dict) -> None:
    """Pause / freeze / resume the lent members like thermal_guard does for the fleet. st['frozen'] in
    {'', 'warn', 'crit'}; st['gpu_paused'] = list of lent GPUs whose worker is paused for its own temperature."""
    inlet = inlet_temp()
    level = st.get("frozen", "")
    gpus = st.get("gpus", LEND_GPUS)
    if inlet is not None:
        if inlet >= INLET_CRIT and level != "crit":
            log(f"THERMAL inlet={inlet:.0f}C >= {INLET_CRIT}: FREEZE lent members (workers + backends)")
            if not DRY:
                for g in gpus:
                    signal_member(g, "STOP", backend_too=True)
            st["frozen"] = "crit"
        elif INLET_WARN <= inlet < INLET_CRIT and level == "":
            log(f"THERMAL inlet={inlet:.0f}C >= {INLET_WARN}: pause lent workers (no new jobs)")
            if not DRY:
                for g in gpus:
                    signal_member(g, "STOP", backend_too=False)
            st["frozen"] = "warn"
        elif inlet <= INLET_CLR and level:
            log(f"THERMAL inlet={inlet:.0f}C <= {INLET_CLR}: resume lent members")
            if not DRY:
                for g in gpus:
                    signal_member(g, "CONT", backend_too=True)
            st["frozen"] = ""
    paused = set(st.get("gpu_paused", []))
    for g in gpus:
        t = gpu_temp(g)
        if t is None:
            continue
        if t >= GPU_DANGER and g not in paused:
            log(f"THERMAL GPU {g} {t:.0f}C >= {GPU_DANGER}: pause worker c{g}")
            if not DRY:
                signal_member(g, "STOP", backend_too=False)
            paused.add(g)
        elif t <= GPU_RESUME and g in paused and not st.get("frozen"):
            log(f"THERMAL GPU {g} {t:.0f}C <= {GPU_RESUME}: resume worker c{g}")
            if not DRY:
                signal_member(g, "CONT", backend_too=False)
            paused.discard(g)
    st["gpu_paused"] = sorted(paused)
    save_state(st)


# ------------------------------------------------------------------------------------------- state machine
def load_state() -> dict:
    try:
        return json.loads(STATE.read_text())
    except Exception:
        return {"mode": "full", "since": time.time()}


def save_state(st: dict) -> None:
    STATE.write_text(json.dumps(st, indent=1))


def main() -> None:
    st = load_state()
    log(f"gpu_lender start: mode={st['mode']} idle_min={IDLE_MIN} min_queue={MIN_QUEUE} lend_gpus={LEND_GPUS} port={PORT} dry_run={DRY}")
    models_json = fetch_models_json()
    proxy: WakeProxy | None = None
    if st["mode"] == "lend":   # restarted while lent: re-own the port if the LLM is down
        if llm_pid() is None and api("/v1/models") is None and not DRY:
            proxy = WakeProxy(PORT, models_json); proxy.start()
        else:
            log("state says lend but an LLM server is up; switching state to full")
            st = {"mode": "full", "since": time.time()}; save_state(st)
    last_activity = time.time()
    last_total = None
    queue_empty_since: float | None = None
    while True:
        try:
            qdepth = fleet_queue_depth()
            now = time.time()
            if qdepth >= MIN_QUEUE:
                queue_empty_since = None
            elif queue_empty_since is None:
                queue_empty_since = now

            if st["mode"] == "full":
                stats = api("/api/stats")
                if stats is not None and "engine_phase" in stats and "lender" not in stats:
                    total = stats.get("total_requests")
                    busy = stats.get("engine_phase") not in (None, "idle") or any(x.get("status") not in ("idle", "completed", None) for x in stats.get("slots", []))
                    if busy or (last_total is not None and total != last_total):
                        last_activity = now
                    last_total = total
                    models_json = fetch_models_json()
                idle_for = now - last_activity
                if llm_pid() is None or stats is None:
                    pass  # LLM not running / not answering: never lend on top of a manual stop or a loading server
                elif idle_for >= IDLE_MIN * 60 and qdepth >= MIN_QUEUE:
                    log(f"LLM idle for {idle_for / 60:.1f} min, fleet queue {qdepth}: LENDING GPUs {LEND_GPUS}")
                    stop_llm()
                    if not DRY:
                        proxy = WakeProxy(PORT, models_json); proxy.start()
                    for g in LEND_GPUS:
                        start_member(g)
                    st = {"mode": "lend", "since": now, "gpus": LEND_GPUS, "frozen": "", "gpu_paused": []}
                    save_state(st)
            else:  # lend mode
                request_seen = proxy is not None and proxy.wake.is_set()
                queue_drained = queue_empty_since is not None and (now - queue_empty_since) >= EMPTY_QUEUE_MIN * 60
                if request_seen or queue_drained or (DRY and False):
                    log(f"RECLAIMING GPUs {st.get('gpus', LEND_GPUS)} ({'LLM request waiting' if request_seen else 'fleet queue empty'})")
                    for g in st.get("gpus", LEND_GPUS):
                        stop_member(g)
                    if proxy is not None:
                        proxy.release_port()
                    ok = start_llm()
                    if proxy is not None:
                        if ok:
                            proxy.ready.set()      # held requests are forwarded now
                        proxy = None
                    st = {"mode": "full", "since": time.time()}
                    save_state(st)
                    last_total = None
                    last_activity = time.time()
                else:
                    thermal_step(st)
                    if not st.get("frozen"):              # supervise the lent members (a frozen backend does not answer HTTP)
                        for g in st.get("gpus", LEND_GPUS):
                            start_member(g)
        except Exception as e:
            log(f"ERROR in loop: {e!r}")
        time.sleep(POLL)


if __name__ == "__main__":
    main()
