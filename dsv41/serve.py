from pathlib import Path
"""OpenAI-compatible HTTP server (no web framework needed: stdlib http.server, threaded).

  python -m dsv41.serve --devices 0,1,2,3,4 --port 8000
  curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
       -d '{"model":"deepseek-v4.1-flash","messages":[{"role":"user","content":"hello"}],"stream":true}'

Endpoints: GET /v1/models, POST /v1/chat/completions (stream or not), POST /v1/completions, GET /health, GET /dashboard.
One request is generated at a time; others wait on the engine lock."""
import sys
import torch
import argparse
import json
import time
import os
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import copy

import os as _os
import traceback
import threading
_os.environ.setdefault("OMP_WAIT_POLICY", "active")  # CPU expert threads keep spinning between layers (libgomp reads this once)
_os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from .engine import Engine, GenParams, parse_budgets
from .stats import StatsTracker

ENGINE: Engine | None = None
STATS_TRACKER: StatsTracker | None = None


def _usage(n_prompt: int, n_out: int, info: dict | None = None) -> dict:
    """OpenAI usage block with the prefix-cache hit in the spellings clients read: prompt_tokens_details.cached_tokens
    (OpenAI), prompt_cache_hit_tokens / prompt_cache_miss_tokens (DeepSeek; mainplane reads both).  cached = prompt
    tokens whose K/V came from a GPU slot or a host/tmpfs prefix snapshot instead of a forward (engine prefill stats
    reused_tokens of THIS request), clamped to the prompt."""
    cached = max(0, min(int((info or {}).get("cached_tokens", 0) or 0), int(n_prompt)))
    return {"prompt_tokens": n_prompt, "completion_tokens": n_out, "total_tokens": n_prompt + n_out,
            "prompt_tokens_details": {"cached_tokens": cached},
            "prompt_cache_hit_tokens": cached, "prompt_cache_miss_tokens": n_prompt - cached}


def _client_opts(body: dict) -> tuple[str, bool]:
    """(prompt_cache_key, stream_options.include_usage) of a request.  prompt_cache_key (OpenAI; mainplane sends its
    session's first record id, a copied session shares it) is kept with the slot state and the prefix-cache entries the
    request makes and is logged; prefix matching itself stays by tokens.  include_usage asks for the usage chunk
    (choices: []) after the finish chunk of a stream."""
    key = body.get("prompt_cache_key") or body.get("cache_key") or ""
    key = str(key)[:128] if isinstance(key, (str, int)) else ""
    so = body.get("stream_options")
    include_usage = bool(isinstance(so, dict) and so.get("include_usage"))
    return key, include_usage


def _params(body: dict) -> GenParams:
    stop = body.get("stop") or []
    if isinstance(stop, str):
        stop = [stop]
    default_rep_pen = float(os.environ.get("DSV41_REPETITION_PENALTY", "1.05"))
    default_pres_pen = float(os.environ.get("DSV41_PRESENCE_PENALTY", "0.0"))
    default_freq_pen = float(os.environ.get("DSV41_FREQUENCY_PENALTY", "0.0"))
    default_window = int(os.environ.get("DSV41_PENALTY_WINDOW", "256"))
    default_prog_pen = float(os.environ.get("DSV41_PROGRESSIVE_PENALTY", "0.0"))
    default_ban_cycles = os.environ.get("DSV41_BAN_CYCLES", "1").strip().lower() not in ("0", "false", "off")
    default_loop_detect = os.environ.get("DSV41_LOOP_DETECT", "0").strip().lower() in ("1", "true", "on")
    default_min_loop_match = int(os.environ.get("DSV41_MIN_LOOP_MATCH", "48"))
    default_min_loop_cycle = int(os.environ.get("DSV41_MIN_LOOP_CYCLE", "1"))

    rep_pen = float(body.get("repetition_penalty") if body.get("repetition_penalty") is not None else default_rep_pen)
    pres_pen = float(body.get("presence_penalty") if body.get("presence_penalty") is not None else default_pres_pen)
    freq_pen = float(body.get("frequency_penalty") if body.get("frequency_penalty") is not None else default_freq_pen)
    window = int(body.get("penalty_window") or default_window)
    prog_pen = float(body.get("progressive_penalty") if body.get("progressive_penalty") is not None else default_prog_pen)
    ban_cycles = bool(body.get("ban_cycles") if body.get("ban_cycles") is not None else default_ban_cycles)
    loop_detect = bool(body.get("loop_detect") if body.get("loop_detect") is not None else default_loop_detect)
    min_loop_match = int(body.get("min_loop_match") or default_min_loop_match)
    min_loop_cycle = int(body.get("min_loop_cycle") or default_min_loop_cycle)

    return GenParams(
        max_new_tokens=int(body.get("max_tokens") or body.get("max_completion_tokens") or int(os.environ.get("DSV41_INTERACTIVE_MAX_NEW", "65536"))),
        temperature=float(body.get("temperature", 0.6)),
        top_p=float(body.get("top_p", 0.95)),
        stop=list(stop),
        seed=body.get("seed"),
        repetition_penalty=rep_pen,
        presence_penalty=pres_pen,
        frequency_penalty=freq_pen,
        penalty_window=window,
        progressive_penalty=prog_pen,
        ban_cycles=ban_cycles,
        loop_detect=loop_detect,
        min_loop_match=min_loop_match,
        min_loop_cycle=min_loop_cycle,
    )


class _KeepAlive:
    """Writes a heartbeat to the client socket every `interval` seconds while the handler waits
    for the engine. A queued request otherwise sends no bytes until its first token, and clients
    behind a read timeout drop the connection while the backend is busy. All writes to the
    socket go through write() so heartbeats never interleave with response data."""

    def __init__(self, wfile, payload: bytes, interval: float):
        self.wfile, self.payload, self.interval = wfile, payload, interval
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.dead = False
        self.beats = 0
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        if self.interval > 0:
            self.thread.start()
        return self

    def _run(self):
        while not self.stop.wait(self.interval):
            with self.lock:
                if self.stop.is_set():
                    return
                try:
                    self.wfile.write(self.payload)
                    self.wfile.flush()
                    self.beats += 1
                except Exception:
                    self.dead = True
                    return

    def write(self, data: bytes):
        with self.lock:
            self.wfile.write(data)
            self.wfile.flush()

    def close(self):
        self.stop.set()
        with self.lock:  # let an in-flight heartbeat finish before the caller writes again
            pass


def _heartbeat_interval() -> float:
    try:
        return float(os.environ.get("DSV41_HTTP_HEARTBEAT_S", "5"))
    except ValueError:
        return 5.0


class _StreamSplitter:
    """Turns the generated pieces of a chat stream into (kind, text) deltas, kind in {"reasoning", "content"},
    and detects the start of DSML tool-call markup.

    The tokenizer splits the markers across tokens ('<' + '｜DSML｜' + ' calls>', '</' + '｜DSML｜' ...), so a
    per-piece substring test never fires.  Detection therefore runs on an accumulated buffer, and a tail that could
    still be the beginning of a marker is held back until the next piece settles it (at most a few characters, one
    token of latency).  Once '<｜DSML｜' is seen nothing more is emitted: the caller parses the full text at the end.
    `in_thinking` starts True when the prompt ends with '<think>' (thinking mode): the model then emits only the
    closing '</think>', which used to leak into the content."""

    MARKERS = ("<｜DSML｜", "<think>", "</think>")

    def __init__(self, in_thinking: bool = False):
        self.in_thinking = in_thinking
        self.has_dsml = False
        self.buf = ""

    def feed(self, piece: str) -> list[tuple[str, str]]:
        if self.has_dsml:
            return []
        self.buf += piece
        return self._drain(final=False)

    def flush(self) -> list[tuple[str, str]]:
        return self._drain(final=True)

    def _emit(self, out: list, text: str) -> None:
        if text:
            out.append(("reasoning" if self.in_thinking else "content", text))

    def _held_len(self) -> int:
        longest = max(len(m) for m in self.MARKERS) - 1
        for L in range(min(len(self.buf), longest), 0, -1):
            suffix = self.buf[-L:]
            if any(m.startswith(suffix) for m in self.MARKERS):
                return L
        return 0

    def _drain(self, final: bool) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        while self.buf and not self.has_dsml:
            best = None
            for m in self.MARKERS:
                i = self.buf.find(m)
                if i >= 0 and (best is None or i < best[0]):
                    best = (i, m)
            if best is not None:
                i, m = best
                self._emit(out, self.buf[:i])
                self.buf = self.buf[i + len(m):]
                if m == "<｜DSML｜":
                    self.has_dsml = True
                    self.buf = ""
                elif m == "<think>":
                    self.in_thinking = True
                else:
                    self.in_thinking = False
                continue
            hold = 0 if final else self._held_len()
            if hold < len(self.buf):
                self._emit(out, self.buf[: len(self.buf) - hold])
                self.buf = self.buf[len(self.buf) - hold:]
            break
        return out


def _prompt_ends_in_thinking(eng, ids, thinking) -> bool:
    """True when generation starts inside a <think> block (the encoder appended the thinking start token)."""
    try:
        if ids:
            return eng.tok.decode(list(ids[-1:])) == "<think>"
    except Exception:
        pass
    return thinking == "thinking"


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quieter log
        if getattr(self, "path", "") in ("/api/metrics", "/api/stats", "/health", "/favicon.ico"):
            return
        print(f"[{time.strftime('%H:%M:%S')}] {self.address_string()} {fmt % args}", flush=True)

    # ---- keep-alive for blocking (non-stream) JSON responses ---------------------------
    # The 200 header goes out immediately with chunked transfer encoding; while the engine works,
    # a chunk holding one space is sent every few seconds (JSON parsers ignore leading whitespace),
    # then the JSON body and the chunked terminator. Errors after that point can only be reported
    # in the body (the status line is already sent).
    def _begin_json_keepalive(self):
        interval = _heartbeat_interval()
        if interval <= 0 or os.environ.get("DSV41_HTTP_HEARTBEAT_NONSTREAM", "1") == "0":
            return None
        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            return None
        return _KeepAlive(self.wfile, b"1\r\n \r\n", interval).start()

    def _end_json_keepalive(self, ka, obj: dict, t0: float | None = None):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        ka.close()
        try:
            ka.write(f"{len(data):x}\r\n".encode() + data + b"\r\n0\r\n\r\n")
        except (BrokenPipeError, ConnectionResetError, OSError):
            elapsed_str = f" elapsed={time.perf_counter()-t0:.3f}s" if t0 is not None else ""
            print(f"[http] client disconnected before response completed{elapsed_str} bytes={len(data)} stream=False heartbeats={ka.beats}", flush=True)
            if STATS_TRACKER:
                STATS_TRACKER.record_disconnect()
        except Exception as e:
            print(f"[http] response write error: {e}", flush=True)

    def _json(self, code: int, obj: dict, t0: float | None = None, is_stream: bool = False, ka=None):
        if ka is not None:
            return self._end_json_keepalive(ka, obj, t0=t0)
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        try:
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            elapsed_str = f" elapsed={time.perf_counter()-t0:.3f}s" if t0 is not None else ""
            print(f"[http] client disconnected before response completed{elapsed_str} bytes={len(data)} stream={is_stream}", flush=True)
            if STATS_TRACKER:
                STATS_TRACKER.record_disconnect()
        except Exception as e:
            print(f"[http] response write error: {e}", flush=True)

    def do_GET(self):
        if self.path == "/v1/models":
            cur_model = ENGINE.model_name if ENGINE else "deepseek-v4.1-flash-abliterated"
            data = [{"id": cur_model, "object": "model", "owned_by": "local"}]
            known_aliases = [
                "deepseek-v4.1-flash-abliterated",
                "deepseek-v4.1-flash",
                "claude-3-5-sonnet-20241022",
                "claude-3-7-sonnet-20250219",
                "claude-3-5-haiku-20241022",
                "claude-3-opus-20240229",
                "gpt-4o",
                "gpt-4",
                "default",
            ]
            for alias in known_aliases:
                if not any(d["id"] == alias for d in data):
                    data.append({"id": alias, "object": "model", "owned_by": "local"})
            self._json(200, {"object": "list", "data": data})
        elif self.path == "/health":
            self._json(200, {"status": "ok", "model": ENGINE.model_name, "max_seq_len": ENGINE.max_seq_len,
                             "max_seqs": ENGINE.max_seqs, "max_input_tokens": ENGINE.max_seq_len - 1,
                             "decode_slots": {str(s_): ENGINE.slot_cap_tokens(s_) for s_ in getattr(ENGINE, "decode_slot_ids", [])}})
        elif self.path in ("/dashboard", "/"):
            self._dashboard()
        elif self.path in (
            "/api/metrics", "/api/stats", "/stats", "/status", "/metrics",
            "/v1/stats", "/api/status", "/metrics.json", "/api/state",
            "/api/monitor", "/queue"
        ):
            self._json(200, STATS_TRACKER.get_metrics(ENGINE) if STATS_TRACKER else {})
        elif self.path in ("/api/slots", "/slots"):
            slots_data = {
                "max_seqs": getattr(ENGINE, "max_seqs", 1) if ENGINE else 0,
                "max_decode_slots": getattr(ENGINE, "max_decode_slots", 1) if ENGINE else 0,
                "free_slots": len(getattr(ENGINE, "_free_decode_slots", [])) if ENGINE else 0,
                "queue_depth": (ENGINE._batch_queue.qsize() + len(getattr(ENGINE, "_batch_waiting", []))) if ENGINE and hasattr(ENGINE, "_batch_queue") else 0,
                "slots": ENGINE.slot_states if ENGINE and hasattr(ENGINE, "slot_states") else {},
            }
            self._json(200, slots_data)
        else:
            self._json(404, {"error": "not found"})

    def _dashboard(self):
        model_name = ENGINE.model_name if ENGINE else "deepseek-v4.1-flash-abliterated"
        try:
            if STATS_TRACKER:
                html = STATS_TRACKER.render_dashboard_html(model_name).encode("utf-8")
            else:
                from .stats import _DASHBOARD_HTML_TEMPLATE
                html = _DASHBOARD_HTML_TEMPLATE.replace("__MODEL_NAME__", model_name).encode("utf-8")
        except Exception as e:
            html = f"<html><body><h1>Dashboard Error</h1><p>{e}</p></body></html>".encode("utf-8")
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self):
        if STATS_TRACKER:
            STATS_TRACKER.client_connected()
        try:
            n = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                return self._json(400, {"error": "invalid JSON"})
            if self.path in ("/v1/chat/completions", "/v1/completions"):
                rf_type = body.get("response_format", {}).get("type")
                has_rf_schema = rf_type == "json_schema" and "schema" in body.get("response_format", {}).get("json_schema", {})
                if has_rf_schema or body.get("schema"):
                    # structured-output requests are answered by ordinary generation, asking for JSON
                    body = self._schema_to_chat(body)
            if self.path == "/v1/chat/completions":
                return self._chat(body)
            if self.path == "/v1/completions":
                return self._completion(body)
            if self.path == "/v1/benchmark/prime":
                return self._benchmark_prime(body)
            if self.path == "/v1/benchmark/eval":
                return self._benchmark_eval(body)
            self._json(404, {"error": "not found"})
        finally:
            if STATS_TRACKER:
                STATS_TRACKER.client_disconnected()

    def _benchmark_prime(self, body: dict):
        eng = ENGINE
        if not eng:
            return self._json(500, {"error": "Engine not initialized"})
        prompt_ids = body.get("prompt_ids")
        if not prompt_ids:
            text = body.get("text")
            messages = body.get("messages")
            if text:
                prompt_ids = eng.tok.encode(text)
            elif messages:
                prompt_ids, _, _ = eng.format_chat(messages)
        if not prompt_ids:
            return self._json(400, {"error": "prompt_ids, text, or messages required"})
        try:
            res = eng.prime_benchmark_base(prompt_ids)
            return self._json(200, res)
        except Exception as e:
            tb = traceback.format_exc()
            return self._json(500, {"error": str(e), "traceback": tb})

    def _benchmark_eval(self, body: dict):
        eng = ENGINE
        if not eng:
            return self._json(500, {"error": "Engine not initialized"})
        prompt_ids = body.get("prompt_ids")
        policy = body.get("policy")
        block_size = body.get("block_size")
        if not prompt_ids:
            text = body.get("text")
            messages = body.get("messages")
            if text:
                prompt_ids = eng.tok.encode(text)
            elif messages:
                prompt_ids, _, _ = eng.format_chat(messages)
        if not prompt_ids or not policy:
            return self._json(400, {"error": "prompt_ids (or text/messages) and policy required"})
        try:
            res = eng.run_isolated_benchmark_step(prompt_ids, policy=policy, block_size=block_size)
            return self._json(200, res)
        except Exception as e:
            tb = traceback.format_exc()
            return self._json(500, {"error": str(e), "traceback": tb})

    # ---------------------------------------------------------------- chat
    def _chat(self, body: dict):
        eng = ENGINE

        # Per-request prefix cache control:
        #
        #   X-DSV41-Prefix-Cache: off
        #   X-DSV41-Prefix-Cache: on
        #
        # "off" bypasses prefix reuse only for this request.
        # "on" leaves normal cache behaviour enabled.
        _prefix_header = (
            self.headers.get(
                "X-DSV41-Prefix-Cache",
                "",
            )
            .strip()
            .lower()
        )

        if _prefix_header in ("off", "0", "false", "disable", "disabled"):
            eng._disable_prefix_cache_once = True
            print(
                "[http-debug] prefix-cache=OFF for this request",
                flush=True,
            )
        elif _prefix_header in ("on", "1", "true", "enable", "enabled"):
            eng._disable_prefix_cache_once = False
            print(
                "[http-debug] prefix-cache=ON for this request",
                flush=True,
            )

        _policy_header = (
            self.headers.get(
                "X-DSV41-Prefill-Policy",
                "",
            )
            .strip()
            .lower()
        )
        if _policy_header and _policy_header not in ("auto", "default", "none"):
            eng._forced_prefill_policy_once = _policy_header
            print(
                f"[http-debug] forced-prefill-policy={_policy_header} for this request",
                flush=True,
            )
        elif body.get("prefill_policy"):
            _p = str(body.get("prefill_policy")).strip().lower()
            if _p not in ("auto", "default", "none"):
                eng._forced_prefill_policy_once = _p
                print(
                    f"[http-debug] forced-prefill-policy={_p} (from body) for this request",
                    flush=True,
                )

        messages = copy.deepcopy(body.get("messages") or [])
        tools = body.get("tools")
        if tools and messages:
            if messages[0].get("role") == "system":
                messages[0]["tools"] = tools
            else:
                messages.insert(0, {"role": "system", "content": "", "tools": tools})
        thinking = "thinking" if body.get("reasoning_effort") or body.get("thinking") else None
        try:
            ids, images, token_types = eng.format_chat(messages, thinking)
        except Exception as e:  # malformed messages / unsupported content
            return self._json(400, {"error": f"cannot encode messages: {e}"})
        if len(ids) + 1 > eng.max_seq_len:
            return self._context_length_error(len(ids))
        params = _params(body)
        # Claude/LiteLLM may retry a stalled stream as a non-stream request
        # with max_tokens=32000.  Keep interactive retries bounded so a
        # response reaches the client and the request can complete.  Hosts
        # that need longer answers can raise this explicitly.
        _interactive_cap = int(os.environ.get("DSV41_INTERACTIVE_MAX_NEW", "65536"))
        requested_max_tokens = body.get("max_tokens") or body.get("max_completion_tokens")
        if _interactive_cap > 0 and params.max_new_tokens > _interactive_cap:
            params.max_new_tokens = _interactive_cap
            print(f"[chat] max_tokens capped={_interactive_cap} (requested={requested_max_tokens})", flush=True)
        rid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        t0 = time.perf_counter()
        cache_key, include_usage = _client_opts(body)
        info: dict = {}   # filled by the engine with this request's prefill stats / finish reason (see Engine._request_info)
        print(
          f"[chat] START req_id={rid} prompt_tokens={len(ids)} "
          f"cache_key={cache_key or '-'} "
          f"max_seq_len={eng.max_seq_len} "
          f"max_tokens={params.max_new_tokens} (effective) "
          f"stream={body.get('stream')} "
          f"rep_pen={params.repetition_penalty} freq_pen={params.frequency_penalty} "
          f"prog_pen={params.progressive_penalty} ban_cycles={params.ban_cycles}",
          flush=True,
        )
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            ka = _KeepAlive(self.wfile, b": keep-alive\n\n", _heartbeat_interval()).start()

            if STATS_TRACKER:
                STATS_TRACKER.record_request_start(len(ids), stream=True)
            t_gen_0 = time.perf_counter()

            def chunk(delta, finish=None):
                obj = {
                    "id": rid,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": body.get("model") or eng.model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": delta,
                            "finish_reason": finish,
                        }
                    ],
                }
                payload = "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"
                ka.write(payload.encode("utf-8"))

            # Markers ('<｜DSML｜', '<think>', '</think>') arrive split across tokens; the splitter detects them on
            # the accumulated text and holds back a possible marker prefix (see _StreamSplitter).
            splitter = _StreamSplitter(in_thinking=_prompt_ends_in_thinking(eng, ids, thinking))
            full_pieces = []
            n = 0

            def emit(events):
                for kind, text in events:
                    chunk({"reasoning_content": text} if kind == "reasoning" else {"content": text})

            try:
                chunk({"role": "assistant"})
                for tok_id, piece in eng.generate_stream(ids, params, images=images, token_types=token_types,
                                                         req_id=rid, cache_key=cache_key, info=info):
                    if not piece:
                        continue
                    full_pieces.append(piece)
                    n += 1
                    emit(splitter.feed(piece))
                emit(splitter.flush())
                has_dsml = splitter.has_dsml

                # If DSML tool calls were encountered, parse and emit structured tool_calls chunk
                full_text = "".join(full_pieces)
                tool_calls = []
                if has_dsml:
                    parsed = eng.parse_completion(full_text, thinking)
                    tool_calls = parsed.get("tool_calls", [])
                    if tool_calls:
                        for i, tc in enumerate(tool_calls):
                            tc = dict(tc)
                            tc.setdefault("id", f"call_{uuid.uuid4().hex[:24]}")
                            tc.setdefault("type", "function")
                            fn = tc.get("function") or {}
                            args = fn.get("arguments", "")
                            if not isinstance(args, str):
                                args = json.dumps(args, ensure_ascii=False)
                            chunk({
                                "tool_calls": [
                                    {
                                        "index": i,
                                        "id": tc["id"],
                                        "type": tc["type"],
                                        "function": {
                                            "name": fn.get("name", ""),
                                            "arguments": args,
                                        },
                                    }
                                ]
                            })

                if tool_calls:
                    finish = "tool_calls"
                elif (info.get("finish_reason") or getattr(eng, "last_finish_reason", None)) == "length" or n >= params.max_new_tokens:
                    finish = "length"
                else:
                    finish = "stop"

                chunk({}, finish)
                if include_usage:
                    # OpenAI stream_options.include_usage: one more chunk, no choices, the usage (mainplane, OpenCode,
                    # LiteLLM read cached tokens from here).
                    obj = {"id": rid, "object": "chat.completion.chunk", "created": created,
                           "model": body.get("model") or eng.model_name, "choices": [], "usage": _usage(len(ids), n, info)}
                    ka.write(("data: " + json.dumps(obj, ensure_ascii=False) + "\n\n").encode("utf-8"))
                ka.write(b"data: [DONE]\n\n")
                self.close_connection = True

            except (BrokenPipeError, ConnectionResetError):
                elapsed = time.perf_counter() - t0
                print(f"[http] client disconnected during chat stream elapsed={elapsed:.3f}s tokens={n} heartbeats={ka.beats}", flush=True)
                if STATS_TRACKER:
                    STATS_TRACKER.record_disconnect()
                return
            except torch.OutOfMemoryError as e:
                torch.cuda.empty_cache()
                try:
                    err_obj = {"error": {"message": f"CUDA out of memory: {e}", "type": "cuda_out_of_memory"}}
                    ka.write(f"data: {json.dumps(err_obj)}\n\ndata: [DONE]\n\n".encode())
                except Exception:
                    pass
                return
            except Exception as e:
                tb = traceback.format_exc()
                print(f"\n[chat-error] streaming generation failed req_id={rid}: {e}\n{tb}", flush=True)
                try:
                    Path("/tmp/dsv41-last-traceback.log").write_text(tb)
                except Exception:
                    pass
                try:
                    err_obj = {"error": {"message": str(e), "type": "generation_error", "traceback": tb}}
                    ka.write(f"data: {json.dumps(err_obj)}\n\ndata: [DONE]\n\n".encode())
                except Exception:
                    pass
                return
            finally:
                ka.close()

            dt_gen = time.perf_counter() - t_gen_0
            decode_tok_s = getattr(eng, "last_decode_tok_s", None) or (n / max(dt_gen, 1e-6))
            print(f"[chat] END req_id={rid} tokens={n} time={dt_gen:.2f}s ({decode_tok_s:.1f} tok/s decode)", flush=True)
            if STATS_TRACKER and n > 0:
                STATS_TRACKER.record_throughput(decode_tok_s, "chat-stream")
                STATS_TRACKER.record_cache_event(
                    f"Chat stream completed: {n} tokens in {dt_gen:.2f}s ({decode_tok_s:.1f} tok/s decode)"
                )
            return
        if STATS_TRACKER:
            STATS_TRACKER.record_request_start(len(ids), stream=False)
        t_gen_0 = time.perf_counter()
        ka = self._begin_json_keepalive()
        try:
            text, n = eng.generate_text(ids, params, images=images, token_types=token_types, req_id=rid, cache_key=cache_key, info=info)
        except Exception as e:
            tb = traceback.format_exc()
            print(f"\n[chat-error] non-streaming generation failed req_id={rid}: {e}\n{tb}", flush=True)
            try:
                Path("/tmp/dsv41-last-traceback.log").write_text(tb)
            except Exception:
                pass
            return self._json(500, {"error": {"message": str(e), "type": "server_error", "traceback": tb}}, t0=t0, is_stream=False, ka=ka)
        dt_gen = time.perf_counter() - t_gen_0
        decode_tok_s = getattr(eng, "last_decode_tok_s", None) or (n / max(dt_gen, 1e-6))
        print(f"[chat] END req_id={rid} tokens={n} time={dt_gen:.2f}s ({decode_tok_s:.1f} tok/s decode)", flush=True)
        if STATS_TRACKER and n > 0:
            STATS_TRACKER.record_throughput(decode_tok_s, "chat")
            STATS_TRACKER.record_cache_event(
                f"Chat completed: {n} tokens in {dt_gen:.2f}s ({decode_tok_s:.1f} tok/s decode)"
            )
        msg = eng.parse_completion(text, thinking)
        content = msg.get("content") if isinstance(msg, dict) else text
        out = {"id": rid, "object": "chat.completion", "created": created, "model": body.get("model") or eng.model_name,
               "choices": [{"index": 0, "message": {"role": "assistant", "content": content if content is not None else text},
                            "finish_reason": "tool_calls" if (isinstance(msg, dict) and msg.get("tool_calls")) else ("length" if ((info.get("finish_reason") or getattr(eng, "last_finish_reason", None)) == "length" or n >= params.max_new_tokens) else "stop")}],
               "usage": _usage(len(ids), n, info)}
        if isinstance(msg, dict) and msg.get("reasoning_content"):
            out["choices"][0]["message"]["reasoning_content"] = msg["reasoning_content"]
        if isinstance(msg, dict) and msg.get("tool_calls"):
            out["choices"][0]["message"]["tool_calls"] = msg["tool_calls"]
        if getattr(eng, "last_policy_decision", None) is not None:
            out["prefill_policy"] = eng.last_policy_decision.to_dict()
        self._json(200, out, t0=t0, is_stream=False, ka=ka)

    def _context_length_error(self, n_prompt: int, t0: float | None = None):
        """OpenAI-style context_length_exceeded (LiteLLM maps it to ContextWindowExceededError, so Claude Code
        compacts instead of retrying).  Without this the request ran a multi-minute prefix replay and then failed
        inside the model (engram cache sized to max_seq_len) with a 500 that clients retried in a loop."""
        limit = ENGINE.max_seq_len
        msg = (f"This model's maximum context length is {limit} tokens. However, your messages resulted in "
               f"{n_prompt} tokens (at least 1 token must be left for the completion). Please reduce the length of the messages.")
        print(f"[chat] REJECT prompt_tokens={n_prompt} > max_seq_len-1={limit - 1} (context_length_exceeded)", flush=True)
        return self._json(400, {"error": {"message": msg, "type": "invalid_request_error", "param": "messages",
                                          "code": "context_length_exceeded"}}, t0=t0, is_stream=False)

    # ------------------------------------------------ raw completions
    def _completion(self, body: dict):
        eng = ENGINE
        t0 = time.perf_counter()
        prompt = body.get("prompt") or ""
        if isinstance(prompt, list):
            prompt = prompt[0]
        if "<image>" in prompt and "</image>" in prompt:
            ids, images, token_types = eng.format_chat([{"role": "user", "content": prompt}], thinking_mode="chat")
        else:
            ids = eng.tok.encode(prompt)
            images, token_types = None, None
        if len(ids) + 1 > eng.max_seq_len:
            return self._context_length_error(len(ids), t0=t0)
        params = _params(body)
        info: dict = {}
        rid = f"cmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        if body.get("stream"):
            if STATS_TRACKER:
                STATS_TRACKER.record_request_start(len(ids), stream=True)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            ka = _KeepAlive(self.wfile, b": keep-alive\n\n", _heartbeat_interval()).start()
            n = 0
            t_gen_0 = time.perf_counter()
            try:
                for _, piece in eng.generate(ids, params, images=images, token_types=token_types):
                    obj = {"id": rid, "object": "text_completion", "created": created, "model": body.get("model") or eng.model_name,
                           "choices": [{"index": 0, "text": piece, "finish_reason": None}]}
                    ka.write(f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode())
                    n += 1
                ka.write(b"data: [DONE]\n\n")
                dt_gen = time.perf_counter() - t_gen_0
                decode_tok_s = getattr(eng, "last_decode_tok_s", None) or (n / max(dt_gen, 1e-6))
                if STATS_TRACKER and n > 0:
                    STATS_TRACKER.record_throughput(decode_tok_s, "completion-stream")
                    STATS_TRACKER.record_cache_event(
                        f"Completion stream: {n} tokens in {dt_gen:.2f}s ({decode_tok_s:.1f} tok/s decode)"
                    )
            except (BrokenPipeError, ConnectionResetError):
                elapsed = time.perf_counter() - t0
                print(f"[http] client disconnected during completion stream elapsed={elapsed:.3f}s tokens={n} heartbeats={ka.beats}", flush=True)
                if STATS_TRACKER:
                    STATS_TRACKER.record_disconnect()
            finally:
                ka.close()
            return
        if STATS_TRACKER:
            STATS_TRACKER.record_request_start(len(ids), stream=False)
        t_gen_0 = time.perf_counter()
        ka = self._begin_json_keepalive()
        try:
            text, n = eng.generate_text(ids, params, images=images, token_types=token_types, info=info)
        except Exception as e:
            tb = traceback.format_exc()
            print(f"\n[completion-error] generation failed req_id={rid}: {e}\n{tb}", flush=True)
            try:
                Path("/tmp/dsv41-last-traceback.log").write_text(tb)
            except Exception:
                pass
            return self._json(500, {"error": {"message": str(e), "type": "server_error", "traceback": tb}}, t0=t0, is_stream=False, ka=ka)
        dt_gen = time.perf_counter() - t_gen_0
        decode_tok_s = getattr(eng, "last_decode_tok_s", None) or (n / max(dt_gen, 1e-6))
        if STATS_TRACKER and n > 0:
            STATS_TRACKER.record_throughput(decode_tok_s, "completion")
            STATS_TRACKER.record_cache_event(
                f"Completion: {n} tokens in {dt_gen:.2f}s ({decode_tok_s:.1f} tok/s decode)"
            )
        self._json(200, {"id": rid, "object": "text_completion", "created": created, "model": body.get("model") or eng.model_name,
                          "choices": [{"index": 0, "text": text, "finish_reason": "length" if (getattr(eng, "last_finish_reason", None) == "length" or n >= params.max_new_tokens) else "stop"}],
                          "usage": _usage(len(ids), n, info)}, t0=t0, is_stream=False, ka=ka)

    # ---------------------------------------------------------------- schema requests
    def _schema_to_chat(self, body: dict) -> dict:
        schema = body.get("schema")
        if not schema and isinstance(body.get("response_format"), dict):
            schema = (body["response_format"].get("json_schema") or {}).get("schema")
        hint = "Respond with a single JSON object only, no prose and no code fence."
        if schema:
            hint += " It must match this JSON schema: " + json.dumps(schema, ensure_ascii=False)
        body = dict(body)
        msgs = list(body.get("messages") or [])
        if not msgs and body.get("prompt"):
            msgs = [{"role": "user", "content": str(body["prompt"])}]
        if msgs and msgs[0].get("role") == "system" and isinstance(msgs[0].get("content"), str):
            msgs[0] = {"role": "system", "content": msgs[0]["content"] + "\n\n" + hint}
        else:
            msgs.insert(0, {"role": "system", "content": hint})
        body["messages"] = msgs
        body.pop("schema", None)
        body.pop("response_format", None)
        print(f"[schema] request answered by plain generation with a JSON hint ({len(msgs)} messages)", flush=True)
        return body


def main():
    global ENGINE, STATS_TRACKER
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--devices", default="2,3,0,1")
    ap.add_argument("--budgets", default="")
    ap.add_argument("--max-seq-len", type=int, default=8192)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-graphs", action="store_true")
    ap.add_argument("--offload-experts", nargs="?", const="cpu", default=False, choices=["gpu", "cpu"], help="single-GPU mode: experts in host RAM; 'cpu' computes them on the CPU (default), 'gpu' streams them over PCIe")
    ap.add_argument("--hot-experts", type=int, default=0, help="cpu offload mode: experts per layer kept on the GPU (by usage stats)")
    ap.add_argument("--hot-stats", default="", help="route stats .pt used to pick the hot experts (default: results/route_stats.pt)")
    ap.add_argument("--ep", action="store_true", help="expert parallelism: experts sharded over the devices (e.g. --devices 2,3,0,1 --ep-shards 100,100,100,84)")
    ap.add_argument("--ep-shards", default="", help="experts per device for --ep (default: even split)")
    ap.add_argument(
        "--mtp",
        type=int,
        default=0,
        help="speculative decoding: DSpark drafts verified per step (3-5; 0 = off)",
    )
    ap.add_argument(
        "--mtp-device",
        type=int,
        default=None,
        help="CUDA device used exclusively for DSpark/MTP weights",
    )
    ap.add_argument(
        "--max-seqs",
        type=int,
        default=1,
        help="concurrent sequence slots (1 = serialized single request; >1 = batched decode)",
    )
    a = ap.parse_args()
    dev_list = [int(d) for d in a.devices.split(",") if int(d) in (0, 1, 2, 3)]
    kw = dict(devices=dev_list, max_seq_len=a.max_seq_len, budgets=parse_budgets(a.budgets),
              use_graphs=not a.no_graphs, offload_experts=a.offload_experts, hot_experts=a.hot_experts, route_stats=a.hot_stats,
              ep=a.ep, ep_shards=[int(v) for v in a.ep_shards.split(",")] if a.ep_shards else None, mtp=a.mtp, mtp_device=a.mtp_device,
              max_seqs=a.max_seqs)
    ENGINE = Engine(a.ckpt, **kw) if a.ckpt else Engine(**kw)
    ENGINE.devices = dev_list
    STATS_TRACKER = StatsTracker(active_devices=dev_list)
    ENGINE.stats_tracker = STATS_TRACKER
    ThreadingHTTPServer.request_queue_size = 128
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"serving OpenAI-compatible API on http://{a.host}:{a.port}/v1 (model '{ENGINE.model_name}')", flush=True)
    print(f"monitoring dashboard active at http://{a.host}:{a.port}/dashboard", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"[server-fatal] serve_forever error: {e}", flush=True)
        traceback.print_exc()



if __name__ == "__main__":
    main()
