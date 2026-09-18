from pathlib import Path
"""OpenAI-compatible HTTP server (no web framework needed: stdlib http.server, threaded).

  python -m dsv41.serve --devices 0,1,2,3,4 --port 8000
  curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
       -d '{"model":"deepseek-v4.1-flash","messages":[{"role":"user","content":"hello"}],"stream":true}'

Endpoints: GET /v1/models, POST /v1/chat/completions (stream or not), POST /v1/completions, GET /health, GET /dashboard.
One request is generated at a time; others wait on the engine lock."""
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

    rep_pen = float(body.get("repetition_penalty") if body.get("repetition_penalty") is not None else default_rep_pen)
    pres_pen = float(body.get("presence_penalty") if body.get("presence_penalty") is not None else default_pres_pen)
    freq_pen = float(body.get("frequency_penalty") if body.get("frequency_penalty") is not None else default_freq_pen)
    window = int(body.get("penalty_window") or default_window)
    prog_pen = float(body.get("progressive_penalty") if body.get("progressive_penalty") is not None else default_prog_pen)
    ban_cycles = bool(body.get("ban_cycles") if body.get("ban_cycles") is not None else default_ban_cycles)

    return GenParams(
        max_new_tokens=int(body.get("max_tokens") or body.get("max_completion_tokens") or 4096),
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
    )


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quieter log
        if getattr(self, "path", "") in ("/api/metrics", "/api/stats", "/health", "/favicon.ico"):
            return
        print(f"[{time.strftime('%H:%M:%S')}] {self.address_string()} {fmt % args}", flush=True)

    def _json(self, code: int, obj: dict, t0: float | None = None, is_stream: bool = False):
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
            cur_model = ENGINE.model_name if ENGINE else "deepseek-v4.1-flash"
            data = [{"id": cur_model, "object": "model", "owned_by": "local"}]
            for alias in ("deepseek-v4.1-flash", "deepseek-v4.1-flash-abliterated"):
                if alias != cur_model:
                    data.append({"id": alias, "object": "model", "owned_by": "local"})
            self._json(200, {"object": "list", "data": data})
        elif self.path == "/health":
            self._json(200, {"status": "ok"})
        elif self.path in ("/dashboard", "/"):
            self._dashboard()
        elif self.path in ("/api/metrics", "/api/stats"):
            self._json(200, STATS_TRACKER.get_metrics(ENGINE) if STATS_TRACKER else {})
        else:
            self._json(404, {"error": "not found"})

    def _dashboard(self):
        if not STATS_TRACKER:
            return self._json(500, {"error": "stats tracker not initialized"})
        try:
            import dsv41.stats as _st_mod
            import importlib
            importlib.reload(_st_mod)
            model_name = ENGINE.model_name if ENGINE else "deepseek-v4.1-flash"
            html = _st_mod._DASHBOARD_HTML_TEMPLATE.replace("__MODEL_NAME__", model_name).encode("utf-8")
        except Exception:
            html = STATS_TRACKER.render_dashboard_html(ENGINE.model_name if ENGINE else "deepseek-v4.1-flash").encode("utf-8")
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
                if body.get("jev") or body.get("mode") == "jev" or has_rf_schema or body.get("schema"):
                    return self._jev(body)
            if self.path == "/v1/chat/completions":
                return self._chat(body)
            if self.path == "/v1/completions":
                return self._completion(body)
            self._json(404, {"error": "not found"})
        finally:
            if STATS_TRACKER:
                STATS_TRACKER.client_disconnected()


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
        messages = copy.deepcopy(body.get("messages") or [])
        tools = body.get("tools")
        if tools and messages:
            if messages[0].get("role") == "system":
                messages[0]["tools"] = tools
            else:
                messages.insert(0, {"role": "system", "content": "", "tools": tools})
        thinking = "thinking" if body.get("reasoning_effort") or body.get("thinking") else None
        try:
            ids = eng.tok.encode(eng.chat_prompt(messages, thinking))
        except Exception as e:  # malformed messages / unsupported content
            return self._json(400, {"error": f"cannot encode messages: {e}"})
        params = _params(body)
        # Claude/LiteLLM may retry a stalled stream as a non-stream request
        # with max_tokens=32000.  Keep interactive retries bounded so a
        # response reaches the client and the request can complete.  Hosts
        # that need longer answers can raise this explicitly.
        _interactive_cap = int(os.environ.get("DSV41_INTERACTIVE_MAX_NEW", "16384"))
        requested_max_tokens = body.get("max_tokens") or body.get("max_completion_tokens")
        if _interactive_cap > 0 and params.max_new_tokens > _interactive_cap:
            params.max_new_tokens = _interactive_cap
            print(f"[chat] max_tokens capped={_interactive_cap} (requested={requested_max_tokens})", flush=True)
        rid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        t0 = time.perf_counter()
        print(
          f"[chat] START req_id={rid} prompt_tokens={len(ids)} "
          f"max_seq_len={eng.max_seq_len} "
          f"max_tokens={params.max_new_tokens} (effective) "
          f"stream={body.get('stream')} "
          f"rep_pen={params.repetition_penalty} freq_pen={params.frequency_penalty} "
          f"prog_pen={params.progressive_penalty} ban_cycles={params.ban_cycles}",
          flush=True,
        )
        if body.get("stream"):
            # Send SSE headers immediately and emit keep-alive comments while
            # long-context prefill/generation is running. Previously the
            # server buffered the entire response, so LiteLLM timed out on
            # 100K+ prompts despite stream=True.
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            heartbeat_stop = threading.Event()
            sse_lock = threading.Lock()
            heartbeat_thread = None
            def _sse_write(data):
                with sse_lock:
                    self.wfile.write(data)
                    self.wfile.flush()
            def _heartbeat():
                while not heartbeat_stop.wait(10.0):
                    try:
                        _sse_write(b": keep-alive\n\n")
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        break
            heartbeat_thread = threading.Thread(target=_heartbeat, daemon=True)
            heartbeat_thread.start()
            if STATS_TRACKER:
                STATS_TRACKER.record_request_start(len(ids), stream=True)
            t_gen_0 = time.perf_counter()
            try:
                text, n = eng.generate_text(ids, params)

            except torch.OutOfMemoryError as e:
                heartbeat_stop.set()
                heartbeat_thread.join(timeout=1.0)
                torch.cuda.empty_cache()
                try:
                    _sse_write(("data: " + json.dumps({"error": {
                        "message": f"CUDA out of memory: {e}",
                        "type": "cuda_out_of_memory",
                    }}, ensure_ascii=False) + "\n\n").encode())
                    _sse_write(b"data: [DONE]\n\n")
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
                return

            except Exception as e:
                heartbeat_stop.set()
                heartbeat_thread.join(timeout=1.0)
                tb = traceback.format_exc()

                print(
                    "\n========== GENERATION TRACEBACK ==========",
                    flush=True,
                )
                print(tb, flush=True)
                print(
                    "==========================================",
                    flush=True,
                )

                try:
                    Path("/tmp/dsv41-last-traceback.log").write_text(tb)
                except Exception:
                    pass

                try:
                    _sse_write(("data: " + json.dumps({"error": {
                        "message": str(e),
                        "type": "generation_error",
                        "traceback": tb,
                    }}, ensure_ascii=False) + "\n\n").encode())
                    _sse_write(b"data: [DONE]\n\n")
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
                return

            heartbeat_stop.set()
            heartbeat_thread.join(timeout=1.0)

            # DeepSeek completion may contain thinking / DSML tool calls.
            # Always parse it before exposing an OpenAI-compatible response.
            msg = eng.parse_completion(text, thinking)

            if isinstance(msg, dict):
                content = msg.get("content")
                reasoning = msg.get("reasoning_content")
                tool_calls = msg.get("tool_calls") or []
            else:
                content = text
                reasoning = None
                tool_calls = []

            # Never silently discard generated text.
            if not content and not tool_calls:
                content = text

            print(
                f"[chat] GENERATED n={n} "
                f"raw_chars={len(text)} "
                f"content_chars={len(content or '')} "
                f"tool_calls={len(tool_calls)} "
                f"raw_preview={text[:300]!r}",
                flush=True,
            )

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

                payload = (
                    "data: "
                    + json.dumps(obj, ensure_ascii=False)
                    + "\n\n"
                )

                _sse_write(payload.encode("utf-8"))

            try:
                # Do NOT emit content:"" here.
                chunk({"role": "assistant"})

                if reasoning:
                    # LiteLLM understands reasoning_content for OpenAI-style
                    # backends. Keep it separate from visible answer text.
                    chunk({"reasoning_content": reasoning})

                if content:
                    # Send text in moderate chunks rather than one huge
                    # delta. This is friendlier to LiteLLM's Anthropic
                    # streaming bridge.
                    for i in range(0, len(content), 64):
                        chunk(
                            {
                                "content": content[i:i + 64]
                            }
                        )

                if tool_calls:
                    for i, tc in enumerate(tool_calls):
                        tc = dict(tc)

                        tc.setdefault(
                            "id",
                            f"call_{uuid.uuid4().hex[:24]}",
                        )
                        tc.setdefault("type", "function")

                        fn = tc.get("function") or {}
                        args = fn.get("arguments", "")

                        if not isinstance(args, str):
                            args = json.dumps(
                                args,
                                ensure_ascii=False,
                            )

                        chunk(
                            {
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
                            }
                        )

                if tool_calls:
                    finish = "tool_calls"
                elif getattr(eng, "last_finish_reason", None) == "length" or n >= params.max_new_tokens:
                    finish = "length"
                else:
                    finish = "stop"

                chunk({}, finish)

                _sse_write(b"data: [DONE]\n\n")

            except (BrokenPipeError, ConnectionResetError):
                elapsed = time.perf_counter() - t0
                print(f"[http] client disconnected during chat stream elapsed={elapsed:.3f}s tokens={n}", flush=True)
                if STATS_TRACKER:
                    STATS_TRACKER.record_disconnect()
                return

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
        try:
            text, n = eng.generate_text(ids, params)
        except Exception as e:
            tb = traceback.format_exc()
            print(f"\n[chat-error] non-streaming generation failed req_id={rid}: {e}\n{tb}", flush=True)
            try:
                Path("/tmp/dsv41-last-traceback.log").write_text(tb)
            except Exception:
                pass
            return self._json(500, {"error": {"message": str(e), "type": "server_error", "traceback": tb}}, t0=t0, is_stream=False)
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
                            "finish_reason": "tool_calls" if (isinstance(msg, dict) and msg.get("tool_calls")) else ("length" if (getattr(eng, "last_finish_reason", None) == "length" or n >= params.max_new_tokens) else "stop")}],
               "usage": {"prompt_tokens": len(ids), "completion_tokens": n, "total_tokens": len(ids) + n}}
        if isinstance(msg, dict) and msg.get("reasoning_content"):
            out["choices"][0]["message"]["reasoning_content"] = msg["reasoning_content"]
        if isinstance(msg, dict) and msg.get("tool_calls"):
            out["choices"][0]["message"]["tool_calls"] = msg["tool_calls"]
        self._json(200, out, t0=t0, is_stream=False)

    # ------------------------------------------------ raw completions
    def _completion(self, body: dict):
        eng = ENGINE
        t0 = time.perf_counter()
        prompt = body.get("prompt") or ""
        if isinstance(prompt, list):
            prompt = prompt[0]
        ids = eng.tok.encode(prompt)
        params = _params(body)
        rid = f"cmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        if body.get("stream"):
            if STATS_TRACKER:
                STATS_TRACKER.record_request_start(len(ids), stream=True)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            n = 0
            t_gen_0 = time.perf_counter()
            try:
                for _, piece in eng.generate(ids, params):
                    obj = {"id": rid, "object": "text_completion", "created": created, "model": body.get("model") or eng.model_name,
                           "choices": [{"index": 0, "text": piece, "finish_reason": None}]}
                    self.wfile.write(f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode())
                    self.wfile.flush()
                    n += 1
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                dt_gen = time.perf_counter() - t_gen_0
                decode_tok_s = getattr(eng, "last_decode_tok_s", None) or (n / max(dt_gen, 1e-6))
                if STATS_TRACKER and n > 0:
                    STATS_TRACKER.record_throughput(decode_tok_s, "completion-stream")
                    STATS_TRACKER.record_cache_event(
                        f"Completion stream: {n} tokens in {dt_gen:.2f}s ({decode_tok_s:.1f} tok/s decode)"
                    )
            except (BrokenPipeError, ConnectionResetError):
                elapsed = time.perf_counter() - t0
                print(f"[http] client disconnected during completion stream elapsed={elapsed:.3f}s tokens={n}", flush=True)
                if STATS_TRACKER:
                    STATS_TRACKER.record_disconnect()
            return
        if STATS_TRACKER:
            STATS_TRACKER.record_request_start(len(ids), stream=False)
        t_gen_0 = time.perf_counter()
        try:
            text, n = eng.generate_text(ids, params)
        except Exception as e:
            tb = traceback.format_exc()
            print(f"\n[completion-error] generation failed req_id={rid}: {e}\n{tb}", flush=True)
            try:
                Path("/tmp/dsv41-last-traceback.log").write_text(tb)
            except Exception:
                pass
            return self._json(500, {"error": {"message": str(e), "type": "server_error", "traceback": tb}}, t0=t0, is_stream=False)
        dt_gen = time.perf_counter() - t_gen_0
        decode_tok_s = getattr(eng, "last_decode_tok_s", None) or (n / max(dt_gen, 1e-6))
        if STATS_TRACKER and n > 0:
            STATS_TRACKER.record_throughput(decode_tok_s, "completion")
            STATS_TRACKER.record_cache_event(
                f"Completion: {n} tokens in {dt_gen:.2f}s ({decode_tok_s:.1f} tok/s decode)"
            )
        self._json(200, {"id": rid, "object": "text_completion", "created": created, "model": body.get("model") or eng.model_name,
                          "choices": [{"index": 0, "text": text, "finish_reason": "length" if (getattr(eng, "last_finish_reason", None) == "length" or n >= params.max_new_tokens) else "stop"}],
                          "usage": {"prompt_tokens": len(ids), "completion_tokens": n, "total_tokens": len(ids) + n}}, t0=t0, is_stream=False)

    # ---------------------------------------------------------------- Jev mode structured output
    def _jev(self, body: dict):
        eng = ENGINE
        t0 = time.perf_counter()
        stream = bool(body.get("stream"))

        raw_schema = body.get("schema")
        if not raw_schema and isinstance(body.get("response_format"), dict):
            rf = body["response_format"]
            if rf.get("type") == "json_schema" and isinstance(rf.get("json_schema"), dict):
                raw_schema = rf["json_schema"].get("schema")
        if not raw_schema:
            return self._json(400, {"error": "schema is required for Jev mode"}, t0=t0, is_stream=stream)

        prompt = body.get("prompt")
        if not prompt and body.get("messages"):
            for m in reversed(body["messages"]):
                if m.get("role") == "user":
                    prompt = m.get("content")
                    break
            if not prompt:
                prompt = body["messages"][-1].get("content", "")

        if not prompt:
            return self._json(400, {"error": "prompt or user message is required"}, t0=t0, is_stream=stream)

        max_batch = int(body.get("max_batch") or os.environ.get("DSV41_JEV_MAX_BATCH", "32"))
        rid = f"jevcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        print(
            f"[jev] request start prompt_len={len(prompt)} schema_fields={len(raw_schema)} stream={stream}",
            flush=True,
        )
        if STATS_TRACKER:
            STATS_TRACKER.record_request_start(len(prompt), stream=stream)

        if stream:
            # SSE streaming response with heartbeat to prevent client / proxy timeouts
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            heartbeat_stop = threading.Event()
            sse_lock = threading.Lock()

            def _sse_write(data: bytes):
                with sse_lock:
                    self.wfile.write(data)
                    self.wfile.flush()

            def _heartbeat():
                while not heartbeat_stop.wait(5.0):
                    try:
                        _sse_write(b": keep-alive\n\n")
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        break

            heartbeat_thread = threading.Thread(target=_heartbeat, daemon=True)
            heartbeat_thread.start()

            try:
                assembled, metrics = eng.jev_inference(prompt, raw_schema, max_batch=max_batch)
            except Exception as e:
                heartbeat_stop.set()
                heartbeat_thread.join(timeout=1.0)
                tb = traceback.format_exc()
                print(f"[jev-error] {e}\n{tb}", flush=True)
                try:
                    err_payload = json.dumps({"error": {"message": str(e), "type": "jev_error"}}, ensure_ascii=False)
                    _sse_write(f"data: {err_payload}\n\ndata: [DONE]\n\n".encode("utf-8"))
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
                return

            heartbeat_stop.set()
            heartbeat_thread.join(timeout=1.0)

            json_content = json.dumps(assembled, ensure_ascii=False)
            dt_total = time.perf_counter() - t0
            eff_tokens = metrics.get("completion_tokens", 0)
            if not eff_tokens:
                eff_tokens = len(eng.tok.encode(json_content, add_special_tokens=False)) if getattr(eng, "tok", None) else max(1, len(json_content.split()))

            if STATS_TRACKER:
                STATS_TRACKER.record_throughput(eff_tokens / max(dt_total, 1e-6), "jev-stream")
                reused = metrics.get("prefix_saved_tokens", 0)
                tot = metrics.get("prompt_tokens", 0)
                hit_str = "HIT" if metrics.get("cache_hit") else "MISS"
                STATS_TRACKER.record_cache_event(
                    f"Jev stream [schema {hit_str}]: {metrics.get('num_fields', 0)} fields, {eff_tokens} output toks, saved {reused}/{tot} tokens ({dt_total*1000:.1f}ms)",
                    event_type="hit" if metrics.get("cache_hit") else "info"
                )

            try:
                role_chunk = {
                    "id": rid,
                    "object": "chat.completion.chunk" if self.path == "/v1/chat/completions" else "text_completion",
                    "created": created,
                    "model": body.get("model") or eng.model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant"},
                            "finish_reason": None,
                        }
                    ],
                }
                _sse_write(f"data: {json.dumps(role_chunk, ensure_ascii=False)}\n\n".encode("utf-8"))

                for i in range(0, len(json_content), 128):
                    content_chunk = {
                        "id": rid,
                        "object": "chat.completion.chunk" if self.path == "/v1/chat/completions" else "text_completion",
                        "created": created,
                        "model": body.get("model") or eng.model_name,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": json_content[i:i + 128]},
                                "finish_reason": None,
                            }
                        ],
                    }
                    _sse_write(f"data: {json.dumps(content_chunk, ensure_ascii=False)}\n\n".encode("utf-8"))

                finish_chunk = {
                    "id": rid,
                    "object": "chat.completion.chunk" if self.path == "/v1/chat/completions" else "text_completion",
                    "created": created,
                    "model": body.get("model") or eng.model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": metrics.get("prompt_tokens", 0),
                        "completion_tokens": eff_tokens,
                        "total_tokens": metrics.get("prompt_tokens", 0) + eff_tokens,
                    },
                    "jev_result": assembled,
                    "jev_metrics": metrics,
                }
                _sse_write(f"data: {json.dumps(finish_chunk, ensure_ascii=False)}\n\n".encode("utf-8"))
                _sse_write(b"data: [DONE]\n\n")
            except (BrokenPipeError, ConnectionResetError, OSError):
                elapsed = time.perf_counter() - t0
                print(
                    f"[http] broken pipe elapsed={elapsed:.3f}s bytes={len(json_content)} stream=True",
                    flush=True,
                )
                if STATS_TRACKER:
                    STATS_TRACKER.record_disconnect()
            return

        # Non-streaming response
        try:
            assembled, metrics = eng.jev_inference(prompt, raw_schema, max_batch=max_batch)
        except Exception as e:
            tb = traceback.format_exc()
            print(f"[jev-error] {e}\n{tb}", flush=True)
            return self._json(500, {"error": str(e), "traceback": tb}, t0=t0, is_stream=False)

        json_content = json.dumps(assembled, ensure_ascii=False)
        dt_total = time.perf_counter() - t0
        eff_tokens = metrics.get("completion_tokens", 0)
        if not eff_tokens:
            eff_tokens = len(eng.tok.encode(json_content, add_special_tokens=False)) if getattr(eng, "tok", None) else max(1, len(json_content.split()))

        if STATS_TRACKER:
            STATS_TRACKER.record_throughput(eff_tokens / max(dt_total, 1e-6), "jev")
            reused = metrics.get("prefix_saved_tokens", 0)
            tot = metrics.get("prompt_tokens", 0)
            hit_str = "HIT" if metrics.get("cache_hit") else "MISS"
            STATS_TRACKER.record_cache_event(
                f"Jev [schema {hit_str}]: {metrics.get('num_fields', 0)} fields, {eff_tokens} output toks, saved {reused}/{tot} tokens ({dt_total*1000:.1f}ms)",
                event_type="hit" if metrics.get("cache_hit") else "info"
            )

        resp = {
            "id": rid,
            "object": "chat.completion" if self.path == "/v1/chat/completions" else "text_completion",
            "created": created,
            "model": body.get("model") or eng.model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": json_content,
                    },
                    "text": json_content,
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": metrics["prompt_tokens"],
                "completion_tokens": eff_tokens,
                "total_tokens": metrics["prompt_tokens"] + eff_tokens,
            },
            "jev_result": assembled,
            "jev_metrics": metrics,
        }
        return self._json(200, resp, t0=t0, is_stream=False)


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
    try:
        ENGINE.jev_engine.prefix_tree.init_system_prompt()
    except Exception as e:
        print(f"[jev-init] Note: Jev system prompt prefill deferred ({e})", flush=True)
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
