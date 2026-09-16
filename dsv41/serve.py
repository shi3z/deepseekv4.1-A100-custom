from pathlib import Path
"""OpenAI-compatible HTTP server (no web framework needed: stdlib http.server, threaded).

  python -m dsv41.serve --devices 2,0,1,4,5,6,7,3 --port 8000
  curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
       -d '{"model":"deepseek-v4.1-flash","messages":[{"role":"user","content":"hello"}],"stream":true}'

Endpoints: GET /v1/models, POST /v1/chat/completions (stream or not), POST /v1/completions, GET /health.
One request is generated at a time; others wait on the engine lock."""
import torch
import argparse
import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import os as _os
import traceback
_os.environ.setdefault("OMP_WAIT_POLICY", "active")  # CPU expert threads keep spinning between layers (libgomp reads this once)
from .engine import Engine, GenParams, parse_budgets

ENGINE: Engine | None = None


def _params(body: dict) -> GenParams:
    stop = body.get("stop") or []
    if isinstance(stop, str):
        stop = [stop]
    return GenParams(
        max_new_tokens=int(body.get("max_tokens") or body.get("max_completion_tokens") or 1024),
        temperature=float(body.get("temperature", 0.6)),
        top_p=float(body.get("top_p", 0.95)),
        stop=list(stop),
        seed=body.get("seed"),
    )


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quieter log
        print(f"[{time.strftime('%H:%M:%S')}] {self.address_string()} {fmt % args}", flush=True)

    def _json(self, code: int, obj: dict):
        data = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/v1/models":
            self._json(200, {"object": "list", "data": [{"id": ENGINE.model_name, "object": "model", "owned_by": "local"}]})
        elif self.path == "/health":
            self._json(200, {"status": "ok"})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            return self._json(400, {"error": "invalid JSON"})
        if self.path == "/v1/chat/completions":
            return self._chat(body)
        if self.path == "/v1/completions":
            return self._completion(body)
        self._json(404, {"error": "not found"})

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
        messages = body.get("messages") or []
        thinking = "thinking" if body.get("reasoning_effort") or body.get("thinking") else None
        try:
            ids = eng.tok.encode(eng.chat_prompt(messages, thinking))
        except Exception as e:  # malformed messages / unsupported content
            return self._json(400, {"error": f"cannot encode messages: {e}"})
        params = _params(body)
        rid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        print(
          f"[chat] prompt_tokens={len(ids)} "
          f"max_seq_len={eng.max_seq_len} "
          f"max_tokens={body.get('max_tokens')} "
          f"stream={body.get('stream')}",
          flush=True,
        )
        if body.get("stream"):
            # Generate before sending HTTP 200 so generation failures can
            # still be returned as proper HTTP errors.
            try:
                text, n = eng.generate_text(ids, params)

            except torch.OutOfMemoryError as e:
                torch.cuda.empty_cache()
                return self._json(
                    507,
                    {
                        "error": {
                            "message": f"CUDA out of memory: {e}",
                            "type": "cuda_out_of_memory",
                        }
                    },
                )

            except Exception as e:
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

                return self._json(
                    500,
                    {
                        "error": {
                            "message": str(e),
                            "type": "generation_error",
                            "traceback": tb,
                        }
                    },
                )

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

            # Generation succeeded. Now it is safe to send HTTP 200.
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            def chunk(delta, finish=None):
                obj = {
                    "id": rid,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": eng.model_name,
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

                self.wfile.write(payload.encode("utf-8"))
                self.wfile.flush()

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
                elif n >= params.max_new_tokens:
                    finish = "length"
                else:
                    finish = "stop"

                chunk({}, finish)

                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()

            except (BrokenPipeError, ConnectionResetError):
                return

            return
        text, n = eng.generate_text(ids, params)
        msg = eng.parse_completion(text, thinking)
        content = msg.get("content") if isinstance(msg, dict) else text
        out = {"id": rid, "object": "chat.completion", "created": created, "model": eng.model_name,
               "choices": [{"index": 0, "message": {"role": "assistant", "content": content if content is not None else text},
                            "finish_reason": "length" if n >= params.max_new_tokens else "stop"}],
               "usage": {"prompt_tokens": len(ids), "completion_tokens": n, "total_tokens": len(ids) + n}}
        if isinstance(msg, dict) and msg.get("reasoning_content"):
            out["choices"][0]["message"]["reasoning_content"] = msg["reasoning_content"]
        if isinstance(msg, dict) and msg.get("tool_calls"):
            out["choices"][0]["message"]["tool_calls"] = msg["tool_calls"]
        self._json(200, out)

    # ---------------------------------------------------------------- raw completions
    def _completion(self, body: dict):
        eng = ENGINE
        prompt = body.get("prompt") or ""
        if isinstance(prompt, list):
            prompt = prompt[0]
        ids = eng.tok.encode(prompt)
        params = _params(body)
        rid = f"cmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            n = 0
            for _, piece in eng.generate(ids, params):
                obj = {"id": rid, "object": "text_completion", "created": created, "model": eng.model_name,
                       "choices": [{"index": 0, "text": piece, "finish_reason": None}]}
                self.wfile.write(f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode())
                self.wfile.flush()
                n += 1
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return
        text, n = eng.generate_text(ids, params)
        self._json(200, {"id": rid, "object": "text_completion", "created": created, "model": eng.model_name,
                         "choices": [{"index": 0, "text": text, "finish_reason": "length" if n >= params.max_new_tokens else "stop"}],
                         "usage": {"prompt_tokens": len(ids), "completion_tokens": n, "total_tokens": len(ids) + n}})


def main():
    global ENGINE
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--devices", default="2,0,1,4,5,6,7,3")
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
    a = ap.parse_args()
    kw = dict(devices=[int(d) for d in a.devices.split(",")], max_seq_len=a.max_seq_len, budgets=parse_budgets(a.budgets),
              use_graphs=not a.no_graphs, offload_experts=a.offload_experts, hot_experts=a.hot_experts, route_stats=a.hot_stats,
              ep=a.ep, ep_shards=[int(v) for v in a.ep_shards.split(",")] if a.ep_shards else None, mtp=a.mtp, mtp_device=a.mtp_device)
    ENGINE = Engine(a.ckpt, **kw) if a.ckpt else Engine(**kw)
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"serving OpenAI-compatible API on http://{a.host}:{a.port}/v1 (model '{ENGINE.model_name}')", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
