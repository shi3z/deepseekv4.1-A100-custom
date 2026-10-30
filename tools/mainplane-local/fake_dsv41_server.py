"""dsv41.serve with a fake engine: exercises the HTTP layer (stream splitter, tool_calls chunk, usage chunk,
prompt_cache_key plumbing) without a model.  Usage: python fake_dsv41_server.py PORT"""
import sys, json, threading, time
from http.server import ThreadingHTTPServer
import dsv41.serve as serve

CACHE = {}  # cache_key -> prompt tokens the fake "cached" (first request: nothing; second: the common prefix)

class FakeEngine:
    model_name = "deepseek-v4.1-flash-fake"
    max_seq_len = 65536
    max_seqs = 5
    last_finish_reason = "stop"
    last_decode_tok_s = 50.0
    decode_slot_ids = [1, 2, 3, 4]
    class tok:
        @staticmethod
        def decode(ids): return ""
        @staticmethod
        def encode(s): return [1] * len(s.split())
    def slot_cap_tokens(self, s): return 65536
    def format_chat(self, messages, thinking):
        # ids = one "token" per word of the serialized messages; the tools go in via the system message like the real encoder
        text = json.dumps(messages, ensure_ascii=False)
        return [hash(w) & 0xFFFF for w in text.split()], None, None
    def generate_stream(self, ids, params, images=None, token_types=None, req_id="", cache_key="", info=None):
        prev = CACHE.get(cache_key or "-")
        reused = 0
        if prev is not None:
            while reused < min(len(prev), len(ids)) and prev[reused] == ids[reused]:
                reused += 1
        CACHE[cache_key or "-"] = list(ids)
        pieces = ["<think>", "let me", " check", "</think>", "I will list", " the files.", " <｜DSML｜", "calls...>"]
        if "SAY_HELLO" in json.dumps(ids):
            pieces = ["hi"]
        for i, p in enumerate(pieces):
            yield i, p
        if info is not None:
            info.update(prefill={"reused_tokens": reused, "prefill_type": "gpu_hit" if reused else "cold"},
                        cached_tokens=reused, finish_reason="stop", decode_tok_s=50.0, cache_key=cache_key)
    def generate_text(self, ids, params, images=None, token_types=None, req_id="", cache_key="", info=None):
        out = [p for _, p in self.generate_stream(ids, params, images, token_types, req_id, cache_key, info)]
        return "".join(out), len(out)
    def parse_completion(self, text, thinking):
        if "<｜DSML｜" in text:
            return {"role": "assistant", "content": text.split("</think>")[-1].split("<｜DSML｜")[0].strip(),
                    "reasoning_content": "let me check",
                    "tool_calls": [{"id": "call_fake1", "type": "function",
                                    "function": {"name": "run", "arguments": {"worker": "admin", "code": "ls", "interpreter": "bash"}}}]}
        return {"role": "assistant", "content": text, "reasoning_content": "", "tool_calls": []}

serve.ENGINE = FakeEngine()
port = int(sys.argv[1])
srv = ThreadingHTTPServer(("127.0.0.1", port), serve.Handler)
print("fake dsv41 on", port, flush=True)
srv.serve_forever()
