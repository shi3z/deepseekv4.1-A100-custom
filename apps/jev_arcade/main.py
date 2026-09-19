"""FastAPI Backend for Jev Arcade 20 Mini-Games Suite."""

from __future__ import annotations

import json
import os
import time
import traceback
import urllib.request
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .games_data import GAMES

app = FastAPI(title="Jev Arcade 20", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LOCAL_SERVER_URL = os.environ.get("DSV41_SERVER_URL", "http://127.0.0.1:8000")
DEFAULT_TYPESAFE_KEY = os.environ.get(
    "TYPESAFE_API_KEY",
    "apikey_213088f1322b5885496b999b159a452ee161_0920d4a50e53f268ee01015aac9f679af6d74bb8d467312bc1bbaeb51ad85cad",
)

GAMES_MAP = {g["id"]: g for g in GAMES}


class PlayRequest(BaseModel):
    game_id: str
    user_input: str
    engine: str = "local"  # "local" | "official"
    model: Optional[str] = None
    api_key: Optional[str] = None


def get_current_local_model() -> str:
    """Fetch the currently active model from the local engine, defaulting to deepseek-v4.1-flash."""
    try:
        req = urllib.request.Request(f"{LOCAL_SERVER_URL}/v1/models", headers={"User-Agent": "JevArcade/1.0"})
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if data.get("data") and len(data["data"]) > 0:
                return data["data"][0]["id"]
    except Exception:
        pass
    return "deepseek-v4.1-flash"


def call_local_jev(prompt: str, schema: dict, model: Optional[str] = None) -> tuple[dict, float, int, dict]:
    """Execute evaluation using local DeepSeek-V4.1 Jev Mode, falling back to current active model if model differs."""
    url = f"{LOCAL_SERVER_URL}/v1/chat/completions"
    current_model = get_current_local_model()
    target_model = model or current_model

    payload = {
        "model": target_model,
        "prompt": prompt,
        "jev": True,
        "schema": schema,
    }

    t0 = time.perf_counter()
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            res = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if target_model != current_model:
            print(f"[JEV] Model '{target_model}' failed ({e.code}), falling back to current model '{current_model}'")
            payload["model"] = current_model
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=45) as resp:
                res = json.loads(resp.read().decode("utf-8"))
        else:
            raise

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    jev_result = res.get("jev_result") or {}
    metrics = res.get("jev_metrics") or {}
    tokens = res.get("usage", {}).get("total_tokens") or res.get("tokens", len(jev_result))
    return jev_result, elapsed_ms, tokens, metrics


def call_official_typesafe_jev(prompt: str, schema: dict, api_key: str) -> tuple[dict, float, int, dict]:
    """Execute evaluation using official TypeSafe System One Jev API."""
    from typesafe_sdk import Choice, Noul, TypeSafeClient

    client = TypeSafeClient(api_key=api_key)

    # Convert schema fields into Choice or Noul questions
    questions = {}
    for field_name, candidates in schema.items():
        if isinstance(candidates, list):
            if len(candidates) == 2 and all(isinstance(c, bool) for c in candidates):
                questions[field_name] = Noul(
                    instructions=f"Is '{field_name}' true or positive based on the input?",
                )
            else:
                criteria = {str(c): f"Option {c}" for c in candidates}
                questions[field_name] = Choice(
                    instructions=f"Select the most accurate value for '{field_name}' given the input scenario.",
                    criteria=criteria,
                )

    t0 = time.perf_counter()
    resp = client.system_one(state=prompt, questions=questions)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    jev_result = {}
    for k, v in resp.answers.items():
        if hasattr(v, "choice"):
            val = v.choice
            # convert back to bool/int if original schema had types
            for orig in schema.get(k, []):
                if str(orig) == val:
                    val = orig
                    break
            jev_result[k] = val
        elif hasattr(v, "noul"):
            jev_result[k] = v.noul > 0.5
        elif hasattr(v, "score"):
            jev_result[k] = v.score

    usage = getattr(resp, "usage", None)
    total_tokens = getattr(usage, "total_tokens", 50) if usage else 50
    return jev_result, elapsed_ms, total_tokens, {}


@app.get("/api/games")
async def list_games():
    """Return all 20 mini-games metadata."""
    return {"games": GAMES, "total": len(GAMES)}


@app.get("/api/health")
async def health_check():
    """Check connectivity to local DeepSeek-V4.1 server and official TypeSafe API."""
    local_ok = False
    local_ms = 0.0
    active_model = "DeepSeek-V4.1-Flash (4x A100)"
    try:
        t0 = time.perf_counter()
        req = urllib.request.Request(f"{LOCAL_SERVER_URL}/health", headers={"User-Agent": "JevArcade/1.0"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            if resp.status == 200:
                local_ok = True
        local_ms = (time.perf_counter() - t0) * 1000.0
        active_model = get_current_local_model()
    except Exception:
        pass

    return {
        "status": "ok",
        "local_engine": {
            "url": LOCAL_SERVER_URL,
            "connected": local_ok,
            "latency_ms": round(local_ms, 1),
            "model": active_model,
        },
        "official_engine": {
            "url": "https://api.typesafe.ai/v1/systemone",
            "model": "jev-latest (Cloud)",
            "key_configured": bool(DEFAULT_TYPESAFE_KEY),
        },
        "tailscale_ip": "100.126.237.55",
    }


@app.post("/api/client_error")
async def client_error(data: Dict[str, Any]):
    """Log client-side errors forwarded from iPad/browser."""
    print(f"🔥 [CLIENT ERROR from iPad]: {json.dumps(data, ensure_ascii=False)}")
    return {"status": "ok"}


@app.post("/api/play")
async def play_game(req: PlayRequest):
    """Run an evaluation step for a specific game."""
    print(f"\n👉 [PLAY REQUEST] game={req.game_id}, engine={req.engine}, model={req.model}, input={req.user_input[:60]!r}")
    game = GAMES_MAP.get(req.game_id)
    if not game:
        print(f"❌ [NOT FOUND] game_id={req.game_id}")
        raise HTTPException(status_code=404, detail=f"Game '{req.game_id}' not found")

    full_prompt = (
        f"【Game: {game['title']}】\n"
        f"Scenario: {game['scenario']}\n\n"
        f"Player Input: {req.user_input.strip()}\n\n"
        f"Task: Evaluate the player input against the game scenario and select the most accurate schema attributes."
    )

    engine_used = req.engine.lower()
    try:
        if engine_used == "official":
            key = req.api_key or DEFAULT_TYPESAFE_KEY
            if not key:
                raise HTTPException(status_code=400, detail="TypeSafe API key is required for official engine.")
            result, elapsed_ms, tokens, metrics = call_official_typesafe_jev(full_prompt, game["schema"], key)
            engine_name = "TypeSafe Jev (Cloud API)"
        else:
            result, elapsed_ms, tokens, metrics = call_local_jev(full_prompt, game["schema"], model=req.model)
            engine_name = "DeepSeek-V4.1 Jev Mode (Local 4x A100)"

        print(f"✅ [PLAY SUCCESS] {req.game_id} ({engine_name}) in {elapsed_ms:.1f}ms -> {result}")
        return {
            "status": "success",
            "game_id": req.game_id,
            "engine": engine_name,
            "elapsed_ms": round(elapsed_ms, 1),
            "tokens": tokens,
            "metrics": metrics,
            "result": result,
            "prompt_preview": full_prompt[:120] + "...",
        }
    except Exception as e:
        print(f"❌ [PLAY ERROR] {req.game_id}: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


# Serve static web assets
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.api_route("/", methods=["GET", "HEAD"])
async def root_index():
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "Jev Arcade API active. Place index.html in static/"}
