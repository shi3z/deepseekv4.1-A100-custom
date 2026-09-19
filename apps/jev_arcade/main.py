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
    api_key: Optional[str] = None


@app.get("/api/games")
async def list_games():
    """Return all 20 mini-games metadata."""
    return {"games": GAMES, "total": len(GAMES)}


@app.get("/api/health")
async def health_check():
    """Check connectivity to local DeepSeek-V4.1 server and official TypeSafe API."""
    local_ok = False
    local_ms = 0.0
    try:
        t0 = time.perf_counter()
        req = urllib.request.Request(f"{LOCAL_SERVER_URL}/health", headers={"User-Agent": "JevArcade/1.0"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            if resp.status == 200:
                local_ok = True
        local_ms = (time.perf_counter() - t0) * 1000.0
    except Exception:
        pass

    return {
        "status": "ok",
        "local_engine": {
            "url": LOCAL_SERVER_URL,
            "connected": local_ok,
            "latency_ms": round(local_ms, 1),
            "model": "DeepSeek-V4.1-Flash (4x A100)",
        },
        "official_engine": {
            "url": "https://api.typesafe.ai/v1/systemone",
            "model": "jev-latest (Cloud)",
            "key_configured": bool(DEFAULT_TYPESAFE_KEY),
        },
        "tailscale_ip": "100.126.237.55",
    }


def call_local_jev(prompt: str, schema: dict) -> tuple[dict, float, int]:
    """Execute evaluation using local DeepSeek-V4.1 Jev Mode."""
    url = f"{LOCAL_SERVER_URL}/v1/chat/completions"
    payload = {
        "model": "deepseek-v4.1-flash",
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
    with urllib.request.urlopen(req, timeout=45) as resp:
        res = json.loads(resp.read().decode("utf-8"))
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


@app.post("/api/play")
async def play_game(req: PlayRequest):
    """Run an evaluation step for a specific game."""
    game = GAMES_MAP.get(req.game_id)
    if not game:
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
            result, elapsed_ms, tokens, metrics = call_local_jev(full_prompt, game["schema"])
            engine_name = "DeepSeek-V4.1 Jev Mode (Local 4x A100)"

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
