import sys
import os
from pathlib import Path
from typing import Optional

# Ensure the project root is on sys.path so models/graders imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from openenv.core.env_server import create_fastapi_app
from fastapi import HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from server.environment import SanskritEnvironment
from server.model_agent import get_model_catalog, run_model_episode
from models import ManuscriptAction, ManuscriptObservation


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _first_nonempty_env(*names: str) -> tuple[str, Optional[str]]:
    for name in names:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip(), name
    return "", None


load_dotenv()

HF_TOKEN_ENV_KEYS = (
    "HF_TOKEN",
    "HUGGINGFACEHUB_API_TOKEN",
    "HUGGINGFACE_API_TOKEN",
    "HF_API_TOKEN",
    "HF_API_KEY",
    "HUGGINGFACE_TOKEN",
)

HF_TOKEN, HF_TOKEN_SOURCE = _first_nonempty_env(*HF_TOKEN_ENV_KEYS)
HF_ROUTER_URL = os.environ.get("HF_ROUTER_URL", "https://router.huggingface.co/v1/chat/completions")
HF_UI_MODELS = os.environ.get("HF_UI_MODELS", "")

HF_TEMPERATURE = _env_float("HF_TEMPERATURE", 0.0)
HF_MAX_TOKENS = _env_int("HF_MAX_TOKENS", 512)
HF_RETRY_WAIT = _env_int("RETRY_WAIT", 5)
HF_REQUEST_TIMEOUT = _env_int("HF_REQUEST_TIMEOUT", 90)


class ModelEpisodeRequest(BaseModel):
    task_id: str
    model_id: str
    seed: Optional[int] = None
    episode_id: Optional[str] = None

# Instantiate the environment once (singleton) to maintain session state
env_instance = SanskritEnvironment()

app = create_fastapi_app(
    lambda: env_instance,      # factory returns the singleton instance
    ManuscriptAction,          # action type class
    ManuscriptObservation,     # observation type class
)

# ── Web UI: serve static files and root route ─────────────────────────────────
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)

@app.get("/")
async def serve_ui():
    return FileResponse(os.path.join(static_dir, "index.html"))

@app.get("/session")
async def check_session():
    return {"active_sessions": list(env_instance._sessions.keys())}


@app.get("/model/options")
async def model_options():
    models = get_model_catalog(HF_UI_MODELS)
    return {
        "models": models,
        "token_configured": bool(HF_TOKEN),
        "token_source": HF_TOKEN_SOURCE,
        "token_env_keys": list(HF_TOKEN_ENV_KEYS),
        "router_url": HF_ROUTER_URL,
        "note": "HF free-tier availability can change by provider load or policy.",
    }


@app.post("/model/run")
async def model_run(payload: ModelEpisodeRequest):
    if not HF_TOKEN:
        expected = ", ".join(HF_TOKEN_ENV_KEYS)
        raise HTTPException(
            status_code=400,
            detail=f"HF token is not configured on the server. Set one of: {expected}.",
        )

    models = get_model_catalog(HF_UI_MODELS)
    allowed = {model["id"] for model in models}
    if payload.model_id not in allowed:
        raise HTTPException(status_code=400, detail="Selected model is not in the configured HF UI model list.")

    try:
        return run_model_episode(
            env=env_instance,
            task_id=payload.task_id,
            model_id=payload.model_id,
            hf_token=HF_TOKEN,
            router_url=HF_ROUTER_URL,
            temperature=HF_TEMPERATURE,
            max_tokens=HF_MAX_TOKENS,
            retry_wait=HF_RETRY_WAIT,
            request_timeout=HF_REQUEST_TIMEOUT,
            seed=payload.seed,
            episode_id=payload.episode_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unexpected model run error: {exc}")

app.mount("/static", StaticFiles(directory=static_dir), name="static")
