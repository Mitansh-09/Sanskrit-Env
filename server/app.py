import sys
import os
from pathlib import Path

# Ensure the project root is on sys.path so models/graders imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from openenv.core.env_server import create_fastapi_app
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from server.environment import SanskritEnvironment
from models import ManuscriptAction, ManuscriptObservation

app = create_fastapi_app(
    SanskritEnvironment,       # callable factory — calling it returns an Environment instance
    ManuscriptAction,          # action type class
    ManuscriptObservation,     # observation type class
)

# ── Web UI: serve static files and root route ─────────────────────────────────
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)

@app.get("/")
async def serve_ui():
    return FileResponse(os.path.join(static_dir, "index.html"))

app.mount("/static", StaticFiles(directory=static_dir), name="static")
