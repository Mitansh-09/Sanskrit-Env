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

app.mount("/static", StaticFiles(directory=static_dir), name="static")
