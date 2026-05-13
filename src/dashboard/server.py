from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import sqlite3
import os

class Command(BaseModel):
    action: str
    params: dict = {}

COMMAND_FILE = "commands.json"
app = FastAPI(title="Axon Terminal API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure static dir exists
os.makedirs("src/dashboard/static", exist_ok=True)
app.mount("/static", StaticFiles(directory="src/dashboard/static"), name="static")

@app.get("/")
def read_root():
    return FileResponse("src/dashboard/static/index.html")

@app.get("/api/telemetry")
def get_telemetry():
    try:
        if os.path.exists("telemetry.json"):
            with open("telemetry.json", "r") as f:
                return json.load(f)
        return {"error": "No telemetry"}
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/command")
def post_command(cmd: Command):
    try:
        with open(COMMAND_FILE, "w") as f:
            json.dump(cmd.dict(), f)
        return {"status": "Command queued", "action": cmd.action}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/trades")
def get_trades():
    db_path = "axon_trades.db"
    if not os.path.exists(db_path):
        return []
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM trades ORDER BY timestamp ASC")
        rows = c.fetchall()
        conn.close()
        return [dict(ix) for ix in rows]
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
