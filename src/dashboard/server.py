from fastapi import FastAPI, HTTPException, Request
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

@app.post("/api/telemetry")
async def post_telemetry(request: Request):
    try:
        raw_body = await request.body()
        # Decode and strip any trailing null bytes or garbage
        clean_body = raw_body.decode('utf-8', errors='ignore').strip().split('}')[0] + '}'
        data = json.loads(clean_body)
        
        # Write Atomic Pulse for the Neural Engine (Bypassing fragile ZMQ sockets)
        with open("pulse.json", "w") as f:
            json.dump(data, f)
            
        return {"status": "success", "pulse": "sent"}
    except Exception as e:
        print(f"[PULSE ERROR] {e}")
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

@app.get("/api/logs")
def get_logs():
    log_path = "axon_ai.log"
    if not os.path.exists(log_path):
        return ["> No logs found. Waiting for engine..."]
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
            if not lines:
                return ["> Synchronizing AI Monologue..."]
            return [line.strip() for line in lines[-30:]]
    except Exception as e:
        return [f"> [SYSTEM ERROR] Log Access Restricted: {str(e)}"]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
