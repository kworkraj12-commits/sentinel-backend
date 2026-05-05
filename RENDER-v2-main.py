"""
Sentinel Backend v2 — FastAPI + YOLOv8 + SQLite Database
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import cv2, asyncio, time, threading, os, shutil, sqlite3
from datetime import datetime
from collections import deque
from detector import DetectorEngine
from models import Detection, Settings, LogEntry

app = FastAPI(title="Sentinel API v2", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Database Setup (SQLite) ────────────────────────────────────────────────
DB_PATH = "sentinel.db"

def init_db():
    """Create the logs table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id TEXT PRIMARY KEY,
            type TEXT,
            label TEXT,
            description TEXT,
            confidence REAL,
            timestamp TEXT,
            camera TEXT,
            known INTEGER,
            name TEXT
        )
    """)
    conn.commit()
    conn.close()
    print("✅ Database ready")

def save_to_db(entry: LogEntry):
    """Save one detection to the SQLite database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""
            INSERT OR IGNORE INTO detections
            (id, type, label, description, confidence, timestamp, camera, known, name)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (entry.id, entry.type, entry.label, entry.description,
              entry.confidence, entry.timestamp, entry.camera,
              1 if entry.known else 0 if entry.known is False else None,
              entry.name))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"DB error: {e}")

def load_from_db(limit=100, type_filter=None):
    """Load detections from database."""
    conn = sqlite3.connect(DB_PATH)
    if type_filter and type_filter != 'all':
        rows = conn.execute("SELECT * FROM detections WHERE type=? ORDER BY timestamp DESC LIMIT ?", (type_filter, limit)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM detections ORDER BY timestamp DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    cols = ['id','type','label','description','confidence','timestamp','camera','known','name']
    return [dict(zip(cols, r)) for r in rows]

# ── Global State ───────────────────────────────────────────────────────────
detector      = DetectorEngine()
recent_dets   = deque(maxlen=100)
connected_ws  = []
frame_lock    = threading.Lock()
latest_frame  = None
current_settings = Settings()

# ── Background Camera Thread ───────────────────────────────────────────────
def camera_loop():
    global latest_frame
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print("📷 Camera started")
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(1)
            cap = cv2.VideoCapture(0)
            continue
        annotated, detections = detector.detect(frame, current_settings)
        for det in detections:
            entry = LogEntry(
                id=str(time.time_ns()),
                type=det.type, label=det.label,
                description=det.description,
                confidence=round(det.confidence, 2),
                timestamp=datetime.now().strftime("%H:%M:%S"),
                camera="CAM-01",
                known=getattr(det,'known',None),
                name=getattr(det,'name',None)
            )
            recent_dets.appendleft(entry)
            save_to_db(entry)            # ← Save to SQLite
            asyncio.run(broadcast(entry))
        with frame_lock:
            latest_frame = annotated
        time.sleep(0.033)

async def broadcast(entry):
    dead = []
    for ws in connected_ws:
        try: await ws.send_json(entry.model_dump())
        except: dead.append(ws)
    for ws in dead:
        if ws in connected_ws: connected_ws.remove(ws)

@app.on_event("startup")
def startup():
    init_db()
    threading.Thread(target=camera_loop, daemon=True).start()
    print("🚀 Sentinel v2 started")

# ── Routes ─────────────────────────────────────────────────────────────────
@app.get("/")
def root(): return {"status":"ok","message":"Sentinel API v2 🛡️"}

@app.get("/health")
def health(): return {"status":"healthy","timestamp":datetime.now().isoformat()}

# ── Video Stream ───────────────────────────────────────────────────────────
def gen_frames():
    global latest_frame
    while True:
        with frame_lock: frame = latest_frame
        if frame is None: frame = detector.placeholder_frame()
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n'
        time.sleep(0.033)

@app.get("/stream")
def stream(): return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

# ── Detections ─────────────────────────────────────────────────────────────
@app.get("/detections")
def get_dets(limit:int=30): return list(recent_dets)[:limit]

@app.get("/detections/stats")
def get_stats():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT type, COUNT(*) FROM detections GROUP BY type").fetchall()
    conn.close()
    stats = {"face":0,"vehicle":0,"qr":0,"object":0}
    for t, c in rows:
        if t in stats: stats[t] = c
    return stats

# ── Logs (from DB) ─────────────────────────────────────────────────────────
@app.get("/logs")
def get_logs(limit:int=100, type:str=None):
    """Returns logs from SQLite database — persists across restarts."""
    return load_from_db(limit, type)

# ── Settings ───────────────────────────────────────────────────────────────
@app.get("/settings")
def get_settings(): return current_settings

@app.post("/settings")
def update_settings(s: Settings):
    global current_settings
    current_settings = s
    return {"status":"updated"}

# ── Face Recognition ───────────────────────────────────────────────────────
FACES_DIR = "known_faces"

@app.post("/faces/register")
async def register_face(name:str, file:UploadFile=File(...)):
    os.makedirs(FACES_DIR, exist_ok=True)
    path = os.path.join(FACES_DIR, f"{name}.jpg")
    with open(path, "wb") as f: shutil.copyfileobj(file.file, f)
    detector.load_known_faces(FACES_DIR)
    return {"status":"registered","name":name}

@app.get("/faces/known")
def list_faces():
    if not os.path.exists(FACES_DIR): return []
    return [f.replace(".jpg","") for f in os.listdir(FACES_DIR) if f.endswith(".jpg")]

@app.delete("/faces/{name}")
def delete_face(name:str):
    path = os.path.join(FACES_DIR, f"{name}.jpg")
    if os.path.exists(path):
        os.remove(path)
        detector.load_known_faces(FACES_DIR)
        return {"status":"deleted"}
    return {"status":"not_found"}

# ── WebSocket ──────────────────────────────────────────────────────────────
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    connected_ws.append(ws)
    try:
        while True: await ws.receive_text()
    except WebSocketDisconnect:
        if ws in connected_ws: connected_ws.remove(ws)
