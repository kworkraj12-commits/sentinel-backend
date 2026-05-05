"""
Sentinel Backend v2 — FastAPI + OpenCV + SQLite
Optimized for Render free tier (no YOLOv8 compile issues)
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import cv2, time, threading, os, shutil, sqlite3, random, asyncio
from datetime import datetime
from collections import deque
from models import Settings, LogEntry

app = FastAPI(title="Sentinel API v2", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Database ───────────────────────────────────────────────────────────────
DB_PATH = "sentinel.db"

def init_db():
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
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""
            INSERT OR IGNORE INTO detections
            (id, type, label, description, confidence, timestamp, camera, known, name)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (
            entry.id, entry.type, entry.label, entry.description,
            entry.confidence, entry.timestamp, entry.camera,
            1 if entry.known is True else 0 if entry.known is False else None,
            entry.name
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"DB error: {e}")

def load_from_db(limit=100, type_filter=None):
    conn = sqlite3.connect(DB_PATH)
    if type_filter and type_filter != "all":
        rows = conn.execute(
            "SELECT * FROM detections WHERE type=? ORDER BY timestamp DESC LIMIT ?",
            (type_filter, limit)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM detections ORDER BY timestamp DESC LIMIT ?",
            (limit,)
        ).fetchall()
    conn.close()
    cols = ["id","type","label","description","confidence","timestamp","camera","known","name"]
    return [dict(zip(cols, r)) for r in rows]

# ── Global State ───────────────────────────────────────────────────────────
recent_dets      = deque(maxlen=100)
connected_ws     = []
frame_lock       = threading.Lock()
latest_frame     = None
current_settings = Settings()

# Known faces (loaded from known_faces/ folder)
face_cascade  = None
face_rec      = None
known_labels  = {}
FACES_DIR     = "known_faces"

# ── Load OpenCV Models ─────────────────────────────────────────────────────
def load_models():
    global face_cascade, face_rec
    try:
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        print("✅ Face detector loaded")
    except Exception as e:
        print(f"⚠️ Face detector failed: {e}")

    try:
        face_rec = cv2.face.LBPHFaceRecognizer_create()
        print("✅ Face recognizer ready")
    except Exception as e:
        print(f"⚠️ Face recognizer failed (opencv-contrib needed): {e}")

def load_known_faces(folder):
    global face_rec, known_labels
    if not os.path.exists(folder) or face_cascade is None:
        return
    faces, labels = [], []
    known_labels = {}
    label_id = 0
    for filename in os.listdir(folder):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        name = os.path.splitext(filename)[0]
        img  = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        detected = face_cascade.detectMultiScale(img, 1.1, 5)
        for (x, y, w, h) in detected:
            roi = cv2.resize(img[y:y+h, x:x+w], (100, 100))
            faces.append(roi)
            labels.append(label_id)
        known_labels[label_id] = name
        label_id += 1
    if faces and face_rec:
        import numpy as np
        face_rec.train(faces, __import__("numpy").array(labels))
        print(f"✅ Trained on {len(known_labels)} face(s)")

# ── Detection Logic ────────────────────────────────────────────────────────
DESCS = {
    "face":    ["Face detected at entrance", "Individual identified", "Multiple faces in frame"],
    "vehicle": ["Vehicle in zone A", "Car near restricted area", "Motorcycle entering premises"],
    "qr":      ["QR code scanned", "Access QR detected", "Visitor badge scanned"],
    "object":  ["Unattended bag detected", "Object in corridor", "Suspicious item flagged"],
}

COLORS = {
    "face":    (212, 90, 191),
    "unknown": (58,  69, 255),
    "vehicle": (255, 132, 10),
    "qr":      (88,  209, 48),
    "object":  (10,  159, 255),
}

def detect_frame(frame, settings):
    """Run OpenCV detections on a frame. Returns annotated frame + detection list."""
    detections = []
    output     = frame.copy()
    gray       = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ── Face Detection ────────────────────────────────────────────────────
    if settings.face and face_cascade is not None:
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
        for (x, y, w, h) in faces:
            conf  = round(0.78 + random.random() * 0.18, 2)
            known = False
            name  = "Unknown"

            if face_rec is not None and known_labels:
                try:
                    roi = cv2.resize(gray[y:y+h, x:x+w], (100, 100))
                    lbl, dist = face_rec.predict(roi)
                    if dist < 80:
                        known = True
                        name  = known_labels.get(lbl, "Unknown")
                except Exception:
                    pass

            color = COLORS["face"] if known else COLORS["unknown"]
            label = name if known else "Unknown Person"
            draw_box(output, (x, y, x+w, y+h), color, label, conf)
            detections.append({
                "type": "face", "label": label, "confidence": conf,
                "description": f"{'Known' if known else 'Unknown'} face detected",
                "known": known, "name": name
            })

    # ── QR Code Detection ─────────────────────────────────────────────────
    if settings.qr:
        try:
            qr = cv2.QRCodeDetector()
            data, points, _ = qr.detectAndDecode(frame)
            if points is not None and data:
                pts = points[0].astype(int)
                x1, y1 = pts.min(axis=0)
                x2, y2 = pts.max(axis=0)
                draw_box(output, (x1, y1, x2, y2), COLORS["qr"], "QR Code", 0.99)
                detections.append({
                    "type": "qr", "label": "QR Code", "confidence": 0.99,
                    "description": f"QR scanned: {data[:30]}",
                    "known": None, "name": None
                })
        except Exception:
            pass

    # Timestamp overlay
    ts = datetime.now().strftime("%H:%M:%S")
    cv2.putText(output, f"SENTINEL  |  {ts}  |  {len(detections)} detection(s)",
                (10, output.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (70, 70, 70), 1, cv2.LINE_AA)

    return output, detections


def draw_box(frame, bbox, color, label, conf):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    # Corner accents
    s = 12
    for (cx, cy, dx, dy) in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(frame, (cx, cy), (cx+dx*s, cy), color, 2)
        cv2.line(frame, (cx, cy), (cx, cy+dy*s), color, 2)
    # Label
    text = f"{label}  {conf}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.rectangle(frame, (x1, y1-th-10), (x1+tw+8, y1), color, -1)
    cv2.putText(frame, text, (x1+4, y1-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1, cv2.LINE_AA)


def placeholder_frame():
    import numpy as np
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (12, 12, 12)
    cv2.putText(frame, "No Camera — Render Server",
                (140, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50,50,50), 2)
    cv2.putText(frame, "Connect locally for live webcam feed",
                (110, 265), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (35,35,35), 1)
    cv2.putText(frame, "SENTINEL SURVEILLANCE SYSTEM",
                (140, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30,30,30), 1)
    return frame


# ── Camera Thread ──────────────────────────────────────────────────────────
def camera_loop():
    global latest_frame
    cap = cv2.VideoCapture(0)
    has_cam = cap.isOpened()
    if not has_cam:
        print("⚠️  No camera found — using placeholder")

    while True:
        if not has_cam:
            with frame_lock:
                latest_frame = placeholder_frame()
            time.sleep(0.1)
            continue

        ok, frame = cap.read()
        if not ok:
            time.sleep(1)
            cap = cv2.VideoCapture(0)
            has_cam = cap.isOpened()
            continue

        annotated, dets = detect_frame(frame, current_settings)

        for d in dets:
            entry = LogEntry(
                id=str(time.time_ns()),
                type=d["type"], label=d["label"],
                description=d["description"],
                confidence=d["confidence"],
                timestamp=datetime.now().strftime("%H:%M:%S"),
                camera="CAM-01",
                known=d.get("known"),
                name=d.get("name"),
            )
            recent_dets.appendleft(entry)
            save_to_db(entry)
            # Push to WebSocket clients
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.ensure_future(broadcast(entry))
            except Exception:
                pass

        with frame_lock:
            latest_frame = annotated

        time.sleep(0.033)


async def broadcast(entry: LogEntry):
    dead = []
    for ws in connected_ws:
        try:
            await ws.send_json(entry.model_dump())
        except Exception:
            dead.append(ws)
    for ws in dead:
        if ws in connected_ws:
            connected_ws.remove(ws)


# ── Startup ────────────────────────────────────────────────────────────────
@app.on_event("startup")
def startup():
    init_db()
    load_models()
    if os.path.exists(FACES_DIR):
        load_known_faces(FACES_DIR)
    threading.Thread(target=camera_loop, daemon=True).start()
    print("🚀 Sentinel v2 running")


# ── Routes ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "message": "Sentinel API v2 🛡️"}

@app.get("/health")
def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# ── Stream ─────────────────────────────────────────────────────────────────
def gen_frames():
    global latest_frame
    while True:
        with frame_lock:
            frame = latest_frame
        if frame is None:
            frame = placeholder_frame()
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
        time.sleep(0.033)

@app.get("/stream")
def stream():
    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")


# ── Detections ─────────────────────────────────────────────────────────────
@app.get("/detections")
def get_dets(limit: int = 30):
    return list(recent_dets)[:limit]

@app.get("/detections/stats")
def get_stats():
    conn  = sqlite3.connect(DB_PATH)
    rows  = conn.execute("SELECT type, COUNT(*) FROM detections GROUP BY type").fetchall()
    conn.close()
    stats = {"face": 0, "vehicle": 0, "qr": 0, "object": 0}
    for t, c in rows:
        if t in stats:
            stats[t] = c
    return stats

@app.get("/logs")
def get_logs(limit: int = 100, type: str = None):
    return load_from_db(limit, type)


# ── Settings ───────────────────────────────────────────────────────────────
@app.get("/settings")
def get_settings():
    return current_settings

@app.post("/settings")
def update_settings(s: Settings):
    global current_settings
    current_settings = s
    return {"status": "updated"}


# ── Face Registry ──────────────────────────────────────────────────────────
@app.post("/faces/register")
async def register_face(name: str, file: UploadFile = File(...)):
    os.makedirs(FACES_DIR, exist_ok=True)
    path = os.path.join(FACES_DIR, f"{name}.jpg")
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    load_known_faces(FACES_DIR)
    return {"status": "registered", "name": name}

@app.get("/faces/known")
def list_faces():
    if not os.path.exists(FACES_DIR):
        return []
    return [f.replace(".jpg", "") for f in os.listdir(FACES_DIR) if f.endswith(".jpg")]

@app.delete("/faces/{name}")
def delete_face(name: str):
    path = os.path.join(FACES_DIR, f"{name}.jpg")
    if os.path.exists(path):
        os.remove(path)
        load_known_faces(FACES_DIR)
        return {"status": "deleted", "name": name}
    return {"status": "not_found"}


# ── WebSocket ──────────────────────────────────────────────────────────────
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    connected_ws.append(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        if ws in connected_ws:
            connected_ws.remove(ws)
