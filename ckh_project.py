# app.py
# Streamlit Face Check-In with PRE-CAPTURE face detection (via WebRTC) and DeepFace matching.
# - Enroll employees (save reference images)
# - Live camera with green box when a face is present; Capture is disabled until a face is detected
# - Check in by matching against employees_db with DeepFace.find
# - View/download attendance log (CSV)
#
# Install (Streamlit Cloud / local):
#   pip install streamlit deepface streamlit-webrtc opencv-python-headless pandas pillow
#   # If you switch MODEL_NAME to "SFace", also: pip install onnxruntime

import os
import time
from datetime import datetime
from pathlib import Path

# Quieter native logs & fewer threads (helps avoid noisy mutex logs)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("FLAGS_minloglevel", "3")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import cv2
cv2.setNumThreads(1)

import streamlit as st
import pandas as pd
from PIL import Image
from deepface import DeepFace
import numpy as np
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoTransformerBase

# ----------------------
# Config / Paths
# ----------------------
DATA_DIR = Path("data")
DB_DIR = DATA_DIR / "employees_db"     # where enrollment photos are stored
CHECKINS_DIR = DATA_DIR / "checkins"   # where check-in snapshots are stored
ATTENDANCE_CSV = DATA_DIR / "attendance.csv"

# Default DeepFace settings (you can change in UI)
DEFAULT_MODEL_NAME = "ArcFace"    # "ArcFace" | "Facenet" | "VGG-Face" | "SFace"
DEFAULT_DETECTOR   = "opencv"     # "opencv" | "retinaface" | "mtcnn" | "mediapipe"
DEFAULT_THRESHOLD  = 0.80         # For ArcFace (cosine distance) a common threshold is ~0.68; adjust to your data

# ----------------------
# Setup
# ----------------------
def ensure_dirs():
    DB_DIR.mkdir(parents=True, exist_ok=True)
    CHECKINS_DIR.mkdir(parents=True, exist_ok=True)
    ATTENDANCE_CSV.parent.mkdir(parents=True, exist_ok=True)
    if not ATTENDANCE_CSV.exists():
        pd.DataFrame(columns=["timestamp", "employee", "match_distance", "image_path"]).to_csv(
            ATTENDANCE_CSV, index=False
        )

def employee_folder(name: str) -> Path:
    safe = name.strip().replace("/", "_")
    p = DB_DIR / safe
    p.mkdir(parents=True, exist_ok=True)
    return p

def get_employee_names() -> list[str]:
    ensure_dirs()
    return sorted([p.name for p in DB_DIR.iterdir() if p.is_dir()])

def save_uploaded_image(uploaded_file, dest_dir: Path, prefix: str) -> Path:
    img = Image.open(uploaded_file).convert("RGB")
    fname = f"{prefix}_{int(datetime.now().timestamp())}.jpg"
    path = dest_dir / fname
    img.save(path, "JPEG", quality=95)
    return path

def save_pil_image(img: Image.Image, dest_dir: Path, prefix: str) -> Path:
    img = img.convert("RGB")
    fname = f"{prefix}_{int(datetime.now().timestamp())}.jpg"
    path = dest_dir / fname
    img.save(path, "JPEG", quality=95)
    return path

def add_attendance_row(employee: str, distance: float, image_path: Path):
    ensure_dirs()
    df = pd.read_csv(ATTENDANCE_CSV)
    ts = datetime.now().isoformat(timespec="seconds")
    df.loc[len(df)] = [ts, employee, float(distance), str(image_path)]
    df.to_csv(ATTENDANCE_CSV, index=False)

def _normalize_results(results):
    """
    DeepFace.find may return a single DataFrame or a list[DataFrame] (one per detected face).
    Normalize to list[DataFrame] and drop empties.
    """
    if results is None:
        return []
    if isinstance(results, list):
        return [df for df in results if hasattr(df, "empty") and not df.empty]
    return [results] if hasattr(results, "empty") and not results.empty else []

def run_face_search(img_path: str, model_name: str, detector: str):
    """
    Use DeepFace.find to search img_path face in DB_DIR.
    Returns (best_name, best_distance, best_identity_path) or (None, None, None).
    Tries the chosen detector first, then a few safe fallbacks.
    """
    try_order = [(model_name, detector)]
    # Add a couple of stable fallbacks
    if detector != "opencv":
        try_order.append((model_name, "opencv"))
    if model_name != "SFace":
        try_order.append(("SFace", "opencv"))   # very stable CPU path (needs onnxruntime)
    try_order.append(("Facenet", "opencv"))

    last_err = None
    for mdl, det in try_order:
        try:
            results = DeepFace.find(
                img_path=img_path,
                db_path=str(DB_DIR),
                model_name=mdl,
                detector_backend=det,
                enforce_detection=True,
                silent=True,
            )
            dfs = _normalize_results(results)
            if not dfs:
                continue
            df = dfs[0].sort_values("distance", ascending=True).reset_index(drop=True)
            best = df.iloc[0]
            identity_path = best["identity"]
            distance = float(best["distance"])

            # Infer employee name from folder or filename
            name = os.path.basename(os.path.dirname(identity_path))
            if not name or name == DB_DIR.name:
                name = os.path.splitext(os.path.basename(identity_path))[0]

            return name, distance, identity_path, mdl, det
        except Exception as e:
            last_err = e
            continue

    st.error("No face detected or no match could be computed.")
    if last_err:
        st.caption(f"Last error: {last_err}")
    return None, None, None, None, None

# ----------------------
# Live pre-capture face detection (WebRTC)
# ----------------------
class FaceDetectTransformer(VideoTransformerBase):
    def __init__(self):
        self.cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.face_count = 0
        self.last_frame_bgr = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.last_frame_bgr = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )
        self.face_count = 0 if faces is None else len(faces)
        if faces is not None:
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return img

# ----------------------
# Streamlit UI
# ----------------------
ensure_dirs()
st.set_page_config(page_title="Face Check-In (DeepFace + Pre-Capture Check)", page_icon="🧑‍💼", layout="centered")
st.title("🧑‍💼 Face Recognition Check-In")

tab_enroll, tab_checkin, tab_log = st.tabs(["👤 Enroll", "✅ Check In (Live)", "📜 Attendance Log"])

# ---------- Enroll Tab ----------
with tab_enroll:
    st.subheader("Enroll an employee")
    st.caption("Save 1–3 clear, frontal photos per employee in a personal folder.")

    name = st.text_input("Employee name", placeholder="e.g., Jane Doe")
    files = st.file_uploader("Upload 1–3 images (JPG/PNG)", type=["jpg","jpeg","png"], accept_multiple_files=True)
    use_cam = st.checkbox("Use camera instead of files")
    cam_img = st.camera_input("Capture a photo") if use_cam else None

    if st.button("Save to gallery", type="primary", use_container_width=True):
        if not name.strip():
            st.error("Please enter an employee name.")
        else:
            dest = employee_folder(name)
            saved = False
            for f in files or []:
                save_uploaded_image(f, dest, prefix=name.replace(" ", "_"))
                saved = True
            if cam_img is not None:
                save_pil_image(Image.open(cam_img), dest, prefix=name.replace(" ", "_"))
                saved = True
            if saved:
                st.success(f"Saved enrollment images to `{dest}`")
            else:
                st.warning("No images provided.")

    st.divider()
    st.caption("Currently enrolled employees:")
    employees = get_employee_names()
    st.write(", ".join(employees) if employees else "— none yet —")

# ---------- Check-In (Live Pre-Check) Tab ----------
with tab_checkin:
    st.subheader("Check in (live pre-capture face detection)")
    st.caption("Capture becomes enabled only when at least one face is detected in the frame.")

    # Choose recognition parameters (these values are used when you click capture)
    col_m, col_d, col_t = st.columns(3)
    with col_m:
        model_name = st.selectbox("Model", ["ArcFace", "Facenet", "VGG-Face", "SFace"], index=["ArcFace","Facenet","VGG-Face","SFace"].index(DEFAULT_MODEL_NAME))
    with col_d:
        detector = st.selectbox("Detector", ["opencv", "retinaface", "mtcnn", "mediapipe"], index=["opencv","retinaface","mtcnn","mediapipe"].index(DEFAULT_DETECTOR))
    with col_t:
        threshold = st.number_input("Distance threshold (≤ to accept)", min_value=0.10, max_value=1.00, step=0.01, value=float(DEFAULT_THRESHOLD))

    # Start live webcam with face detection overlay
    ctx = webrtc_streamer(
        key="checkin-stream",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={"video": True, "audio": False},
        video_transformer_factory=FaceDetectTransformer,
        async_processing=True,
    )

    face_present = False
    if ctx and ctx.video_transformer:
        cnt = ctx.video_transformer.face_count
        face_present = cnt > 0
        (st.info if cnt else st.warning)(f"Faces detected: {cnt}" if cnt else "No face detected yet…")

    capture = st.button("📸 Capture & Check In", type="primary", use_container_width=True, disabled=not face_present)

    if capture:
        if not ctx or not ctx.video_transformer or ctx.video_transformer.last_frame_bgr is None:
            st.error("No frame available from camera.")
        else:
            # Convert last live frame to PIL and save temp
            bgr = ctx.video_transformer.last_frame_bgr
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            tmp_path = str(Path("tmp_capture.jpg"))
            pil.save(tmp_path, "JPEG", quality=95)

            with st.spinner("Matching face…"):
                name, distance, ref_path, used_model, used_det = run_face_search(tmp_path, model_name=model_name, detector=detector)

            if name is None:
                st.error("No matching face found or face not detected. Try again with frontal lighting and one face in frame.")
            else:
                st.write(f"**Best match**: {name}")
                st.write(f"**Match distance**: {distance:.4f}  (accept if ≤ {threshold:.2f})")
                if used_model and used_det:
                    st.caption(f"Used model: {used_model} | detector: {used_det}")

                if distance <= threshold:
                    CHECKINS_DIR.mkdir(parents=True, exist_ok=True)
                    out_path = CHECKINS_DIR / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    pil.save(out_path, "JPEG", quality=95)

                    # Optional: verify saved file
                    try:
                        size = os.path.getsize(out_path)
                        st.toast(f"Saved {out_path} ({size} bytes)", icon="✅")
                    except FileNotFoundError:
                        st.error("Save failed (file missing after write).")

                    add_attendance_row(name, distance, out_path)
                    st.success(f"✅ Check-in recorded for **{name}** at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    st.image(pil, caption="Saved check-in image", use_column_width=True)
                else:
                    st.warning("Face found but distance is above threshold. Not recording.")

# ---------- Log Tab ----------
with tab_log:
    st.subheader("Attendance log")
    if ATTENDANCE_CSV.exists():
        df = pd.read_csv(ATTENDANCE_CSV)
        if df.empty:
            st.info("No attendance yet.")
        else:
            st.dataframe(df.sort_values("timestamp", ascending=False), use_column_width=True)
            st.download_button(
                "Download CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="attendance.csv",
                mime="text/csv",
                use_container_width=True,
            )
    else:
        st.info("No attendance yet.")

st.caption("Pre-capture face check uses OpenCV; matching uses DeepFace against images in data/employees_db/")
