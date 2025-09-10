import os
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd
from PIL import Image
import cv2
import numpy as np

# ----------------------
# Config / Paths
# ----------------------
DATA_DIR = Path("data")
DB_DIR = DATA_DIR / "employees_db"     # enrollment photos
CHECKINS_DIR = DATA_DIR / "checkins"   # check-in snapshots
ATTENDANCE_CSV = DATA_DIR / "attendance.csv"

# ----------------------
# Setup
# ----------------------
def ensure_dirs():
    DB_DIR.mkdir(parents=True, exist_ok=True)
    CHECKINS_DIR.mkdir(parents=True, exist_ok=True)
    ATTENDANCE_CSV.parent.mkdir(parents=True, exist_ok=True)
    if not ATTENDANCE_CSV.exists():
        pd.DataFrame(
            columns=["timestamp", "employee", "event_type", "image_path"]
        ).to_csv(ATTENDANCE_CSV, index=False)

def employee_folder(name: str) -> Path:
    safe = name.strip().replace("/", "_")
    p = DB_DIR / safe
    p.mkdir(parents=True, exist_ok=True)
    return p

def get_employee_names() -> list[str]:
    ensure_dirs()
    return sorted([p.name for p in DB_DIR.iterdir() if p.is_dir()])

def save_pil_image(img: Image.Image, dest_dir: Path, prefix: str) -> Path:
    img = img.convert("RGB")
    fname = f"{prefix}_{int(datetime.now().timestamp())}.jpg"
    path = dest_dir / fname
    img.save(path, "JPEG", quality=95)
    return path

def add_attendance_row(employee: str, event_type: str, image_path: Path):
    ensure_dirs()
    df = pd.read_csv(ATTENDANCE_CSV)
    ts = datetime.now().isoformat(timespec="seconds")
    df.loc[len(df)] = [ts, employee, event_type, str(image_path)]
    df.to_csv(ATTENDANCE_CSV, index=False)

def load_attendance_df() -> pd.DataFrame:
    ensure_dirs()
    return pd.read_csv(ATTENDANCE_CSV)

# ----------------------
# Simple face detector (no deep learning, just Haarcascade)
# ----------------------
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

def face_exists(pil_img: Image.Image) -> bool:
    arr = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    return len(faces) > 0

# ----------------------
# Streamlit UI
# ----------------------
ensure_dirs()
st.set_page_config(page_title="Employee Check In/Out (Face Check)", page_icon="🧑‍💼", layout="centered")
st.title("🧑‍💼 Employee Check In/Out (Face Check)")

tab_enroll, tab_check, tab_log = st.tabs(["👤 Enroll", "✅ Check In/Out", "📜 Attendance Log"])

# ---------- Enroll Tab ----------
with tab_enroll:
    st.subheader("Enroll an employee")
    name = st.text_input("Employee name", placeholder="e.g., Jane Doe")
    cam_img = st.camera_input("Capture a face photo")

    if st.button("Save to gallery", type="primary", use_container_width=True):
        if not name.strip():
            st.error("Please enter an employee name.")
        elif cam_img is None:
            st.error("Please capture a photo.")
        else:
            pil = Image.open(cam_img)
            if face_exists(pil):
                dest = employee_folder(name)
                path = save_pil_image(pil, dest, prefix=name.replace(" ", "_"))
                st.success(f"✅ Face detected, image saved at {path}")
            else:
                st.warning("❌ No face detected. Please retake the photo.")

    st.divider()
    st.caption("Currently enrolled employees:")
    employees = get_employee_names()
    st.write(", ".join(employees) if employees else "— none yet —")

# ---------- Check In/Out Tab ----------
with tab_check:
    st.subheader("Manual Check In/Out (with face existence check)")
    employees = get_employee_names()
    if not employees:
        st.warning("No employees enrolled yet. Please add someone on the Enroll tab.")
    else:
        employee = st.selectbox("Select employee", employees)
        event_type = st.radio("Event type", ["in", "out"], horizontal=True)
        cam_img = st.camera_input("Take a photo for the log")

        if st.button("Record event", type="primary", use_container_width=True):
            if cam_img is None:
                st.error("Please capture a photo.")
            else:
                pil = Image.open(cam_img)
                if face_exists(pil):
                    CHECKINS_DIR.mkdir(parents=True, exist_ok=True)
                    out_path = save_pil_image(pil, CHECKINS_DIR, prefix=f"{employee}_{event_type}")
                    add_attendance_row(employee, event_type, out_path)
                    st.success(f"✅ Recorded {event_type.upper()} for {employee}")
                    st.image(pil, caption="Saved snapshot", use_column_width=True)
                else:
                    st.warning("❌ No face detected. Not recording.")

# ---------- Log Tab ----------
with tab_log:
    st.subheader("Attendance log")
    df = load_attendance_df()
    if df.empty:
        st.info("No attendance yet.")
    else:
        df_sorted = df.sort_values("timestamp", ascending=False)
        st.dataframe(df_sorted, use_container_width=True)
        st.download_button(
            "Download CSV",
            data=df_sorted.to_csv(index=False).encode("utf-8"),
            file_name="attendance.csv",
            mime="text/csv",
            use_container_width=True,
        )

st.caption("This demo just checks if a face exists (via Haarcascade) before saving. No face recognition or tracking.")
