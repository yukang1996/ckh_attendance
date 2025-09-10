# # app.py
# # Streamlit-only UI (no face recognition). Handles:
# # - Enroll employees (save reference images)
# # - Manual check-in/out (select employee + capture photo)
# # - View/download attendance log

# import os
# from pathlib import Path
# from datetime import datetime

# import streamlit as st
# import pandas as pd
# from PIL import Image

# # ----------------------
# # Config / Paths
# # ----------------------
# DATA_DIR = Path("data")
# DB_DIR = DATA_DIR / "employees_db"     # where enrollment photos are stored
# CHECKINS_DIR = DATA_DIR / "checkins"   # where check-in/out snapshots are stored
# ATTENDANCE_CSV = DATA_DIR / "attendance.csv"

# # ----------------------
# # Setup
# # ----------------------
# def ensure_dirs():
#     DB_DIR.mkdir(parents=True, exist_ok=True)
#     CHECKINS_DIR.mkdir(parents=True, exist_ok=True)
#     ATTENDANCE_CSV.parent.mkdir(parents=True, exist_ok=True)
#     if not ATTENDANCE_CSV.exists():
#         pd.DataFrame(
#             columns=["timestamp", "employee", "event_type", "image_path"]
#         ).to_csv(ATTENDANCE_CSV, index=False)

# def employee_folder(name: str) -> Path:
#     safe = name.strip().replace("/", "_")
#     p = DB_DIR / safe
#     p.mkdir(parents=True, exist_ok=True)
#     return p

# def get_employee_names() -> list[str]:
#     ensure_dirs()
#     return sorted([p.name for p in DB_DIR.iterdir() if p.is_dir()])

# def save_uploaded_image(uploaded_file, dest_dir: Path, prefix: str) -> Path:
#     img = Image.open(uploaded_file).convert("RGB")
#     fname = f"{prefix}_{int(datetime.now().timestamp())}.jpg"
#     path = dest_dir / fname
#     img.save(path, "JPEG", quality=95)
#     return path

# def save_pil_image(img: Image.Image, dest_dir: Path, prefix: str) -> Path:
#     img = img.convert("RGB")
#     fname = f"{prefix}_{int(datetime.now().timestamp())}.jpg"
#     path = dest_dir / fname
#     img.save(path, "JPEG", quality=95)
#     return path

# def add_attendance_row(employee: str, event_type: str, image_path: Path):
#     ensure_dirs()
#     df = pd.read_csv(ATTENDANCE_CSV)
#     ts = datetime.now().isoformat(timespec="seconds")
#     df.loc[len(df)] = [ts, employee, event_type, str(image_path)]
#     df.to_csv(ATTENDANCE_CSV, index=False)

# def load_attendance_df() -> pd.DataFrame:
#     ensure_dirs()
#     return pd.read_csv(ATTENDANCE_CSV)

# # ----------------------
# # Streamlit UI
# # ----------------------
# ensure_dirs()
# st.set_page_config(page_title="Employee Check In/Out (No Face Recognition)", page_icon="🧑‍💼", layout="centered")
# st.title("🧑‍💼 Employee Check In/Out (No Face Recognition)")

# tab_enroll, tab_check, tab_log = st.tabs(["👤 Enroll", "✅ Check In/Out", "📜 Attendance Log"])

# # ---------- Enroll Tab ----------
# with tab_enroll:
#     st.subheader("Enroll an employee")
#     st.caption("Save 1–3 clear, frontal photos per employee. These are just stored for reference.")

#     name = st.text_input("Employee name", placeholder="e.g., Jane Doe")
#     files = st.file_uploader("Upload 1–3 images (JPG/PNG)", type=["jpg","jpeg","png"], accept_multiple_files=True)

#     use_cam = st.checkbox("Use camera instead of files")
#     cam_img = st.camera_input("Capture a photo") if use_cam else None

#     if st.button("Save to gallery", type="primary", use_container_width=True):
#         if not name.strip():
#             st.error("Please enter an employee name.")
#         else:
#             dest = employee_folder(name)
#             saved = False
#             for f in files or []:
#                 save_uploaded_image(f, dest, prefix=name.replace(" ", "_"))
#                 saved = True
#             if cam_img is not None:
#                 save_pil_image(Image.open(cam_img), dest, prefix=name.replace(" ", "_"))
#                 saved = True

#             if saved:
#                 st.success(f"Saved enrollment images to `{dest}`")
#             else:
#                 st.warning("No images provided.")

#     # Show existing employees
#     st.divider()
#     st.caption("Currently enrolled employees:")
#     employees = get_employee_names()
#     if employees:
#         st.write(", ".join(employees))
#     else:
#         st.write("— none yet —")

# # ---------- Check In/Out Tab ----------
# with tab_check:
#     st.subheader("Manual Check In/Out (no face recognition)")
#     employees = get_employee_names()
#     if not employees:
#         st.warning("No employees enrolled yet. Please add someone on the Enroll tab.")
#     else:
#         employee = st.selectbox("Select employee", employees)
#         event_type = st.radio("Event type", ["in", "out"], horizontal=True)
#         st.caption("Capture a photo to save with the log entry.")
#         cam_img = st.camera_input("Take a photo for the log (optional but recommended)")

#         if st.button("Record event", type="primary", use_container_width=True):
#             # Save snapshot (optional)
#             if cam_img is not None:
#                 CHECKINS_DIR.mkdir(parents=True, exist_ok=True)
#                 fname = f"{employee}_{event_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
#                 out_path = CHECKINS_DIR / fname
#                 Image.open(cam_img).convert("RGB").save(out_path, "JPEG", quality=95)
#             else:
#                 out_path = CHECKINS_DIR / f"{employee}_{event_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_noimg.txt"
#                 out_path.write_text("No image captured.", encoding="utf-8")

#             add_attendance_row(employee, event_type, out_path)
#             st.success(f"Recorded **{event_type.upper()}** for **{employee}** at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
#             if out_path.suffix.lower() == ".jpg":
#                 st.image(str(out_path), caption="Saved snapshot", use_column_width=True)

# # ---------- Log Tab ----------
# with tab_log:
#     st.subheader("Attendance log")
#     df = load_attendance_df()
#     if df.empty:
#         st.info("No attendance yet.")
#     else:
#         df_sorted = df.sort_values("timestamp", ascending=False)
#         st.dataframe(df_sorted, use_container_width=True)
#         st.download_button(
#             "Download CSV",
#             data=df_sorted.to_csv(index=False).encode("utf-8"),
#             file_name="attendance.csv",
#             mime="text/csv",
#             use_container_width=True,
#         )

# st.caption("This demo intentionally contains no face recognition logic—just Streamlit UI and simple file/CSV storage.")


import os
import io
import time
from datetime import datetime

import streamlit as st
import pandas as pd
from PIL import Image
from deepface import DeepFace

# ----------------------
# Config
# ----------------------
DB_DIR = "employees_db"          # Face "database" (just a folder of images)
ATTENDANCE_CSV = "attendance.csv"
MODEL_NAME = "ArcFace"           # Good balance of accuracy/speed
DETECTOR = "retinaface"          # More robust detection; fallback to "opencv" if needed
SIMILARITY_THRESHOLD = 0.35      # Lower is stricter for ArcFace (distance). Tune as needed.

os.makedirs(DB_DIR, exist_ok=True)

# Ensure attendance CSV exists
if not os.path.exists(ATTENDANCE_CSV):
    pd.DataFrame(columns=["timestamp", "employee", "match_distance", "image_path"]).to_csv(
        ATTENDANCE_CSV, index=False
    )

# ----------------------
# Small helpers
# ----------------------
def save_uploaded_image(uploaded_file, dest_path):
    img = Image.open(uploaded_file).convert("RGB")
    img.save(dest_path, format="JPEG", quality=95)
    return dest_path

def save_pil_image(pil_img, dest_path):
    pil_img.convert("RGB").save(dest_path, format="JPEG", quality=95)
    return dest_path

def add_attendance_row(employee_name, distance, image_path):
    ts = datetime.now().isoformat(timespec="seconds")
    df = pd.read_csv(ATTENDANCE_CSV)
    df.loc[len(df)] = [ts, employee_name, float(distance), image_path]
    df.to_csv(ATTENDANCE_CSV, index=False)

def run_face_search(img_path):
    """
    Use DeepFace.find to search img_path face in DB_DIR.
    Returns (best_name, best_distance, best_identity_path) or (None, None, None).
    """
    try:
        # DeepFace.find returns a list of dataframes (one per detected face). We take the first.
        results = DeepFace.find(
            img_path=img_path,
            db_path=DB_DIR,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR,
            enforce_detection=True,   # raise if no face
            silent=True
        )
        if isinstance(results, list) and len(results) > 0 and not results[0].empty:
            df = results[0].sort_values(by="distance", ascending=True).reset_index(drop=True)
            best = df.iloc[0]
            identity_path = best["identity"]
            distance = float(best["distance"])

            # Infer employee name from folder or filename
            # Expected layout: employees_db/<employee_name>/<image>.jpg OR a flat file with name in filename
            name = os.path.basename(os.path.dirname(identity_path))
            if not name or name == DB_DIR:
                # Fallback: filename before extension
                name = os.path.splitext(os.path.basename(identity_path))[0]

            return name, distance, identity_path
        return None, None, None
    except Exception as e:
        st.error(f"Recognition error: {e}")
        return None, None, None

def enroll_example_note():
    st.info(
        "💡 Tip: Create one subfolder per employee in `employees_db/` "
        "and add 1–3 clear, front-facing images (good lighting, no sunglasses). "
        "You can also use the uploader below to do this."
    )

# ----------------------
# UI
# ----------------------
st.set_page_config(page_title="Face Check-In (DeepFace)", page_icon="🧑‍💼", layout="centered")
st.title("🧑‍💼 Face Recognition Check-In (DeepFace)")

tab_enroll, tab_checkin, tab_log = st.tabs(["👤 Enroll", "✅ Check In", "📜 Attendance Log"])

# -------- Enroll Tab --------
with tab_enroll:
    st.subheader("Enroll an employee")
    enroll_example_note()

    employee_name = st.text_input("Employee name", placeholder="e.g., Jane Doe")
    uploaded_imgs = st.file_uploader(
        "Upload 1–3 face images (JPG/PNG).",
        accept_multiple_files=True,
        type=["jpg", "jpeg", "png"],
    )
    col1, col2 = st.columns(2)
    with col1:
        do_capture = st.checkbox("Use camera instead of files")

    captured_img = None
    if do_capture:
        cam = st.camera_input("Capture a face image")
        if cam is not None:
            captured_img = Image.open(cam)

    enroll_btn = st.button("Save to database", type="primary", use_container_width=True)

    if enroll_btn:
        if not employee_name.strip():
            st.error("Please enter an employee name.")
        else:
            emp_folder = os.path.join(DB_DIR, employee_name.strip().replace("/", "_"))
            os.makedirs(emp_folder, exist_ok=True)
            saved_any = False

            # Save uploaded files
            if uploaded_imgs:
                for i, uf in enumerate(uploaded_imgs, start=1):
                    dest = os.path.join(emp_folder, f"{employee_name}_{int(time.time())}_{i}.jpg")
                    save_uploaded_image(uf, dest)
                    saved_any = True

            # Save captured image
            if captured_img is not None:
                dest = os.path.join(emp_folder, f"{employee_name}_{int(time.time())}_cam.jpg")
                save_pil_image(captured_img, dest)
                saved_any = True

            if saved_any:
                st.success(f"Enrolled images saved under `{emp_folder}`.")
                st.caption("DeepFace will (re)index this folder automatically on first use.")
            else:
                st.warning("No images provided. Upload files or capture from camera.")

# -------- Check-In Tab --------
with tab_checkin:
    st.subheader("Check in via webcam")
    st.caption("Position your face in the frame, ensure good lighting, and capture a photo.")

    cam_img = st.camera_input("Take a photo for check-in")
    model_col, det_col, thr_col = st.columns(3)
    with model_col:
        st.selectbox("Model", ["ArcFace", "Facenet", "VGG-Face", "SFace"], index=["ArcFace","Facenet","VGG-Face","SFace"].index(MODEL_NAME), disabled=True)
    with det_col:
        st.selectbox("Detector", ["retinaface", "mtcnn", "opencv"], index=["retinaface","mtcnn","opencv"].index(DETECTOR), disabled=True)
    with thr_col:
        st.number_input("Distance threshold", min_value=0.1, max_value=1.0, value=float(SIMILARITY_THRESHOLD), step=0.01, help="Lower = stricter match", disabled=True)

    check_btn = st.button("🔍 Match & Check In", type="primary", use_container_width=True)

    if check_btn:
        if cam_img is None:
            st.error("Please capture an image first.")
        else:
            # Save the captured image temporarily
            tmp_path = os.path.join("tmp_capture.jpg")
            Image.open(cam_img).convert("RGB").save(tmp_path, "JPEG", quality=95)

            with st.spinner("Matching face…"):
                name, distance, ref_path = run_face_search(tmp_path)

            if name is None:
                st.error("No matching face found or no face detected. Try again with better lighting or angle.")
            else:
                st.write(f"**Best match**: {name}")
                st.write(f"**Match distance**: {distance:.4f} (threshold ≤ {SIMILARITY_THRESHOLD})")

                if distance <= SIMILARITY_THRESHOLD:
                    # Save a copy of the captured image for audit log
                    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    checkin_img_path = os.path.join("checkins", f"{name}_{stamp}.jpg")
                    os.makedirs("checkins", exist_ok=True)
                    Image.open(cam_img).convert("RGB").save(checkin_img_path, "JPEG", quality=95)

                    add_attendance_row(name, distance, checkin_img_path)
                    st.success(f"✅ Check-in recorded for **{name}** at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    st.caption(f"Reference: {ref_path}")
                    st.image(Image.open(checkin_img_path), caption="Saved check-in image", use_column_width=True)
                else:
                    st.warning("Face found, but distance is above threshold. Not recording check-in.")

# -------- Log Tab --------
with tab_log:
    st.subheader("Attendance log")
    if os.path.exists(ATTENDANCE_CSV):
        log_df = pd.read_csv(ATTENDANCE_CSV)
        st.dataframe(log_df.sort_values("timestamp", ascending=False), use_container_width=True)
        st.download_button(
            "Download CSV",
            data=log_df.to_csv(index=False).encode("utf-8"),
            file_name="attendance.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.info("No attendance yet.")

st.caption("Powered by DeepFace. Images are processed locally for this demo.")
