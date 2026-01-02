# app_yolo.py
# ------------------------------------------------------------
# YOLOv8 Learning App (Streamlit)
# ------------------------------------------------------------
# How to run:
#   pip install streamlit ultralytics opencv-python pillow numpy pandas
#   streamlit run app_yolo.py
#
# What students learn:
# - What YOLO does: object detection = boxes + labels + confidence
# - How confidence threshold changes what appears
# - How IoU threshold affects duplicate boxes (NMS)
# - How class filtering works
# - How to inspect detections as a table + counts
#
# Sources (Ultralytics docs):
# - Python usage and predict mode: https://docs.ultralytics.com/usage/python/ and /modes/predict/
# - Streamlit inference guide: https://docs.ultralytics.com/guides/streamlit-live-inference/
# ------------------------------------------------------------

import time
from typing import List, Optional, Tuple

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import cv2

from ultralytics import YOLO

# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="Learn YOLOv8 (Ultralytics) - Simple App", layout="wide")
st.title("🎯 Learn YOLOv8 by Playing (Ultralytics + Streamlit)")
st.write(
    "YOLO = **You Only Look Once**.\n"
    "It looks at an image and returns **bounding boxes + class names + confidence**.\n"
    "Use the controls to see how detection changes."
)

# ----------------------------
# Sidebar: learning controls
# ----------------------------
st.sidebar.header("⚙️ YOLO Settings")

MODEL_CHOICES = {
    "yolov8n (fastest, smallest)": "yolov8n.pt",
    "yolov8s (small)": "yolov8s.pt",
    "yolov8m (medium)": "yolov8m.pt",
    "yolov8l (large)": "yolov8l.pt",
    "yolov8x (largest, slowest)": "yolov8x.pt",
}

model_label = st.sidebar.selectbox("Choose a model size", list(MODEL_CHOICES.keys()), index=0)
weights = MODEL_CHOICES[model_label]

conf = st.sidebar.slider("Confidence threshold (0.0–1.0)", 0.0, 1.0, 0.25, 0.01)
iou = st.sidebar.slider("IoU threshold for NMS (0.0–1.0)", 0.0, 1.0, 0.45, 0.01)

st.sidebar.markdown("---")
st.sidebar.subheader("🎓 Quick meanings")
st.sidebar.write(
    "**Confidence**: how sure YOLO is.\n"
    "- Higher = fewer boxes, more strict.\n"
    "- Lower = more boxes, more mistakes.\n\n"
    "**IoU**: overlap between boxes.\n"
    "- NMS removes duplicates.\n"
    "- Higher IoU can keep more overlapping boxes."
)

# ----------------------------
# Cache model loading
# ----------------------------
@st.cache_resource
def load_model(weights_path: str) -> YOLO:
    # Loads YOLO weights once and reuses them (fast for Streamlit).
    return YOLO(weights_path)

model = load_model(weights)

# ----------------------------
# Helper: run prediction on an image (numpy RGB)
# ----------------------------
def yolo_predict_image(
    img_rgb: np.ndarray,
    conf: float,
    iou: float,
    classes: Optional[List[int]] = None,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Runs YOLO on a single RGB image and returns:
    1) annotated image (RGB) with boxes drawn
    2) dataframe of detections (class, name, conf, x1,y1,x2,y2)
    """
    # Ultralytics can accept numpy arrays directly.
    results = model.predict(
        source=img_rgb,
        conf=conf,
        iou=iou,
        classes=classes,   # None = all classes
        verbose=False,
    )

    r = results[0]  # one image => first result
    annotated_bgr = r.plot()  # plot() returns BGR image (OpenCV style)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    det_rows = []
    if r.boxes is not None and len(r.boxes) > 0:
        # r.boxes.xyxy (N,4), r.boxes.conf (N,), r.boxes.cls (N,)
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)

        names = r.names  # dict: class_id -> class_name

        for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
            det_rows.append({
                "class_id": int(k),
                "class_name": names.get(int(k), str(int(k))),
                "confidence": float(c),
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
            })

    df = pd.DataFrame(det_rows)
    return annotated_rgb, df


def class_picker_ui() -> Optional[List[int]]:
    """
    Lets user pick which classes to detect.
    Returns list of class IDs or None for "all".
    """
    names = model.names  # dict class_id -> name
    all_items = [f"{k}: {v}" for k, v in names.items()]

    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 Class filter (optional)")
    use_filter = st.sidebar.checkbox("Detect only selected classes", value=False)

    if not use_filter:
        return None

    selected = st.sidebar.multiselect(
        "Pick classes",
        options=all_items,
        default=["0: person"] if "0: person" in all_items else [],
    )

    class_ids = []
    for item in selected:
        # item looks like "0: person"
        cid = int(item.split(":")[0].strip())
        class_ids.append(cid)

    return class_ids if class_ids else None


selected_classes = class_picker_ui()

# ----------------------------
# Tabs for easy learning flow
# ----------------------------
tab_learn, tab_image, tab_camera, tab_video = st.tabs(
    ["📘 Learn YOLO", "🖼️ Image", "📷 Camera", "🎥 Video"]
)

with tab_learn:
    st.subheader("What YOLO returns (the important idea)")
    st.write(
        "YOLO turns an image into a list of detections. Each detection has:\n"
        "- **box**: (x1, y1, x2, y2)\n"
        "- **class**: what it thinks it is (person, car, dog...)\n"
        "- **confidence**: a number from 0 to 1\n"
    )

    st.subheader("How to learn fast with this app")
    st.write(
        "1) Start with **Image** tab → upload a photo.\n"
        "2) Move **confidence** slider and observe:\n"
        "   - low confidence → many boxes (including wrong ones)\n"
        "   - high confidence → fewer, cleaner boxes\n"
        "3) Turn on **class filter** in the sidebar (try only 'person').\n"
        "4) Try **Video** tab to see detection frame-by-frame.\n"
    )

    st.info(
        "Tip: Start with **yolov8n** (fast). Try bigger models only if your machine can handle it."
    )

with tab_image:
    st.subheader("🖼️ Image Upload Detection")
    up = st.file_uploader("Upload an image (jpg/png)", type=["jpg", "jpeg", "png"])
    if up:
        pil = Image.open(up).convert("RGB")
        img_rgb = np.array(pil)

        col1, col2 = st.columns(2)
        with col1:
            st.write("Original")
            st.image(pil, use_container_width=True)

        with st.spinner("Running YOLO..."):
            annotated_rgb, df = yolo_predict_image(img_rgb, conf=conf, iou=iou, classes=selected_classes)

        with col2:
            st.write("YOLO Result (boxes + labels)")
            st.image(annotated_rgb, use_container_width=True)

        st.subheader("Detections table")
        if df.empty:
            st.warning("No detections found. Try lowering confidence.")
        else:
            st.dataframe(df.sort_values("confidence", ascending=False), use_container_width=True)

            st.subheader("Counts by class")
            counts = df["class_name"].value_counts().reset_index()
            counts.columns = ["class_name", "count"]
            st.bar_chart(counts.set_index("class_name"))

with tab_camera:
    st.subheader("📷 Camera Photo Detection (easiest for students)")
    st.write("Click **Take a picture**, then YOLO will detect objects in it.")
    cam = st.camera_input("Take a picture")
    if cam:
        pil = Image.open(cam).convert("RGB")
        img_rgb = np.array(pil)

        col1, col2 = st.columns(2)
        with col1:
            st.write("Your photo")
            st.image(pil, use_container_width=True)

        with st.spinner("Running YOLO..."):
            annotated_rgb, df = yolo_predict_image(img_rgb, conf=conf, iou=iou, classes=selected_classes)

        with col2:
            st.write("YOLO Result")
            st.image(annotated_rgb, use_container_width=True)

        if not df.empty:
            st.subheader("Detections")
            st.dataframe(df.sort_values("confidence", ascending=False), use_container_width=True)

with tab_video:
    st.subheader("🎥 Video Detection (frame-by-frame)")
    st.write(
        "Upload a short video (mp4). The app runs YOLO on frames and shows an annotated preview.\n"
        "Tip: Use **yolov8n** and a smaller video for speed."
    )

    vid = st.file_uploader("Upload a video (mp4/mov/avi)", type=["mp4", "mov", "avi"])
    max_frames = st.slider("Max frames to process (demo)", 30, 600, 120, 10)
    every_n = st.slider("Process every Nth frame (speed trick)", 1, 10, 2, 1)

    if vid:
        # Save uploaded video to a temp file
        tmp_path = "temp_video.mp4"
        with open(tmp_path, "wb") as f:
            f.write(vid.read())

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            st.error("Could not open video.")
        else:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            st.write(f"Video frames: **{frame_count}** | FPS: **{fps:.2f}**")

            run = st.button("▶️ Run YOLO on video")
            if run:
                preview = st.empty()
                prog = st.progress(0)
                info = st.empty()

                processed = 0
                shown = 0
                t0 = time.time()

                # simple counts accumulator
                total_counts = {}

                while processed < max_frames:
                    ok, frame_bgr = cap.read()
                    if not ok:
                        break

                    # skip frames to go faster
                    frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    if frame_index % every_n != 0:
                        continue

                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                    annotated_rgb, df = yolo_predict_image(
                        frame_rgb, conf=conf, iou=iou, classes=selected_classes
                    )

                    # update running counts
                    if not df.empty:
                        for name, cnt in df["class_name"].value_counts().to_dict().items():
                            total_counts[name] = total_counts.get(name, 0) + int(cnt)

                    preview.image(annotated_rgb, caption=f"Frame {frame_index}", use_container_width=True)

                    processed += 1
                    shown += 1
                    prog.progress(min(1.0, processed / max_frames))

                    elapsed = time.time() - t0
                    info.write(f"Processed frames: **{processed}/{max_frames}** | Elapsed: **{elapsed:.1f}s**")

                cap.release()

                st.success("Done!")
                if total_counts:
                    st.subheader("Total detected counts (across processed frames)")
                    cdf = pd.DataFrame(
                        [{"class_name": k, "count": v} for k, v in sorted(total_counts.items(), key=lambda x: -x[1])]
                    )
                    st.dataframe(cdf, use_container_width=True)
                    st.bar_chart(cdf.set_index("class_name"))

# Footer tips
st.markdown("---")
st.caption(
    "Learning tip: If the model detects too many wrong boxes, increase **confidence**. "
    "If you see duplicates, try adjusting **IoU**."
)
