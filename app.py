# app_opencv.py
# Streamlit + OpenCV demo app:
# ✅ Upload image
# ✅ Resize
# ✅ Rotate
# ✅ Annotate (text + rectangle)
# ✅ Canny edge detection
# ✅ Face detection (Haar Cascade)

import streamlit as st
import numpy as np
import cv2

st.set_page_config(page_title="OpenCV Playground", layout="wide")
st.title("🖼️ OpenCV Playground (Resize • Rotate • Annotate • Canny • Face Detect)")

st.write(
    "Upload an image and try classic OpenCV operations.\n\n"
    "**Tip:** Face detection works best on clear front-facing faces."
)

# -------------------------
# Helpers
# -------------------------
def read_image_as_bgr(uploaded_file):
    """Read uploaded image bytes -> OpenCV BGR image."""
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return bgr

def bgr_to_rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def clamp_int(x, lo, hi):
    return int(max(lo, min(hi, x)))

# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.header("1) Upload")
uploaded = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "bmp", "webp"])

st.sidebar.markdown("---")
st.sidebar.header("2) Resize")
resize_on = st.sidebar.checkbox("Enable resize", value=True)
resize_mode = st.sidebar.selectbox("Resize mode", ["Scale (%)", "Exact (W,H)"], index=0)

scale_percent = st.sidebar.slider("Scale percent", 10, 300, 100, 5)

target_w = st.sidebar.slider("Target width", 50, 2000, 640, 10)
target_h = st.sidebar.slider("Target height", 50, 2000, 480, 10)

st.sidebar.markdown("---")
st.sidebar.header("3) Rotate")
rotate_on = st.sidebar.checkbox("Enable rotate", value=False)
angle = st.sidebar.slider("Angle (degrees)", -180, 180, 0, 1)

st.sidebar.markdown("---")
st.sidebar.header("4) Annotate")
annotate_on = st.sidebar.checkbox("Enable annotation", value=False)
text = st.sidebar.text_input("Text to draw", "Hello OpenCV!")
text_scale = st.sidebar.slider("Text size", 0.3, 3.0, 1.0, 0.1)
text_thickness = st.sidebar.slider("Text thickness", 1, 10, 2, 1)
text_x = st.sidebar.slider("Text X", 0, 2000, 30, 5)
text_y = st.sidebar.slider("Text Y", 0, 2000, 60, 5)

rect_on = st.sidebar.checkbox("Draw rectangle box", value=False)
rect_x1 = st.sidebar.slider("Rect x1", 0, 2000, 50, 5)
rect_y1 = st.sidebar.slider("Rect y1", 0, 2000, 50, 5)
rect_x2 = st.sidebar.slider("Rect x2", 0, 2000, 250, 5)
rect_y2 = st.sidebar.slider("Rect y2", 0, 2000, 250, 5)
rect_thickness = st.sidebar.slider("Rect thickness", 1, 15, 3, 1)

st.sidebar.markdown("---")
st.sidebar.header("5) Canny Edge Detection")
canny_on = st.sidebar.checkbox("Enable Canny", value=False)
canny_t1 = st.sidebar.slider("Threshold 1", 0, 500, 100, 5)
canny_t2 = st.sidebar.slider("Threshold 2", 0, 500, 200, 5)
canny_blur = st.sidebar.slider("Blur kernel (odd number)", 1, 21, 5, 2)

st.sidebar.markdown("---")
st.sidebar.header("6) Face Detection")
face_on = st.sidebar.checkbox("Enable face detection", value=False)
face_scaleFactor = st.sidebar.slider("scaleFactor (smaller = more detections, slower)", 1.01, 1.50, 1.10, 0.01)
face_minNeighbors = st.sidebar.slider("minNeighbors (bigger = stricter)", 1, 15, 5, 1)
min_face_size = st.sidebar.slider("min face size (pixels)", 20, 300, 40, 5)

# -------------------------
# Main
# -------------------------
if uploaded is None:
    st.info("👈 Upload an image from the sidebar to start.")
    st.stop()

# Read original
orig_bgr = read_image_as_bgr(uploaded)
if orig_bgr is None:
    st.error("Could not read the image. Try another file.")
    st.stop()

img_bgr = orig_bgr.copy()

# -------------------------
# Resize
# -------------------------
if resize_on:
    h, w = img_bgr.shape[:2]
    if resize_mode == "Scale (%)":
        new_w = int(w * scale_percent / 100)
        new_h = int(h * scale_percent / 100)
    else:
        new_w, new_h = int(target_w), int(target_h)

    # Guard
    new_w = max(1, new_w)
    new_h = max(1, new_h)

    img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

# -------------------------
# Rotate
# -------------------------
if rotate_on and angle != 0:
    h, w = img_bgr.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    img_bgr = cv2.warpAffine(img_bgr, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

# -------------------------
# Annotate (text + rectangle)
# -------------------------
if annotate_on:
    # Clamp coordinates into image
    h, w = img_bgr.shape[:2]
    tx = clamp_int(text_x, 0, w - 1)
    ty = clamp_int(text_y, 0, h - 1)

    # Text in green
    cv2.putText(
        img_bgr,
        text,
        (tx, ty),
        cv2.FONT_HERSHEY_SIMPLEX,
        float(text_scale),
        (0, 255, 0),
        int(text_thickness),
        lineType=cv2.LINE_AA
    )

    if rect_on:
        x1 = clamp_int(min(rect_x1, rect_x2), 0, w - 1)
        y1 = clamp_int(min(rect_y1, rect_y2), 0, h - 1)
        x2 = clamp_int(max(rect_x1, rect_x2), 0, w - 1)
        y2 = clamp_int(max(rect_y1, rect_y2), 0, h - 1)

        # Rectangle in red
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 0, 255), int(rect_thickness))

# -------------------------
# Face detection
# -------------------------
faces = []
if face_on:
    # Haar cascade file comes with OpenCV-python
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=float(face_scaleFactor),
        minNeighbors=int(face_minNeighbors),
        minSize=(int(min_face_size), int(min_face_size))
    )

    # Draw face boxes in blue
    for (x, y, w, h) in faces:
        cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (255, 0, 0), 2)

# -------------------------
# Canny (edges)
# -------------------------
edges = None
if canny_on:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    k = int(canny_blur)
    if k % 2 == 0:
        k += 1  # must be odd
    if k > 1:
        gray = cv2.GaussianBlur(gray, (k, k), 0)

    edges = cv2.Canny(gray, int(canny_t1), int(canny_t2))

# -------------------------
# Display results
# -------------------------
st.subheader("Results")

colA, colB = st.columns(2)

with colA:
    st.write("✅ Processed Image")
    st.image(bgr_to_rgb(img_bgr), use_container_width=True)

    if face_on:
        st.write(f"👤 Faces detected: **{len(faces)}**")

with colB:
    st.write("🧩 Original Image")
    st.image(bgr_to_rgb(orig_bgr), use_container_width=True)

    if edges is not None:
        st.write("🟦 Canny Edges")
        st.image(edges, clamp=True, use_container_width=True)

st.caption(
    "Install requirements: `pip install streamlit opencv-python numpy`  |  Run: `streamlit run app_opencv.py`"
)
