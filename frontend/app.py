import os
import time
import base64

import cv2
import numpy as np
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="PPE Detection", page_icon="🦺", layout="wide")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@st.cache_data(ttl=30)
def get_classes():
    try:
        resp = requests.get(f"{BACKEND_URL}/classes", timeout=5)
        resp.raise_for_status()
        return resp.json()["classes"]
    except Exception:
        return []


def b64_to_bgr(b64_string: str) -> np.ndarray:
    img_bytes = base64.b64decode(b64_string)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def frame_to_b64(frame_bgr: np.ndarray) -> str:
    _, buffer = cv2.imencode(".jpg", frame_bgr)
    return base64.b64encode(buffer).decode("utf-8")


# ---------------------------------------------------------------------------
# Sidebar — shared controls
# ---------------------------------------------------------------------------
st.sidebar.title("🦺 PPE Detection")
st.sidebar.caption(f"Backend: {BACKEND_URL}")

try:
    health = requests.get(f"{BACKEND_URL}/health", timeout=3).json()
    st.sidebar.success("Backend connected ✅")
except Exception:
    st.sidebar.error("Backend not reachable ❌\nMake sure FastAPI is running.")
    health = None

conf_threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.4, 0.05)

available_classes = get_classes()
selected_classes = st.sidebar.multiselect(
    "Classes to show (empty = all)",
    options=available_classes,
    default=[],
)
class_filter = selected_classes if selected_classes else None
class_filter_param = ",".join(class_filter) if class_filter else None

st.sidebar.markdown("---")
st.sidebar.caption("Model classes: " + (", ".join(available_classes) if available_classes else "unavailable"))


# ---------------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------------
st.title("Real-Time PPE Detection")
st.caption("Upload an image or video, or use your webcam, to detect helmets, vests, gloves, masks, and boots.")

tab_image, tab_video, tab_webcam = st.tabs(["📷 Image", "🎬 Video", "🔴 Live Webcam"])


# ---- Tab 1: Image ----------------------------------------------------------
with tab_image:
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="image_uploader")

    if uploaded_image is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded_image, caption="Original", use_container_width=True)

        if st.button("Run Detection", key="run_image"):
            with st.spinner("Running inference..."):
                files = {"file": (uploaded_image.name, uploaded_image.getvalue(), uploaded_image.type)}
                params = {"conf_threshold": conf_threshold}
                if class_filter_param:
                    params["class_filter"] = class_filter_param

                try:
                    resp = requests.post(f"{BACKEND_URL}/predict/image", files=files, params=params, timeout=30)
                    resp.raise_for_status()
                    result = resp.json()

                    annotated = b64_to_bgr(result["annotated_image_base64"])
                    with col2:
                        st.image(bgr_to_rgb(annotated), caption="Detections", use_container_width=True)

                    st.success(f"Inference time: {result['inference_ms']} ms — {len(result['detections'])} objects found")
                    if result["detections"]:
                        st.dataframe(result["detections"], use_container_width=True)
                except Exception as e:
                    st.error(f"Prediction failed: {e}")


# ---- Tab 2: Video -----------------------------------------------------------
with tab_video:
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"], key="video_uploader")

    if uploaded_video is not None:
        st.video(uploaded_video)

        if st.button("Process Video", key="run_video"):
            with st.spinner("Processing video frame-by-frame — this can take a while..."):
                files = {"file": (uploaded_video.name, uploaded_video.getvalue(), uploaded_video.type)}
                params = {"conf_threshold": conf_threshold}
                if class_filter_param:
                    params["class_filter"] = class_filter_param

                try:
                    resp = requests.post(f"{BACKEND_URL}/predict/video", files=files, params=params, timeout=600)
                    resp.raise_for_status()

                    st.success("Done! Preview and download below.")
                    st.video(resp.content)
                    st.download_button(
                        "⬇️ Download annotated video",
                        data=resp.content,
                        file_name=f"annotated_{uploaded_video.name}",
                        mime="video/mp4",
                    )
                except Exception as e:
                    st.error(f"Video processing failed: {e}")


# ---- Tab 3: Live Webcam -----------------------------------------------------
with tab_webcam:
    st.caption("Uses your machine's webcam. Each frame is sent to the backend for detection.")

    if "webcam_running" not in st.session_state:
        st.session_state.webcam_running = False

    col_start, col_stop = st.columns(2)
    if col_start.button("▶️ Start Webcam"):
        st.session_state.webcam_running = True
    if col_stop.button("⏹️ Stop Webcam"):
        st.session_state.webcam_running = False

    frame_placeholder = st.empty()
    fps_placeholder = st.empty()

    if st.session_state.webcam_running:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not access the webcam. Check camera permissions.")
        else:
            while st.session_state.webcam_running:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to read from webcam.")
                    break

                t0 = time.time()
                payload = {
                    "image_base64": frame_to_b64(frame),
                    "conf_threshold": conf_threshold,
                    "class_filter": class_filter,
                }
                try:
                    resp = requests.post(f"{BACKEND_URL}/predict/frame", json=payload, timeout=5)
                    resp.raise_for_status()
                    result = resp.json()
                    annotated = b64_to_bgr(result["annotated_frame_base64"])
                    frame_placeholder.image(bgr_to_rgb(annotated), channels="RGB", use_container_width=True)
                    fps = 1.0 / max(time.time() - t0, 1e-6)
                    fps_placeholder.caption(f"~{fps:.1f} FPS | inference: {result['inference_ms']} ms")
                except Exception as e:
                    st.error(f"Frame prediction failed: {e}")
                    break

            cap.release()
    else:
        st.info("Click 'Start Webcam' to begin live detection.")
