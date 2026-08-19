import os
import time
import threading
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

try:
    import winsound  # Windows built-in alarm sound support
    WINDOWS_SOUND_AVAILABLE = True
except ImportError:
    winsound = None
    WINDOWS_SOUND_AVAILABLE = False

st.set_page_config(
    page_title="Bus Driver Fatigue Detection Dashboard",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Keep the dashboard title safely below Streamlit's top toolbar. */
    .block-container {
        padding-top: 3.6rem !important;
        padding-bottom: 2rem !important;
    }

    .dashboard-title {
        font-size: clamp(2rem, 3vw, 2.65rem);
        font-weight: 700;
        line-height: 1.25 !important;
        margin: 0 !important;
        padding: 0.35rem 0 0.15rem 0 !important;
        overflow: visible !important;
        white-space: normal !important;
    }

    .dashboard-subtitle {
        font-size: 1.05rem;
        line-height: 1.5;
        opacity: .75;
        margin-top: 0.1rem;
        margin-bottom: 1.2rem;
    }

    .note {
        padding: .85rem 1rem;
        border: 1px solid rgba(128,128,128,.3);
        border-radius: .7rem;
        margin: .5rem 0 1rem 0;
    }

    @media (max-width: 900px) {
        .block-container {
            padding-top: 3.9rem !important;
        }
        .dashboard-title {
            font-size: 2rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

GITHUB_BASE = "https://github.com/mawei-research/bus-driver-fatigue-detection"

# ============================================================
# VERIFIED THESIS RESULTS
# ============================================================
image_models = pd.DataFrame([
    ["Eye-State 2D-CNN", 12735, 99.05, 99.05, 99.05, 99.05, 99.95],
    ["Mouth-State 2D-CNN", 768, 97.66, 97.65, 97.66, 97.66, 99.61],
], columns=["Model", "Test Images", "Accuracy (%)", "Macro Precision (%)",
            "Macro Recall (%)", "Macro F1 (%)", "ROC-AUC (%)"])

traditional_ml = pd.DataFrame([
    ["Frame Level", "6,429 feature rows", 63.46, 54.56, 55.81, 54.32, 57.47],
    ["Video Level", "51 test videos", 60.78, 53.44, 52.94, 52.78, 57.79],
], columns=["Evaluation Level", "Test Samples", "Accuracy (%)", "Macro Precision (%)",
            "Macro Recall (%)", "Macro F1 (%)", "ROC-AUC (%)"])

same_dataset = pd.DataFrame([
    ["Fixed Threshold Baseline", 319, 54.86, 36.28, 34.63],
    ["Final Dual-CNN System", 319, 81.50, 65.50, 0.49],
], columns=["Method", "Evaluation Videos", "Accuracy (%)", "Fatigue F1 (%)",
            "False Alarm Rate (%)"])

geometry = pd.DataFrame([
    ["Without Geometric Filter", 90.28, 88.79, 83.33, 94.15, 85.97, 89.27, 5.85, 16.67, 193, 12, 19, 95],
    ["With Geometric Filter", 81.50, 98.25, 49.12, 99.51, 65.50, 76.43, 0.49, 50.88, 204, 1, 58, 56],
], columns=["Setting", "Accuracy (%)", "Fatigue Precision (%)", "Fatigue Recall (%)",
            "Specificity (%)", "Fatigue F1 (%)", "Macro F1 (%)", "False Alarm Rate (%)",
            "Miss Rate (%)", "TN", "FP", "FN", "TP"])

framestep = pd.DataFrame([
    [1, 81.82, 98.28, 50.00, 99.51, 66.28, 76.92, 0.49, 50.00, 17.3743, 17.3743, 59.0337, 85.3739, 0.5797, 7.9881, 7.4074, 2.002, 0.002],
    [2, 81.50, 98.25, 49.12, 99.51, 65.50, 76.43, 0.49, 50.88, 16.5059, 32.9883, 65.7056, 91.4158, 1.1007, 7.6291, 7.3740, 2.002, 0.002],
    [3, 80.88, 98.18, 47.37, 99.51, 63.91, 75.45, 0.49, 52.63, 17.7544, 53.1830, 55.0391, 80.3549, 1.7745, 7.5835, 7.1071, 2.002, 0.002],
    [4, 81.50, 98.25, 49.12, 99.51, 65.50, 76.43, 0.49, 50.88, 17.6363, 70.3892, 54.1543, 80.3268, 2.3487, 7.8340, 7.4074, 2.002, 0.002],
], columns=["FrameStep", "Accuracy (%)", "Fatigue Precision (%)", "Fatigue Recall (%)",
            "Specificity (%)", "Fatigue F1 (%)", "Macro F1 (%)", "FAR (%)", "Miss Rate (%)",
            "Processing FPS", "Effective Source FPS", "Mean Latency (ms)", "Mean P95 Latency (ms)",
            "Real-Time Factor", "Mean First Alarm Time (s)", "Median First Alarm Time (s)",
            "Mean Trigger Delay (s)", "Trigger MAE (s)"])

final_cm = pd.DataFrame([[204, 1], [58, 56]],
                        index=["Actual Normal", "Actual Fatigue"],
                        columns=["Predicted Normal", "Predicted Fatigue"])
eye_cm = pd.DataFrame([[6218, 74], [47, 6396]],
                      index=["Actual Closed", "Actual Open"],
                      columns=["Predicted Closed", "Predicted Open"])
mouth_cm = pd.DataFrame([[378, 11], [7, 372]],
                        index=["Actual Normal", "Actual Yawn"],
                        columns=["Predicted Normal", "Predicted Yawn"])
rf_cm = pd.DataFrame([[26, 8], [12, 5]],
                     index=["Actual Normal", "Actual Fatigue"],
                     columns=["Predicted Normal", "Predicted Fatigue"])

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Dashboard Section",
    [
        "Overview",
        "Real-Time Monitoring",
        "Model Performance",
        "Video-Level Evaluation",
        "Geometric Filtering",
        "FrameStep & Runtime",
        "Research Questions",
        "Code & Resources",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Thesis**")
st.sidebar.caption("Machine Learning-Based Pre-Departure Fatigue Detection for Bus Drivers")
st.sidebar.markdown(f"[Open GitHub Repository]({GITHUB_BASE})")

st.markdown('<div class="dashboard-title">Bus Driver Fatigue Detection Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="dashboard-subtitle">Experimental results and real-time prototype monitoring.</div>', unsafe_allow_html=True)

# ============================================================
# REAL-TIME MONITORING HELPERS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
EYE_MODEL_PATH = BASE_DIR / "models" / "best_eye_2dcnn.h5"
MOUTH_MODEL_PATH = BASE_DIR / "models" / "best_mouth_2dcnn.h5"

IMAGE_SIZE = 80
EYE_THRESHOLD = 0.50
MOUTH_THRESHOLD = 0.50
SMILE_RATIO_THRESHOLD = 1.50
ALARM_SECONDS = 2.0
ALARM_SOUND_REPEAT_SECONDS = 3.0
EYE_EXPANSION = 0.25
MOUTH_EXPANSION = 0.25

# Real-time optimisation settings.
FRAME_STEP = 2
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480
CAMERA_IDEAL_FPS = 24

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308,
         324, 318, 402, 317, 14, 87, 178, 88, 95, 78, 191, 80, 81, 82,
         13, 312, 311, 310, 415]
MOUTH_LEFT, MOUTH_RIGHT, MOUTH_TOP, MOUTH_BOTTOM = 61, 291, 0, 17



def get_twilio_ice_servers():
    """
    Request short-lived Twilio STUN/TURN credentials and return
    (ice_servers, status_message). The status message never exposes secrets.
    """
    account_sid = None
    auth_token = None

    try:
        # Check the two required Streamlit secrets explicitly.
        if "TWILIO_ACCOUNT_SID" not in st.secrets:
            return None, "Missing Streamlit secret: TWILIO_ACCOUNT_SID"
        if "TWILIO_AUTH_TOKEN" not in st.secrets:
            return None, "Missing Streamlit secret: TWILIO_AUTH_TOKEN"

        account_sid = str(st.secrets["TWILIO_ACCOUNT_SID"]).strip()
        auth_token = str(st.secrets["TWILIO_AUTH_TOKEN"]).strip()

        if not account_sid:
            return None, "TWILIO_ACCOUNT_SID is empty"
        if not auth_token:
            return None, "TWILIO_AUTH_TOKEN is empty"
        if not account_sid.startswith("AC"):
            return None, "TWILIO_ACCOUNT_SID does not start with AC"

        from twilio.rest import Client

        client = Client(account_sid, auth_token)

        # Twilio NTS token. 1 hour is sufficient for the dashboard session.
        token = client.tokens.create(ttl=3600)

        ice_servers = token.ice_servers
        if not ice_servers:
            return None, "Twilio returned no ICE servers"

        return ice_servers, "Twilio STUN/TURN token created successfully"

    except ImportError:
        return None, "Python package 'twilio' is not installed on Streamlit Cloud"

    except Exception as exc:
        # Sanitize possible credentials before showing the error.
        message = f"{type(exc).__name__}: {exc}"
        if account_sid:
            message = message.replace(account_sid, "[ACCOUNT_SID]")
        if auth_token:
            message = message.replace(auth_token, "[AUTH_TOKEN]")
        return None, message[:500]


def build_rtc_configuration():
    """Prefer Twilio TURN on cloud; fall back to Google STUN."""
    twilio_servers, _ = get_twilio_ice_servers()

    if twilio_servers:
        return {"iceServers": twilio_servers}

    return {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]}
        ]
    }


def resize_with_letterbox(frame, target_width=DISPLAY_WIDTH, target_height=DISPLAY_HEIGHT):
    """Resize without distorting facial geometry and pad to a fixed processing size."""
    if frame is None or frame.size == 0:
        return frame

    h, w = frame.shape[:2]
    if h <= 0 or w <= 0:
        return frame

    scale = min(target_width / float(w), target_height / float(h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(
        frame,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
    )

    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    x0 = (target_width - new_w) // 2
    y0 = (target_height - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def get_square_box(landmarks, indices, frame_width, frame_height, expansion_ratio):
    xs, ys = [], []
    landmark_count = len(landmarks.landmark)
    for index in indices:
        if index >= landmark_count:
            continue
        point = landmarks.landmark[index]
        xs.append(point.x * frame_width)
        ys.append(point.y * frame_height)
    if not xs or not ys:
        return None
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    box_width, box_height = x_max - x_min, y_max - y_min
    if box_width <= 1 or box_height <= 1:
        return None
    center_x, center_y = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
    side = max(box_width, box_height) * (1.0 + expansion_ratio)
    side = min(side, float(frame_width), float(frame_height))
    half = side / 2.0
    x1, y1 = int(round(center_x - half)), int(round(center_y - half))
    x2, y2 = int(round(center_x + half)), int(round(center_y + half))
    if x1 < 0:
        x2 -= x1; x1 = 0
    if y1 < 0:
        y2 -= y1; y1 = 0
    if x2 > frame_width:
        shift = x2 - frame_width; x1 -= shift; x2 = frame_width
    if y2 > frame_height:
        shift = y2 - frame_height; y1 -= shift; y2 = frame_height
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame_width, x2), min(frame_height, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def preprocess_roi(frame, bounding_box):
    if bounding_box is None:
        return None
    x1, y1, x2, y2 = bounding_box
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    try:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    except cv2.error:
        return None
    return np.expand_dims(resized.astype(np.float32), axis=-1)


def get_mouth_ratio(landmarks, frame_width, frame_height):
    required = [MOUTH_LEFT, MOUTH_RIGHT, MOUTH_TOP, MOUTH_BOTTOM]
    if any(i >= len(landmarks.landmark) for i in required):
        return None
    left = landmarks.landmark[MOUTH_LEFT]
    right = landmarks.landmark[MOUTH_RIGHT]
    top = landmarks.landmark[MOUTH_TOP]
    bottom = landmarks.landmark[MOUTH_BOTTOM]
    width = abs((right.x - left.x) * frame_width)
    height = abs((bottom.y - top.y) * frame_height)
    if height < 1e-6:
        return None
    return float(width / height)


def draw_box(frame, box, color, label=None):
    """Draw a clean ROI rectangle without text over the driver's face."""
    if box is None:
        return
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)


def draw_hud_panel(
    img,
    face_detected,
    eye_status,
    mouth_status,
    left_prob,
    right_prob,
    yawn_prob,
    mouth_ratio,
    closed_duration,
    yawn_duration,
    alarm_text,
):
    """Attach a compact side HUD that fits the optimized 640x480 live stream."""
    h, w = img.shape[:2]
    panel_w = 320
    panel = np.full((h, panel_w, 3), (29, 32, 40), dtype=np.uint8)

    white = (242, 242, 242)
    muted = (170, 178, 190)
    green = (80, 210, 120)
    red = (70, 70, 235)
    orange = (60, 170, 255)
    cyan = (230, 190, 70)
    divider = (65, 70, 82)

    def put(value, x, y, scale=0.50, color=white, thickness=1):
        cv2.putText(
            panel, str(value), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
            scale, color, thickness, cv2.LINE_AA
        )

    put("LIVE FATIGUE MONITOR", 18, 30, 0.60, white, 2)
    put("Dual 2D-CNN | FrameStep 2", 18, 52, 0.40, muted, 1)
    cv2.line(panel, (18, 65), (panel_w - 18, 65), divider, 1, cv2.LINE_AA)

    face_color = green if face_detected else red
    eye_color = red if eye_status == "CLOSED" else (green if eye_status == "OPEN" else muted)
    mouth_color = (
        orange if mouth_status == "YAWNING"
        else cyan if "SMILE" in mouth_status
        else green if mouth_status == "NORMAL"
        else muted
    )

    put("FACE", 18, 92, 0.42, muted, 1)
    put("DETECTED" if face_detected else "NOT DETECTED", 110, 92, 0.50, face_color, 2)
    put("EYES", 18, 120, 0.42, muted, 1)
    put(eye_status, 110, 120, 0.50, eye_color, 2)
    put("MOUTH", 18, 148, 0.42, muted, 1)
    put(mouth_status, 110, 148, 0.46, mouth_color, 2)

    cv2.line(panel, (18, 162), (panel_w - 18, 162), divider, 1, cv2.LINE_AA)

    lp = "--" if left_prob is None else f"{left_prob:.3f}"
    rp = "--" if right_prob is None else f"{right_prob:.3f}"
    yp = "--" if yawn_prob is None else f"{yawn_prob:.3f}"
    mr = "--" if mouth_ratio is None else f"{mouth_ratio:.3f}"

    put("CNN OUTPUTS", 18, 184, 0.42, muted, 1)
    put(f"Left eye open      {lp}", 18, 208, 0.43, white, 1)
    put(f"Right eye open     {rp}", 18, 232, 0.43, white, 1)
    put(f"Yawn probability   {yp}", 18, 256, 0.43, white, 1)
    put(f"Mouth ratio        {mr}", 18, 280, 0.43, white, 1)

    cv2.line(panel, (18, 294), (panel_w - 18, 294), divider, 1, cv2.LINE_AA)

    put("ACTUAL-TIME DURATION", 18, 316, 0.42, muted, 1)
    put(f"Eye closure  {closed_duration:4.2f} s", 18, 340, 0.44, white, 1)
    put(f"Yawn         {yawn_duration:4.2f} s", 18, 382, 0.44, white, 1)

    bar_x1, bar_x2 = 18, panel_w - 18
    bar_w = max(1, bar_x2 - bar_x1)
    for y, value, color in [
        (348, closed_duration, red),
        (390, yawn_duration, orange),
    ]:
        cv2.rectangle(panel, (bar_x1, y), (bar_x2, y + 7), (55, 59, 69), -1)
        frac = min(max(value / ALARM_SECONDS, 0.0), 1.0)
        fill_x = bar_x1 + int(bar_w * frac)
        if fill_x > bar_x1:
            cv2.rectangle(panel, (bar_x1, y), (fill_x, y + 7), color, -1)

    card_y1 = max(410, h - 58)
    card_y2 = h - 10
    if alarm_text:
        card_color = (45, 45, 185)
        status_line1 = "FATIGUE ALERT"
        status_line2 = "EYE CLOSURE >= 2.0 s" if "EYE" in alarm_text else "YAWNING >= 2.0 s"
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), red, 4)
    else:
        card_color = (42, 110, 70)
        status_line1 = "STATUS: NORMAL"
        status_line2 = "Monitoring active"

    cv2.rectangle(panel, (14, card_y1), (panel_w - 14, card_y2), card_color, -1)
    put(status_line1, 26, card_y1 + 21, 0.50, white, 2)
    put(status_line2, 26, min(card_y1 + 42, card_y2 - 4), 0.36, white, 1)

    badge_text = "NORMAL" if not alarm_text else "FATIGUE ALERT"
    badge_color = (45, 145, 75) if not alarm_text else (45, 45, 200)
    cv2.rectangle(img, (12, 12), (200, 46), badge_color, -1)
    cv2.putText(
        img, badge_text, (24, 36), cv2.FONT_HERSHEY_SIMPLEX,
        0.56, (255, 255, 255), 2, cv2.LINE_AA
    )

    return np.hstack([img, panel])


def play_alarm_pattern(alarm_kind="FATIGUE"):
    """Play a short non-blocking Windows alarm pattern.

    This runs on the computer hosting Streamlit. In the user's local Windows
    setup, that is the same laptop that displays the dashboard.
    """
    if not WINDOWS_SOUND_AVAILABLE:
        return

    try:
        if alarm_kind == "EYE":
            # Higher-pitched two-tone pattern for prolonged eye closure.
            pattern = [(1800, 220), (1200, 180), (1800, 220)]
        elif alarm_kind == "YAWN":
            # Three short pulses for prolonged yawning.
            pattern = [(1450, 180), (1450, 180), (1450, 260)]
        else:
            pattern = [(1600, 220), (1000, 180), (1600, 300)]

        for frequency, duration_ms in pattern:
            winsound.Beep(frequency, duration_ms)
            time.sleep(0.06)
    except Exception:
        # Audio should never crash the monitoring pipeline.
        pass


def start_alarm_sound(alarm_kind="FATIGUE"):
    """Start the alarm sound in a daemon thread so video processing never blocks."""
    threading.Thread(
        target=play_alarm_pattern,
        args=(alarm_kind,),
        daemon=True,
    ).start()


@st.cache_resource(show_spinner="Loading eye and mouth CNN models...")
def load_realtime_models():
    import tensorflow as tf

    if not EYE_MODEL_PATH.exists():
        raise FileNotFoundError(f"Eye model not found: {EYE_MODEL_PATH}")
    if not MOUTH_MODEL_PATH.exists():
        raise FileNotFoundError(f"Mouth model not found: {MOUTH_MODEL_PATH}")

    eye_model = tf.keras.models.load_model(str(EYE_MODEL_PATH), compile=False)
    mouth_model = tf.keras.models.load_model(str(MOUTH_MODEL_PATH), compile=False)

    try:
        eye_model(
            np.zeros((2, IMAGE_SIZE, IMAGE_SIZE, 1), dtype=np.float32),
            training=False,
        )
        mouth_model(
            np.zeros((1, IMAGE_SIZE, IMAGE_SIZE, 1), dtype=np.float32),
            training=False,
        )
    except Exception:
        pass

    return eye_model, mouth_model


# ============================================================
# PAGES
# ============================================================
if page == "Overview":
    st.subheader("Project Overview")
    st.markdown(
        "This dashboard summarises the thesis results and includes a browser-based "
        "real-time monitoring prototype using the same eye/mouth decision rules."
    )
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Evaluated Videos", "319")
    c2.metric("Accuracy", "81.50%")
    c3.metric("Fatigue Precision", "98.25%")
    c4.metric("Specificity", "99.51%")
    c5.metric("False Alarm Rate", "0.49%")

    st.markdown("### System Pipeline")
    st.graphviz_chart(
        """
        digraph G {
          rankdir=LR;
          node [shape=box, style="rounded"];
          A [label="RGB Camera / Video"];
          B [label="MediaPipe Face Mesh"];
          C [label="Eye ROIs"];
          D [label="Mouth ROI"];
          E [label="Eye 2D-CNN"];
          F [label="Mouth 2D-CNN"];
          G [label="Geometric Filter"];
          H [label="Actual-Time Duration"];
          I [label="Fatigue Alarm"];
          A -> B; B -> C; B -> D; C -> E; D -> F; F -> G; E -> H; G -> H; H -> I;
        }
        """,
        width="stretch",
    )

elif page == "Real-Time Monitoring":
    st.subheader("Real-Time Fatigue Monitoring")
    st.markdown(
        "Click **START** below and allow camera access in the browser. "
        "This optimized prototype uses **FrameStep = 2**, 640x480 processing, "
        "MediaPipe Face Mesh, the two trained 2D-CNN models, geometric filtering, "
        "and actual-time duration judgement."
    )

    p1, p2, p3, p4, p5 = st.columns(5)
    p1.metric("FrameStep", "2")
    p2.metric("Eye Threshold", "0.50")
    p3.metric("Yawn Threshold", "0.50")
    p4.metric("Mouth Ratio", "<= 1.50")
    p5.metric("Alarm Duration", "2.0 s")

    if WINDOWS_SOUND_AVAILABLE:
        sound_col1, sound_col2 = st.columns([3, 1])
        with sound_col1:
            st.success(
                "Local Windows alarm sound is enabled. "
                "Visual alerts remain active on all deployments."
            )
        with sound_col2:
            if st.button("Test alarm sound"):
                start_alarm_sound("FATIGUE")
    else:
        st.info(
            "Server-side Windows alarm sound is unavailable on this host. "
            "Visual fatigue alerts remain active."
        )

    twilio_servers, twilio_status = get_twilio_ice_servers()
    if twilio_servers:
        st.success(
            "Cloud camera relay: Twilio STUN/TURN is configured. "
            f"Status: {twilio_status}"
        )
    else:
        st.error(
            "Twilio STUN/TURN is NOT active. "
            f"Diagnostic: {twilio_status}"
        )
        st.warning(
            "The app will temporarily fall back to Google STUN, "
            "but the cloud camera may fail to connect on restrictive networks."
        )

    if not EYE_MODEL_PATH.exists() or not MOUTH_MODEL_PATH.exists():
        st.error(
            "The two trained model files are required for real-time detection. "
            "Please make sure these files exist:\n\n"
            f"- `{EYE_MODEL_PATH}`\n"
            f"- `{MOUTH_MODEL_PATH}`"
        )
    else:
        try:
            from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
            import av
            import mediapipe as mp

            eye_model, mouth_model = load_realtime_models()

            class FatigueVideoProcessor(VideoProcessorBase):
                def __init__(self):
                    self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                        static_image_mode=False,
                        max_num_faces=1,
                        refine_landmarks=False,
                        min_detection_confidence=0.50,
                        min_tracking_confidence=0.50,
                    )

                    self.frame_index = 0
                    self.eye_closed_start = None
                    self.yawn_start = None
                    self.alarm_active = False
                    self.last_alarm_sound_time = 0.0
                    self.last_alarm_kind = None

                    self.face_detected = False
                    self.left_prob = None
                    self.right_prob = None
                    self.yawn_prob = None
                    self.mouth_ratio = None
                    self.eye_status = "Unknown"
                    self.mouth_status = "Unknown"
                    self.left_box = None
                    self.right_box = None
                    self.mouth_box = None

                @staticmethod
                def _run_model(model, batch):
                    output = model(batch, training=False)
                    if hasattr(output, "numpy"):
                        output = output.numpy()
                    return np.asarray(output).reshape(-1)

                def _reset_detection_state(self):
                    self.face_detected = False
                    self.left_prob = None
                    self.right_prob = None
                    self.yawn_prob = None
                    self.mouth_ratio = None
                    self.eye_status = "Unknown"
                    self.mouth_status = "Unknown"
                    self.left_box = None
                    self.right_box = None
                    self.mouth_box = None
                    self.eye_closed_start = None
                    self.yawn_start = None

                def _durations(self, now):
                    closed_duration = 0.0
                    yawn_duration = 0.0

                    if self.eye_status == "CLOSED" and self.eye_closed_start is not None:
                        closed_duration = max(0.0, now - self.eye_closed_start)

                    if self.mouth_status == "YAWNING" and self.yawn_start is not None:
                        yawn_duration = max(0.0, now - self.yawn_start)

                    return closed_duration, yawn_duration

                def recv(self, frame):
                    raw = frame.to_ndarray(format="bgr24")
                    raw = cv2.flip(raw, 1)
                    img = resize_with_letterbox(raw)

                    h, w = img.shape[:2]
                    now = time.monotonic()

                    self.frame_index += 1
                    do_inference = (self.frame_index % FRAME_STEP) == 1

                    if do_inference:
                        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        result = self.face_mesh.process(rgb)
                        self.face_detected = bool(result.multi_face_landmarks)

                        if self.face_detected:
                            lm = result.multi_face_landmarks[0]

                            self.left_box = get_square_box(
                                lm, LEFT_EYE, w, h, EYE_EXPANSION
                            )
                            self.right_box = get_square_box(
                                lm, RIGHT_EYE, w, h, EYE_EXPANSION
                            )
                            self.mouth_box = get_square_box(
                                lm, MOUTH, w, h, MOUTH_EXPANSION
                            )

                            left_roi = preprocess_roi(img, self.left_box)
                            right_roi = preprocess_roi(img, self.right_box)
                            mouth_roi = preprocess_roi(img, self.mouth_box)

                            if left_roi is not None and right_roi is not None:
                                eye_batch = np.stack([left_roi, right_roi], axis=0)
                                preds = self._run_model(eye_model, eye_batch)

                                if len(preds) >= 2:
                                    self.left_prob = float(preds[0])
                                    self.right_prob = float(preds[1])

                                    eyes_closed = (
                                        self.left_prob < EYE_THRESHOLD
                                        and self.right_prob < EYE_THRESHOLD
                                    )

                                    if eyes_closed:
                                        self.eye_status = "CLOSED"
                                        if self.eye_closed_start is None:
                                            self.eye_closed_start = now
                                    else:
                                        self.eye_status = "OPEN"
                                        self.eye_closed_start = None
                                else:
                                    self.left_prob = None
                                    self.right_prob = None
                                    self.eye_status = "Unknown"
                                    self.eye_closed_start = None
                            else:
                                self.left_prob = None
                                self.right_prob = None
                                self.eye_status = "Unknown"
                                self.eye_closed_start = None

                            self.mouth_ratio = get_mouth_ratio(lm, w, h)

                            if mouth_roi is not None:
                                pred = self._run_model(
                                    mouth_model,
                                    np.expand_dims(mouth_roi, axis=0),
                                )

                                if len(pred):
                                    self.yawn_prob = float(pred[0])

                                    likely_smile = (
                                        self.mouth_ratio is not None
                                        and self.mouth_ratio > SMILE_RATIO_THRESHOLD
                                    )

                                    yawning = (
                                        self.yawn_prob >= MOUTH_THRESHOLD
                                        and not likely_smile
                                    )

                                    if yawning:
                                        self.mouth_status = "YAWNING"
                                        if self.yawn_start is None:
                                            self.yawn_start = now
                                    elif (
                                        likely_smile
                                        and self.yawn_prob >= MOUTH_THRESHOLD
                                    ):
                                        self.mouth_status = "SMILE / IGNORED"
                                        self.yawn_start = None
                                    else:
                                        self.mouth_status = "NORMAL"
                                        self.yawn_start = None
                                else:
                                    self.yawn_prob = None
                                    self.mouth_status = "Unknown"
                                    self.yawn_start = None
                            else:
                                self.yawn_prob = None
                                self.mouth_status = "Unknown"
                                self.yawn_start = None
                        else:
                            self._reset_detection_state()

                    closed_duration, yawn_duration = self._durations(now)

                    if self.face_detected:
                        draw_box(
                            img,
                            self.left_box,
                            (0, 255, 0) if self.eye_status == "OPEN" else (0, 0, 255),
                        )
                        draw_box(
                            img,
                            self.right_box,
                            (0, 255, 0) if self.eye_status == "OPEN" else (0, 0, 255),
                        )
                        draw_box(
                            img,
                            self.mouth_box,
                            (0, 165, 255)
                            if self.mouth_status == "YAWNING"
                            else (255, 255, 0),
                        )

                    alarm_text = None
                    alarm_kind = None

                    if closed_duration >= ALARM_SECONDS:
                        alarm_text = "WARNING: PROLONGED EYE CLOSURE"
                        alarm_kind = "EYE"

                    if yawn_duration >= ALARM_SECONDS:
                        alarm_text = "WARNING: PROLONGED YAWNING"
                        alarm_kind = "YAWN"

                    if alarm_text:
                        should_sound = (
                            not self.alarm_active
                            or self.last_alarm_kind != alarm_kind
                            or (now - self.last_alarm_sound_time)
                            >= ALARM_SOUND_REPEAT_SECONDS
                        )

                        if should_sound:
                            start_alarm_sound(alarm_kind or "FATIGUE")
                            self.last_alarm_sound_time = now
                            self.last_alarm_kind = alarm_kind

                        self.alarm_active = True
                    else:
                        self.alarm_active = False
                        self.last_alarm_kind = None

                    display = draw_hud_panel(
                        img,
                        self.face_detected,
                        self.eye_status,
                        self.mouth_status,
                        self.left_prob,
                        self.right_prob,
                        self.yawn_prob,
                        self.mouth_ratio,
                        closed_duration,
                        yawn_duration,
                        alarm_text,
                    )

                    return av.VideoFrame.from_ndarray(display, format="bgr24")

            webrtc_streamer(
                key="fatigue-monitor",
                video_processor_factory=FatigueVideoProcessor,
                media_stream_constraints={
                    "video": {
                        "width": {"ideal": DISPLAY_WIDTH},
                        "height": {"ideal": DISPLAY_HEIGHT},
                        "frameRate": {"ideal": CAMERA_IDEAL_FPS, "max": 30},
                    },
                    "audio": False,
                },
                rtc_configuration=build_rtc_configuration(),
                async_processing=True,
            )

            st.caption(
                "Camera frames are processed in memory for the live prototype; "
                "no webcam recording is saved. MediaPipe/CNN inference runs every "
                "other frame (FrameStep = 2), while the 2.0-second decision uses "
                "actual elapsed time."
            )

        except ImportError as exc:
            st.error(
                "Real-time browser camera support is not installed. "
                "Ensure requirements.txt contains streamlit-webrtc, av, "
                "mediapipe, and twilio.\n\n"
                f"Missing package details: {exc}"
            )
        except Exception as exc:
            st.exception(exc)

elif page == "Model Performance":
    st.subheader("Local CNN and Traditional Machine-Learning Performance")
    st.markdown("### Independent Image-Level CNN Results")
    st.dataframe(image_models, width="stretch", hide_index=True)
    st.bar_chart(image_models.set_index("Model")[["Accuracy (%)", "Macro F1 (%)", "ROC-AUC (%)"]])
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Eye-State CNN Confusion Matrix")
        st.table(eye_cm)
    with c2:
        st.markdown("#### Mouth-State CNN Confusion Matrix")
        st.table(mouth_cm)
    st.markdown("### Traditional Machine-Learning Baseline")
    st.dataframe(traditional_ml, width="stretch", hide_index=True)
    st.markdown("#### Random Forest Video-Level Confusion Matrix")
    st.table(rf_cm)
    st.info("Traditional ML uses 51 independent test videos; the threshold baseline and final dual-CNN use 319 labelled videos.")

elif page == "Video-Level Evaluation":
    st.subheader("Final Video-Level System Evaluation")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", "81.50%")
    c2.metric("Fatigue Precision", "98.25%")
    c3.metric("Fatigue Recall", "49.12%")
    c4.metric("Fatigue F1", "65.50%")
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Specificity", "99.51%")
    c6.metric("Macro F1", "76.43%")
    c7.metric("False Alarm Rate", "0.49%")
    c8.metric("Miss Rate", "50.88%")
    st.markdown("### Final Dual-CNN Confusion Matrix")
    st.table(final_cm)
    st.markdown("### Same-Dataset Comparison")
    st.dataframe(same_dataset, width="stretch", hide_index=True)
    st.bar_chart(same_dataset.set_index("Method")[["Accuracy (%)", "Fatigue F1 (%)"]])

elif page == "Geometric Filtering":
    st.subheader("Mouth Geometric Filtering Ablation")
    st.dataframe(geometry, width="stretch", hide_index=True)
    metric = st.selectbox("Select a metric to compare",
                          ["Accuracy (%)", "Fatigue Precision (%)", "Fatigue Recall (%)",
                           "Specificity (%)", "Fatigue F1 (%)", "Macro F1 (%)",
                           "False Alarm Rate (%)", "Miss Rate (%)"], index=2)
    st.bar_chart(geometry.set_index("Setting")[[metric]])
    st.warning("The geometric filter reduces false alarms but introduces a substantial recall trade-off.")

elif page == "FrameStep & Runtime":
    st.subheader("FrameStep Sensitivity and Runtime Efficiency")
    st.dataframe(framestep, width="stretch", hide_index=True)
    st.markdown("### Classification Performance")
    st.line_chart(framestep.set_index("FrameStep")[["Accuracy (%)", "Fatigue Recall (%)", "Fatigue F1 (%)"]])
    st.markdown("### Effective Processing Speed")
    st.line_chart(framestep.set_index("FrameStep")[["Processing FPS", "Effective Source FPS"]])
    st.markdown("### Real-Time Factor")
    st.line_chart(framestep.set_index("FrameStep")[["Real-Time Factor"]])
    st.success("FrameStep 2 was selected for the final system; mean trigger delay remained about 2.002 s.")

elif page == "Research Questions":
    st.subheader("Research Questions and Evidence")
    rq = pd.DataFrame([
        ["RQ1", "Eye CNN 99.05%; mouth CNN 97.66%; final accuracy 81.50%; threshold baseline 54.86%.",
         "Dual-CNN improves overall performance, but video-level fatigue recall remains limited."],
        ["RQ2", "FAR 5.85% → 0.49%; false positives 12 → 1; recall 83.33% → 49.12%.",
         "Geometric filtering controls false alarms but causes a substantial recall trade-off."],
        ["RQ3", "Trigger delay ≈ 2.002 s across FrameStep 1–4; RTF 1.1007 at FrameStep 2.",
         "Actual-time duration judgement remains stable under tested FrameStep settings."],
    ], columns=["Research Question", "Main Evidence", "Conclusion"])
    st.dataframe(rq, width="stretch", hide_index=True)

elif page == "Code & Resources":
    st.subheader("Source Code and Research Resources")
    resources = [
        ("Complete Source Code Repository", GITHUB_BASE),
        ("Eye-State CNN Training", f"{GITHUB_BASE}/blob/main/train_2d_cnn_eyes.py"),
        ("Mouth-State CNN Training", f"{GITHUB_BASE}/blob/main/train_2d_cnn_mouth.py"),
        ("Facial Feature Extraction", f"{GITHUB_BASE}/blob/main/extract_features.py"),
        ("Traditional Machine-Learning Baseline", f"{GITHUB_BASE}/blob/main/train_baseline_models.py"),
        ("Yawning Detection Evaluation", f"{GITHUB_BASE}/blob/main/evaluate_yawdd_performance.py"),
        ("Final Video-Level System Evaluation", f"{GITHUB_BASE}/blob/main/test_videos_with_real_system.py"),
        ("Real-Time Fatigue Detection Prototype", f"{GITHUB_BASE}/blob/main/realtime_demo_2dcnn.py"),
    ]
    for title, url in resources:
        st.markdown(f"**{title}**")
        st.markdown(f"[{url}]({url})")
