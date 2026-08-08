"""
Real-Time Fatigue Detection System

Functions:
    1. Detect eye and mouth regions with MediaPipe Face Mesh.
    2. Classify eye state and mouth state using two 2D-CNN models.
    3. Ignore likely smiles using mouth width-to-height ratio.
    4. Trigger an alarm when eye closure or yawning lasts for 2 seconds.

Model labels:
    Eye model:
        0 = Closed eyes
        1 = Open eyes

    Mouth model:
        0 = Normal / No yawn
        1 = Yawn
"""

import os

# 减少TensorFlow启动时的普通INFO提示
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import threading
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import winsound


# ============================================================
# 1. 路径设置
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\DELL\PycharmProjects\pythonProject9"
    r"\BusDriverFatigueDetection"
)

EYE_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "best_eye_2dcnn.h5"
)

MOUTH_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "best_mouth_2dcnn.h5"
)


# ============================================================
# 2. 摄像头与模型参数
# ============================================================

CAMERA_INDEX = 0

FRAME_WIDTH = 960
FRAME_HEIGHT = 540

MODEL_IMAGE_WIDTH = 80
MODEL_IMAGE_HEIGHT = 80

EYE_PREDICTION_THRESHOLD = 0.50
MOUTH_PREDICTION_THRESHOLD = 0.50

# 闭眼或哈欠持续时间超过该值后报警
ALARM_DURATION_THRESHOLD = 2.0

# 两次声音警报之间的最短间隔
ALARM_COOLDOWN_SECONDS = 1.0

# ROI扩张比例
EYE_ROI_EXPANSION_RATIO = 0.25
MOUTH_ROI_EXPANSION_RATIO = 0.25

# 嘴巴宽高比大于该值时，优先认为是笑容
SMILE_RATIO_THRESHOLD = 1.50

ALARM_FREQUENCY = 1500
ALARM_SOUND_DURATION_MS = 500


# ============================================================
# 3. MediaPipe关键点
# ============================================================

# 左眼和右眼主要关键点
LEFT_EYE_INDICES = [
    33,
    160,
    158,
    133,
    153,
    144
]

RIGHT_EYE_INDICES = [
    362,
    385,
    387,
    263,
    373,
    380
]

# 嘴唇区域关键点
MOUTH_INDICES = [
    61,
    146,
    91,
    181,
    84,
    17,
    314,
    405,
    321,
    375,
    291,
    308,
    324,
    318,
    402,
    317,
    14,
    87,
    178,
    88,
    95,
    78,
    191,
    80,
    81,
    82,
    13,
    312,
    311,
    310,
    415
]

# 嘴巴比例使用的四个关键点
MOUTH_LEFT_INDEX = 61
MOUTH_RIGHT_INDEX = 291
MOUTH_TOP_INDEX = 0
MOUTH_BOTTOM_INDEX = 17


# ============================================================
# 4. 显示参数
# ============================================================

COLOR_GREEN = (
    0,
    255,
    0
)

COLOR_RED = (
    0,
    0,
    255
)

COLOR_ORANGE = (
    0,
    165,
    255
)

COLOR_YELLOW = (
    0,
    255,
    255
)

COLOR_PINK = (
    255,
    105,
    180
)

COLOR_WHITE = (
    255,
    255,
    255
)

COLOR_BLACK = (
    0,
    0,
    0
)

COLOR_CYAN = (
    255,
    255,
    0
)


# ============================================================
# 5. 模型加载
# ============================================================

def load_cnn_model(
    model_path,
    model_name
):
    """读取训练完成的Keras模型。"""

    if not model_path.exists():

        raise FileNotFoundError(
            f"{model_name}模型不存在：\n"
            f"{model_path}"
        )

    try:

        model = tf.keras.models.load_model(
            str(model_path),
            compile=False
        )

    except Exception as error:

        raise RuntimeError(
            f"{model_name}模型加载失败：{error}"
        ) from error

    return model


# ============================================================
# 6. ROI区域处理
# ============================================================

def get_square_bounding_box(
    landmarks,
    landmark_indices,
    frame_width,
    frame_height,
    expansion_ratio=0.25
):
    """
    根据关键点生成正方形ROI。

    先取关键点的最小包围框，再以较长边为基准，
    按比例扩张并形成1:1区域。
    """

    if landmarks is None:
        return None

    if not landmark_indices:
        return None

    x_coordinates = []
    y_coordinates = []

    for index in landmark_indices:

        if index >= len(
            landmarks.landmark
        ):
            continue

        landmark = landmarks.landmark[
            index
        ]

        x_coordinates.append(
            landmark.x * frame_width
        )

        y_coordinates.append(
            landmark.y * frame_height
        )

    if (
        len(x_coordinates) == 0
        or len(y_coordinates) == 0
    ):
        return None

    x_min = min(
        x_coordinates
    )

    x_max = max(
        x_coordinates
    )

    y_min = min(
        y_coordinates
    )

    y_max = max(
        y_coordinates
    )

    box_width = (
        x_max - x_min
    )

    box_height = (
        y_max - y_min
    )

    if (
        box_width <= 1
        or box_height <= 1
    ):
        return None

    center_x = (
        x_min + x_max
    ) / 2.0

    center_y = (
        y_min + y_max
    ) / 2.0

    # 在较长边基础上增加25%
    square_side = max(
        box_width,
        box_height
    )

    square_side = (
        square_side
        * (
            1.0
            + expansion_ratio
        )
    )

    half_side = (
        square_side / 2.0
    )

    roi_x_min = int(
        round(
            center_x - half_side
        )
    )

    roi_y_min = int(
        round(
            center_y - half_side
        )
    )

    roi_x_max = int(
        round(
            center_x + half_side
        )
    )

    roi_y_max = int(
        round(
            center_y + half_side
        )
    )

    # 将ROI限制在画面范围内
    roi_x_min = max(
        0,
        roi_x_min
    )

    roi_y_min = max(
        0,
        roi_y_min
    )

    roi_x_max = min(
        frame_width,
        roi_x_max
    )

    roi_y_max = min(
        frame_height,
        roi_y_max
    )

    if (
        roi_x_max <= roi_x_min
        or roi_y_max <= roi_y_min
    ):
        return None

    return (
        roi_x_min,
        roi_y_min,
        roi_x_max,
        roi_y_max
    )


def preprocess_roi(
    frame,
    bounding_box
):
    """
    裁剪ROI并转换为模型输入格式。

    输出形状：
        80 × 80 × 1
    """

    if bounding_box is None:
        return None

    (
        x_min,
        y_min,
        x_max,
        y_max
    ) = bounding_box

    roi = frame[
        y_min:y_max,
        x_min:x_max
    ]

    if roi.size == 0:
        return None

    try:

        gray_roi = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2GRAY
        )

        resized_roi = cv2.resize(
            gray_roi,
            (
                MODEL_IMAGE_WIDTH,
                MODEL_IMAGE_HEIGHT
            ),
            interpolation=cv2.INTER_LINEAR
        )

    except cv2.error:

        return None

    resized_roi = resized_roi.astype(
        np.float32
    )

    # 添加通道维度，结果为80×80×1
    model_input = np.expand_dims(
        resized_roi,
        axis=-1
    )

    return model_input


# ============================================================
# 7. 嘴巴宽高比
# ============================================================

def get_mouth_width_height_ratio(
    landmarks,
    frame_width,
    frame_height
):
    """
    计算嘴巴宽度与高度之比。

    使用关键点：
        左侧：61
        右侧：291
        上侧：0
        下侧：17
    """

    required_indices = [
        MOUTH_LEFT_INDEX,
        MOUTH_RIGHT_INDEX,
        MOUTH_TOP_INDEX,
        MOUTH_BOTTOM_INDEX
    ]

    if any(
        index >= len(
            landmarks.landmark
        )
        for index in required_indices
    ):
        return None

    left_point = landmarks.landmark[
        MOUTH_LEFT_INDEX
    ]

    right_point = landmarks.landmark[
        MOUTH_RIGHT_INDEX
    ]

    top_point = landmarks.landmark[
        MOUTH_TOP_INDEX
    ]

    bottom_point = landmarks.landmark[
        MOUTH_BOTTOM_INDEX
    ]

    mouth_width = abs(
        (
            right_point.x
            - left_point.x
        )
        * frame_width
    )

    mouth_height = abs(
        (
            bottom_point.y
            - top_point.y
        )
        * frame_height
    )

    if mouth_height < 1e-6:
        return None

    return float(
        mouth_width
        / mouth_height
    )


# ============================================================
# 8. 绘图函数
# ============================================================

def draw_bounding_box(
    frame,
    bounding_box,
    color
):
    """绘制ROI边界框。"""

    if bounding_box is None:
        return

    (
        x_min,
        y_min,
        x_max,
        y_max
    ) = bounding_box

    cv2.rectangle(
        frame,
        (
            x_min,
            y_min
        ),
        (
            x_max,
            y_max
        ),
        color,
        1
    )


def draw_face_connections(
    frame,
    landmarks,
    connections,
    frame_width,
    frame_height,
    color
):
    """绘制眼睛或嘴巴关键点连线。"""

    for start_index, end_index in connections:

        if (
            start_index
            >= len(landmarks.landmark)
            or end_index
            >= len(landmarks.landmark)
        ):
            continue

        start_landmark = landmarks.landmark[
            start_index
        ]

        end_landmark = landmarks.landmark[
            end_index
        ]

        start_point = (
            int(
                start_landmark.x
                * frame_width
            ),
            int(
                start_landmark.y
                * frame_height
            )
        )

        end_point = (
            int(
                end_landmark.x
                * frame_width
            ),
            int(
                end_landmark.y
                * frame_height
            )
        )

        cv2.line(
            frame,
            start_point,
            end_point,
            color,
            1
        )


def draw_text_with_background(
    frame,
    text,
    position,
    text_color,
    font_scale=0.65,
    thickness=2
):
    """绘制带黑色背景的文字。"""

    font = cv2.FONT_HERSHEY_SIMPLEX

    (
        text_width,
        text_height
    ), baseline = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness
    )

    x_position, y_position = position

    cv2.rectangle(
        frame,
        (
            x_position - 5,
            y_position
            - text_height
            - 6
        ),
        (
            x_position
            + text_width
            + 5,
            y_position
            + baseline
            + 5
        ),
        COLOR_BLACK,
        -1
    )

    cv2.putText(
        frame,
        text,
        position,
        font,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA
    )


def draw_warning_banner(
    frame,
    warning_text,
    vertical_position,
    color
):
    """在画面中央绘制警报文字。"""

    frame_height, frame_width = (
        frame.shape[:2]
    )

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.1
    thickness = 3

    (
        text_width,
        text_height
    ), baseline = cv2.getTextSize(
        warning_text,
        font,
        font_scale,
        thickness
    )

    text_x = max(
        10,
        (
            frame_width
            - text_width
        ) // 2
    )

    text_y = min(
        frame_height - 10,
        vertical_position
    )

    cv2.rectangle(
        frame,
        (
            text_x - 12,
            text_y
            - text_height
            - 12
        ),
        (
            text_x
            + text_width
            + 12,
            text_y
            + baseline
            + 12
        ),
        COLOR_BLACK,
        -1
    )

    cv2.putText(
        frame,
        warning_text,
        (
            text_x,
            text_y
        ),
        font,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA
    )


# ============================================================
# 9. 报警声音
# ============================================================

def play_alarm_sound():
    """播放Windows蜂鸣声。"""

    try:

        winsound.Beep(
            ALARM_FREQUENCY,
            ALARM_SOUND_DURATION_MS
        )

    except RuntimeError:

        pass


def start_alarm_thread():
    """使用后台线程播放声音，避免阻塞视频画面。"""

    alarm_thread = threading.Thread(
        target=play_alarm_sound,
        daemon=True
    )

    alarm_thread.start()


# ============================================================
# 10. 模型预测
# ============================================================

def predict_eye_states(
    eye_model,
    left_eye_image,
    right_eye_image
):
    """
    在一个批次中预测左右眼状态。

    返回：
        left_probability
        right_probability
    """

    if (
        left_eye_image is None
        or right_eye_image is None
    ):
        return None, None

    eye_batch = np.stack(
        [
            left_eye_image,
            right_eye_image
        ],
        axis=0
    )

    predictions = eye_model.predict(
        eye_batch,
        verbose=0
    ).reshape(-1)

    if len(predictions) < 2:
        return None, None

    return (
        float(predictions[0]),
        float(predictions[1])
    )


def predict_mouth_state(
    mouth_model,
    mouth_image
):
    """预测嘴巴属于哈欠类别的概率。"""

    if mouth_image is None:
        return None

    mouth_batch = np.expand_dims(
        mouth_image,
        axis=0
    )

    prediction = mouth_model.predict(
        mouth_batch,
        verbose=0
    ).reshape(-1)

    if len(prediction) == 0:
        return None

    return float(
        prediction[0]
    )


# ============================================================
# 11. 实时检测主程序
# ============================================================

def main():

    print("=" * 75)
    print("Real-Time 2D-CNN Fatigue Detection")
    print("=" * 75)

    # --------------------------------------------------------
    # 加载模型
    # --------------------------------------------------------

    print("\n正在加载眼睛状态模型……")

    try:

        eye_model = load_cnn_model(
            EYE_MODEL_PATH,
            "眼睛状态"
        )

        print("眼睛状态模型加载完成。")

        print("正在加载嘴巴状态模型……")

        mouth_model = load_cnn_model(
            MOUTH_MODEL_PATH,
            "嘴巴状态"
        )

        print("嘴巴状态模型加载完成。")

    except (
        FileNotFoundError,
        RuntimeError
    ) as error:

        print(
            f"\n[ERROR] {error}"
        )

        return

    # --------------------------------------------------------
    # 初始化MediaPipe
    # --------------------------------------------------------

    mp_face_mesh = (
        mp.solutions.face_mesh
    )

    left_eye_connections = (
        mp.solutions
        .face_mesh_connections
        .FACEMESH_LEFT_EYE
    )

    right_eye_connections = (
        mp.solutions
        .face_mesh_connections
        .FACEMESH_RIGHT_EYE
    )

    mouth_connections = (
        mp.solutions
        .face_mesh_connections
        .FACEMESH_LIPS
    )

    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.50,
        min_tracking_confidence=0.50
    )

    # --------------------------------------------------------
    # 打开摄像头
    # --------------------------------------------------------

    camera = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_DSHOW
    )

    if not camera.isOpened():

        # 某些环境不支持CAP_DSHOW，改用默认方式重试
        camera.release()

        camera = cv2.VideoCapture(
            CAMERA_INDEX
        )

    if not camera.isOpened():

        print(
            f"\n[ERROR] 无法打开摄像头："
            f"{CAMERA_INDEX}"
        )

        face_mesh.close()

        return

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        FRAME_WIDTH
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        FRAME_HEIGHT
    )

    print(
        "\n摄像头已启动，按Q键退出。"
    )

    # --------------------------------------------------------
    # 状态变量
    # --------------------------------------------------------

    eye_closed_start_time = None
    yawn_start_time = None
    last_alarm_time = (
        -ALARM_COOLDOWN_SECONDS
    )

    previous_frame_time = (
        time.monotonic()
    )

    display_fps = 0.0

    try:

        while True:

            success, frame = (
                camera.read()
            )

            if not success:

                print(
                    "\n[WARNING] 无法读取摄像头画面。"
                )

                break

            frame = cv2.flip(
                frame,
                1
            )

            frame_height, frame_width = (
                frame.shape[:2]
            )

            current_time = (
                time.monotonic()
            )

            frame_interval = (
                current_time
                - previous_frame_time
            )

            previous_frame_time = (
                current_time
            )

            if frame_interval > 0:

                current_fps = (
                    1.0
                    / frame_interval
                )

                if display_fps == 0:

                    display_fps = (
                        current_fps
                    )

                else:

                    display_fps = (
                        0.90
                        * display_fps
                        + 0.10
                        * current_fps
                    )

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            # MediaPipe输入图像设为不可写，可减少部分内存复制
            rgb_frame.flags.writeable = False

            results = face_mesh.process(
                rgb_frame
            )

            rgb_frame.flags.writeable = True

            # 默认显示状态
            face_status_text = (
                "Face: Not Detected"
            )

            eye_status_text = (
                "Eyes: Unknown"
            )

            mouth_status_text = (
                "Mouth: Unknown"
            )

            eye_color = COLOR_YELLOW
            mouth_color = COLOR_YELLOW

            left_eye_probability = None
            right_eye_probability = None
            mouth_probability = None
            mouth_ratio = None

            closed_duration = 0.0
            yawn_duration = 0.0

            eye_alarm = False
            yawn_alarm = False

            if results.multi_face_landmarks:

                face_landmarks = (
                    results.multi_face_landmarks[0]
                )

                face_status_text = (
                    "Face: Detected"
                )

                # ====================================================
                # 眼睛区域
                # ====================================================

                left_eye_box = (
                    get_square_bounding_box(
                        face_landmarks,
                        LEFT_EYE_INDICES,
                        frame_width,
                        frame_height,
                        EYE_ROI_EXPANSION_RATIO
                    )
                )

                right_eye_box = (
                    get_square_bounding_box(
                        face_landmarks,
                        RIGHT_EYE_INDICES,
                        frame_width,
                        frame_height,
                        EYE_ROI_EXPANSION_RATIO
                    )
                )

                left_eye_image = preprocess_roi(
                    frame,
                    left_eye_box
                )

                right_eye_image = preprocess_roi(
                    frame,
                    right_eye_box
                )

                (
                    left_eye_probability,
                    right_eye_probability
                ) = predict_eye_states(
                    eye_model,
                    left_eye_image,
                    right_eye_image
                )

                if (
                    left_eye_probability is not None
                    and right_eye_probability is not None
                ):

                    left_eye_closed = (
                        left_eye_probability
                        < EYE_PREDICTION_THRESHOLD
                    )

                    right_eye_closed = (
                        right_eye_probability
                        < EYE_PREDICTION_THRESHOLD
                    )

                    # 两只眼睛都判定为闭眼时才开始计时
                    both_eyes_closed = (
                        left_eye_closed
                        and right_eye_closed
                    )

                    if both_eyes_closed:

                        eye_status_text = (
                            "Eyes: CLOSED"
                        )

                        eye_color = COLOR_RED

                        if (
                            eye_closed_start_time
                            is None
                        ):

                            eye_closed_start_time = (
                                current_time
                            )

                        closed_duration = (
                            current_time
                            - eye_closed_start_time
                        )

                    else:

                        eye_status_text = (
                            "Eyes: Open"
                        )

                        eye_color = COLOR_GREEN

                        eye_closed_start_time = None

                        closed_duration = 0.0

                else:

                    eye_status_text = (
                        "Eyes: ROI Invalid"
                    )

                    eye_closed_start_time = None

                draw_face_connections(
                    frame,
                    face_landmarks,
                    left_eye_connections,
                    frame_width,
                    frame_height,
                    eye_color
                )

                draw_face_connections(
                    frame,
                    face_landmarks,
                    right_eye_connections,
                    frame_width,
                    frame_height,
                    eye_color
                )

                draw_bounding_box(
                    frame,
                    left_eye_box,
                    eye_color
                )

                draw_bounding_box(
                    frame,
                    right_eye_box,
                    eye_color
                )

                # ====================================================
                # 嘴巴区域
                # ====================================================

                mouth_box = (
                    get_square_bounding_box(
                        face_landmarks,
                        MOUTH_INDICES,
                        frame_width,
                        frame_height,
                        MOUTH_ROI_EXPANSION_RATIO
                    )
                )

                mouth_image = preprocess_roi(
                    frame,
                    mouth_box
                )

                mouth_probability = (
                    predict_mouth_state(
                        mouth_model,
                        mouth_image
                    )
                )

                mouth_ratio = (
                    get_mouth_width_height_ratio(
                        face_landmarks,
                        frame_width,
                        frame_height
                    )
                )

                if mouth_probability is not None:

                    predicted_as_yawn = (
                        mouth_probability
                        >= MOUTH_PREDICTION_THRESHOLD
                    )

                    if predicted_as_yawn:

                        likely_smile = (
                            mouth_ratio is not None
                            and mouth_ratio
                            > SMILE_RATIO_THRESHOLD
                        )

                        if likely_smile:

                            mouth_status_text = (
                                "Mouth: Smile - Ignored"
                            )

                            mouth_color = COLOR_PINK

                            yawn_start_time = None
                            yawn_duration = 0.0

                        else:

                            mouth_status_text = (
                                "Mouth: YAWNING"
                            )

                            mouth_color = COLOR_ORANGE

                            if yawn_start_time is None:

                                yawn_start_time = (
                                    current_time
                                )

                            yawn_duration = (
                                current_time
                                - yawn_start_time
                            )

                    else:

                        mouth_status_text = (
                            "Mouth: Normal"
                        )

                        mouth_color = COLOR_CYAN

                        yawn_start_time = None
                        yawn_duration = 0.0

                else:

                    mouth_status_text = (
                        "Mouth: ROI Invalid"
                    )

                    yawn_start_time = None

                draw_face_connections(
                    frame,
                    face_landmarks,
                    mouth_connections,
                    frame_width,
                    frame_height,
                    mouth_color
                )

                draw_bounding_box(
                    frame,
                    mouth_box,
                    mouth_color
                )

                # ====================================================
                # 持续时间判断
                # ====================================================

                eye_alarm = (
                    closed_duration
                    >= ALARM_DURATION_THRESHOLD
                )

                yawn_alarm = (
                    yawn_duration
                    >= ALARM_DURATION_THRESHOLD
                )

            else:

                # 人脸消失后清除计时，防止时间继续累计
                eye_closed_start_time = None
                yawn_start_time = None

            # ========================================================
            # HUD状态显示
            # ========================================================

            draw_text_with_background(
                frame,
                face_status_text,
                (
                    20,
                    35
                ),
                (
                    COLOR_GREEN
                    if results.multi_face_landmarks
                    else COLOR_YELLOW
                )
            )

            draw_text_with_background(
                frame,
                eye_status_text,
                (
                    20,
                    70
                ),
                eye_color
            )

            draw_text_with_background(
                frame,
                mouth_status_text,
                (
                    20,
                    105
                ),
                mouth_color
            )

            if (
                left_eye_probability is not None
                and right_eye_probability is not None
            ):

                eye_probability_text = (
                    f"Eye Open Prob. "
                    f"L={left_eye_probability:.3f} "
                    f"R={right_eye_probability:.3f}"
                )

            else:

                eye_probability_text = (
                    "Eye Open Prob. L=-- R=--"
                )

            draw_text_with_background(
                frame,
                eye_probability_text,
                (
                    20,
                    140
                ),
                COLOR_WHITE,
                font_scale=0.55,
                thickness=1
            )

            if mouth_probability is not None:

                mouth_probability_text = (
                    f"Yawn Probability: "
                    f"{mouth_probability:.3f}"
                )

            else:

                mouth_probability_text = (
                    "Yawn Probability: --"
                )

            draw_text_with_background(
                frame,
                mouth_probability_text,
                (
                    20,
                    170
                ),
                COLOR_WHITE,
                font_scale=0.55,
                thickness=1
            )

            if mouth_ratio is not None:

                mouth_ratio_text = (
                    f"Mouth Ratio: "
                    f"{mouth_ratio:.3f}"
                )

            else:

                mouth_ratio_text = (
                    "Mouth Ratio: --"
                )

            draw_text_with_background(
                frame,
                mouth_ratio_text,
                (
                    20,
                    200
                ),
                COLOR_WHITE,
                font_scale=0.55,
                thickness=1
            )

            draw_text_with_background(
                frame,
                (
                    f"Eye Closed Time: "
                    f"{closed_duration:.2f}s"
                ),
                (
                    20,
                    230
                ),
                (
                    COLOR_RED
                    if closed_duration > 0
                    else COLOR_WHITE
                ),
                font_scale=0.55,
                thickness=1
            )

            draw_text_with_background(
                frame,
                (
                    f"Yawn Time: "
                    f"{yawn_duration:.2f}s"
                ),
                (
                    20,
                    260
                ),
                (
                    COLOR_ORANGE
                    if yawn_duration > 0
                    else COLOR_WHITE
                ),
                font_scale=0.55,
                thickness=1
            )

            draw_text_with_background(
                frame,
                (
                    f"Alarm Threshold: "
                    f"{ALARM_DURATION_THRESHOLD:.1f}s"
                ),
                (
                    20,
                    290
                ),
                COLOR_WHITE,
                font_scale=0.55,
                thickness=1
            )

            draw_text_with_background(
                frame,
                (
                    f"FPS: "
                    f"{display_fps:.1f}"
                ),
                (
                    20,
                    320
                ),
                COLOR_WHITE,
                font_scale=0.55,
                thickness=1
            )

            # ========================================================
            # 警报文字和声音
            # ========================================================

            if eye_alarm:

                draw_warning_banner(
                    frame,
                    "WARNING: PROLONGED EYE CLOSURE",
                    frame_height // 2,
                    COLOR_RED
                )

            if yawn_alarm:

                draw_warning_banner(
                    frame,
                    "WARNING: PROLONGED YAWNING",
                    frame_height // 2 + 65,
                    COLOR_ORANGE
                )

            need_alarm = (
                eye_alarm
                or yawn_alarm
            )

            if (
                need_alarm
                and current_time
                - last_alarm_time
                >= ALARM_COOLDOWN_SECONDS
            ):

                start_alarm_thread()

                last_alarm_time = (
                    current_time
                )

            # ========================================================
            # 显示画面
            # ========================================================

            cv2.imshow(
                "Bus Driver Fatigue Detection",
                frame
            )

            pressed_key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if pressed_key in (
                ord("q"),
                ord("Q")
            ):

                break

    except KeyboardInterrupt:

        print(
            "\n检测已由用户终止。"
        )

    finally:

        camera.release()
        face_mesh.close()
        cv2.destroyAllWindows()

    print(
        "实时检测程序已关闭。"
    )


if __name__ == "__main__":
    main()