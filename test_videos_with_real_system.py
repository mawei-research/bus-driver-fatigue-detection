"""
Offline video tester and screenshot capture tool for the final dual 2D-CNN fatigue detection system.

Eye model:
    0 = closed eyes
    1 = open eyes

Mouth model:
    0 = normal / no yawn
    1 = yawn

Offline duration is calculated from the source video's own timestamp rather
than the computer's processing time.
"""

import os
import re
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf


# ============================================================
# 1. 路径设置
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\DELL\PycharmProjects\pythonProject9"
    r"\BusDriverFatigueDetection"
)

VIDEO_DIR = (
    PROJECT_ROOT
    / "data"
    / "archive123"
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

SAVE_DIR = Path(
    r"C:\Users\DELL\Desktop\Thesis_Screenshots"
)


# ============================================================
# 2. 检测参数
# ============================================================

IMAGE_SIZE = 80

# 眼睛模型输出小于0.5时判断为闭眼
EYE_THRESHOLD = 0.50

# 嘴巴模型输出大于等于0.5时判断为哈欠
MOUTH_THRESHOLD = 0.50

# 闭眼或哈欠持续2秒触发警告
ALARM_SECONDS = 2.0

# 与最终评价系统保持一致：每隔2帧执行一次完整检测
# 时间仍按照源视频时间戳计算
FRAME_STEP = 2

# 嘴部宽高比大于该值时优先判断为笑容
SMILE_RATIO_THRESHOLD = 1.50

# ROI区域扩张比例
EYE_EXPANSION = 0.25
MOUTH_EXPANSION = 0.25

# 显示画面宽度
DISPLAY_WIDTH = 960

# 每帧显示等待时间
PLAYBACK_DELAY_MS = 10

VIDEO_EXTENSIONS = {
    ".avi",
    ".mp4",
    ".mov",
    ".mkv"
}


# ============================================================
# 3. MediaPipe关键点
# ============================================================

LEFT_EYE = [
    33,
    133,
    159,
    145,
    160,
    144,
    158,
    153
]

RIGHT_EYE = [
    362,
    263,
    386,
    374,
    385,
    380,
    387,
    373
]

MOUTH = [
    61,
    291,
    0,
    17,
    13,
    14,
    78,
    308,
    82,
    312,
    87,
    317
]

# 嘴部宽高比使用的关键点
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
MOUTH_TOP = 0
MOUTH_BOTTOM = 17


# ============================================================
# 4. OpenCV颜色
# ============================================================

GREEN = (
    0,
    255,
    0
)

RED = (
    0,
    0,
    255
)

ORANGE = (
    0,
    165,
    255
)

YELLOW = (
    0,
    255,
    255
)

PINK = (
    255,
    105,
    180
)

WHITE = (
    255,
    255,
    255
)

BLACK = (
    0,
    0,
    0
)

CYAN = (
    255,
    255,
    0
)


# ============================================================
# 5. 模型和视频工具
# ============================================================

def load_cnn_model(
    model_path,
    model_name
):
    """加载训练完成的Keras模型。"""

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


def collect_videos(
    video_directory
):
    """递归查找目录中的所有视频文件。"""

    video_files = [
        path
        for path in video_directory.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower()
            in VIDEO_EXTENSIONS
        )
    ]

    video_files.sort(
        key=lambda path: str(path).lower()
    )

    return video_files


def resize_frame(
    frame,
    target_width
):
    """保持原始宽高比缩放视频画面。"""

    height, width = frame.shape[:2]

    if (
        width <= 0
        or height <= 0
        or width == target_width
    ):
        return frame

    scale = (
        target_width
        / float(width)
    )

    target_height = max(
        1,
        int(
            round(
                height * scale
            )
        )
    )

    interpolation = (
        cv2.INTER_AREA
        if scale < 1.0
        else cv2.INTER_LINEAR
    )

    return cv2.resize(
        frame,
        (
            target_width,
            target_height
        ),
        interpolation=interpolation
    )


def get_video_time(
    video_capture,
    frame_index,
    video_fps
):
    """
    获取当前帧在原始视频中的时间。

    优先使用视频时间戳。
    时间戳不可用时，使用帧编号除以FPS。
    """

    position_ms = video_capture.get(
        cv2.CAP_PROP_POS_MSEC
    )

    if position_ms > 0:

        return float(
            position_ms / 1000.0
        )

    if video_fps > 0:

        return float(
            max(
                0,
                frame_index - 1
            )
            / video_fps
        )

    return 0.0


# ============================================================
# 6. ROI区域处理
# ============================================================

def get_square_box(
    landmarks,
    indices,
    frame_width,
    frame_height,
    expansion_ratio
):
    """
    根据关键点生成与最终评价程序一致的正方形ROI。

    expansion_ratio=0.25表示在ROI四周各增加25%的边界，
    因此最终边长为原始较长边的1.50倍。
    """

    points = []

    landmark_count = len(
        landmarks.landmark
    )

    for index in indices:

        if index >= landmark_count:
            continue

        landmark = landmarks.landmark[
            index
        ]

        points.append(
            (
                landmark.x * frame_width,
                landmark.y * frame_height
            )
        )

    if not points:
        return None

    point_array = np.asarray(
        points,
        dtype=np.float32
    )

    minimum_x = float(
        np.min(
            point_array[:, 0]
        )
    )

    maximum_x = float(
        np.max(
            point_array[:, 0]
        )
    )

    minimum_y = float(
        np.min(
            point_array[:, 1]
        )
    )

    maximum_y = float(
        np.max(
            point_array[:, 1]
        )
    )

    width = maximum_x - minimum_x
    height = maximum_y - minimum_y

    base_side = max(
        width,
        height,
        4.0
    )

    side_length = base_side * (
        1.0
        + 2.0 * expansion_ratio
    )

    center_x = (
        minimum_x
        + maximum_x
    ) / 2.0

    center_y = (
        minimum_y
        + maximum_y
    ) / 2.0

    x1 = int(
        round(
            center_x
            - side_length / 2.0
        )
    )

    y1 = int(
        round(
            center_y
            - side_length / 2.0
        )
    )

    x2 = int(
        round(
            center_x
            + side_length / 2.0
        )
    )

    y2 = int(
        round(
            center_y
            + side_length / 2.0
        )
    )

    x1 = max(
        0,
        x1
    )

    y1 = max(
        0,
        y1
    )

    x2 = min(
        frame_width,
        x2
    )

    y2 = min(
        frame_height,
        y2
    )

    if (
        x2 - x1 < 4
        or y2 - y1 < 4
    ):
        return None

    return (
        x1,
        y1,
        x2,
        y2
    )


def preprocess_roi(
    frame,
    bounding_box
):
    """
    裁剪ROI并转换为80×80灰度图。

    返回形状：
        80 × 80 × 1
    """

    if bounding_box is None:
        return None

    (
        x1,
        y1,
        x2,
        y2
    ) = bounding_box

    roi = frame[
        y1:y2,
        x1:x2
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
                IMAGE_SIZE,
                IMAGE_SIZE
            ),
            interpolation=cv2.INTER_AREA
        )

    except cv2.error:

        return None

    resized_roi = resized_roi.astype(
        np.float32
    )

    return np.expand_dims(
        resized_roi,
        axis=-1
    )


def get_mouth_ratio(
    landmarks,
    frame_width,
    frame_height
):
    """
    计算嘴巴宽高比，与最终评价程序保持一致。

    宽度：关键点61至291的欧氏距离
    高度：关键点0至17的欧氏距离
    """

    required_indices = [
        MOUTH_LEFT,
        MOUTH_RIGHT,
        MOUTH_TOP,
        MOUTH_BOTTOM
    ]

    if any(
        index
        >= len(
            landmarks.landmark
        )
        for index
        in required_indices
    ):
        return None

    def landmark_to_pixel(index):

        landmark = landmarks.landmark[
            index
        ]

        return np.asarray(
            [
                landmark.x * frame_width,
                landmark.y * frame_height
            ],
            dtype=np.float32
        )

    left_point = landmark_to_pixel(
        MOUTH_LEFT
    )

    right_point = landmark_to_pixel(
        MOUTH_RIGHT
    )

    top_point = landmark_to_pixel(
        MOUTH_TOP
    )

    bottom_point = landmark_to_pixel(
        MOUTH_BOTTOM
    )

    mouth_width = float(
        np.linalg.norm(
            right_point
            - left_point
        )
    )

    mouth_height = float(
        np.linalg.norm(
            bottom_point
            - top_point
        )
    )

    if mouth_height < 1e-6:
        return None

    return float(
        mouth_width
        / mouth_height
    )


# ============================================================
# 7. 模型预测
# ============================================================

def predict_eye_states(
    eye_model,
    left_eye_image,
    right_eye_image
):
    """在同一个批次中预测左右眼状态。"""

    if (
        left_eye_image is None
        or right_eye_image is None
    ):

        return (
            None,
            None
        )

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
    ).reshape(
        -1
    )

    if predictions.size < 2:

        return (
            None,
            None
        )

    return (
        float(
            predictions[0]
        ),
        float(
            predictions[1]
        )
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
    ).reshape(
        -1
    )

    if prediction.size == 0:
        return None

    return float(
        prediction[0]
    )


# ============================================================
# 8. 绘图工具
# ============================================================

def draw_connections(
    frame,
    landmarks,
    connections,
    frame_width,
    frame_height,
    color
):
    """绘制MediaPipe关键点连线。"""

    landmark_count = len(
        landmarks.landmark
    )

    for (
        start_index,
        end_index
    ) in connections:

        if (
            start_index
            >= landmark_count
            or end_index
            >= landmark_count
        ):

            continue

        start_point = landmarks.landmark[
            start_index
        ]

        end_point = landmarks.landmark[
            end_index
        ]

        start_xy = (
            int(
                start_point.x
                * frame_width
            ),
            int(
                start_point.y
                * frame_height
            )
        )

        end_xy = (
            int(
                end_point.x
                * frame_width
            ),
            int(
                end_point.y
                * frame_height
            )
        )

        cv2.line(
            frame,
            start_xy,
            end_xy,
            color,
            1
        )


def draw_box(
    frame,
    bounding_box,
    color
):
    """绘制ROI边界框。"""

    if bounding_box is None:
        return

    (
        x1,
        y1,
        x2,
        y2
    ) = bounding_box

    cv2.rectangle(
        frame,
        (
            x1,
            y1
        ),
        (
            x2,
            y2
        ),
        color,
        1
    )


def draw_text(
    frame,
    text,
    position,
    color,
    font_scale=0.58,
    thickness=1
):
    """绘制带黑色背景的文字。"""

    font = (
        cv2.FONT_HERSHEY_SIMPLEX
    )

    (
        text_width,
        text_height
    ), baseline = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness
    )

    x_position, y_position = (
        position
    )

    cv2.rectangle(
        frame,
        (
            x_position - 4,
            y_position
            - text_height
            - 5
        ),
        (
            x_position
            + text_width
            + 4,
            y_position
            + baseline
            + 4
        ),
        BLACK,
        -1
    )

    cv2.putText(
        frame,
        text,
        position,
        font,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA
    )


def draw_warning(
    frame,
    text,
    vertical_position,
    color
):
    """在画面中间显示报警文字。"""

    (
        frame_height,
        frame_width
    ) = frame.shape[:2]

    font = (
        cv2.FONT_HERSHEY_SIMPLEX
    )

    font_scale = 1.0
    thickness = 3

    (
        text_width,
        text_height
    ), baseline = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness
    )

    x_position = max(
        10,
        (
            frame_width
            - text_width
        ) // 2
    )

    y_position = min(
        frame_height - 10,
        vertical_position
    )

    cv2.rectangle(
        frame,
        (
            x_position - 10,
            y_position
            - text_height
            - 10
        ),
        (
            x_position
            + text_width
            + 10,
            y_position
            + baseline
            + 10
        ),
        BLACK,
        -1
    )

    cv2.putText(
        frame,
        text,
        (
            x_position,
            y_position
        ),
        font,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA
    )


# ============================================================
# 9. 截图
# ============================================================

def save_screenshot(
    frame,
    video_path,
    video_time,
    screenshot_number
):
    """保存当前带HUD的视频画面。"""

    video_name = re.sub(
        r'[<>:"/\\|?*]+',
        "_",
        video_path.stem
    ).strip(
        " ._"
    )

    if not video_name:
        video_name = "video"

    time_text = (
        f"{video_time:.2f}"
        .replace(
            ".",
            "_"
        )
    )

    file_name = (
        f"{screenshot_number:03d}_"
        f"{video_name}_"
        f"{time_text}s.jpg"
    )

    output_path = (
        SAVE_DIR
        / file_name
    )

    success = cv2.imwrite(
        str(output_path),
        frame
    )

    if success:

        print(
            f"[Screenshot] {output_path}"
        )

    else:

        print(
            f"[WARNING] 截图保存失败："
            f"{output_path}"
        )


# ============================================================
# 10. 主程序
# ============================================================

def main():

    print("=" * 72)
    print(
        "Offline Video Test for "
        "the 2D-CNN Fatigue Detection System"
    )
    print("=" * 72)

    SAVE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if not VIDEO_DIR.exists():

        print(
            "\n[ERROR] 视频目录不存在："
        )

        print(
            VIDEO_DIR
        )

        return

    video_paths = collect_videos(
        VIDEO_DIR
    )

    if not video_paths:

        print(
            "\n[ERROR] 视频目录中没有找到可用视频："
        )

        print(
            VIDEO_DIR
        )

        return

    try:

        print(
            "\n正在加载眼睛状态模型……"
        )

        eye_model = load_cnn_model(
            EYE_MODEL_PATH,
            "眼睛状态"
        )

        print(
            "正在加载嘴巴状态模型……"
        )

        mouth_model = load_cnn_model(
            MOUTH_MODEL_PATH,
            "嘴巴状态"
        )

    except (
        FileNotFoundError,
        RuntimeError
    ) as error:

        print(
            f"\n[ERROR] {error}"
        )

        return

    face_mesh_module = (
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

    face_mesh = (
        face_mesh_module.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.50,
            min_tracking_confidence=0.50
        )
    )

    print(
        f"\n共找到 {len(video_paths)} 个视频。"
    )

    print(
        "快捷键：S截图，D下一个视频，"
        "空格暂停，Q退出。"
    )

    print(
        f"最终系统参数：FrameStep={FRAME_STEP}，"
        f"报警阈值={ALARM_SECONDS:.2f}s，"
        f"几何阈值={SMILE_RATIO_THRESHOLD:.2f}"
    )

    screenshot_count = 0
    quit_program = False

    cv2.namedWindow(
        "Offline Fatigue Test",
        cv2.WINDOW_NORMAL
    )

    try:

        for (
            video_number,
            video_path
        ) in enumerate(
            video_paths,
            start=1
        ):

            print(
                f"\n[{video_number}/"
                f"{len(video_paths)}] "
                f"{video_path}"
            )

            video_capture = (
                cv2.VideoCapture(
                    str(video_path)
                )
            )

            if not video_capture.isOpened():

                print(
                    "[WARNING] 视频无法打开，"
                    "已跳过。"
                )

                continue

            video_fps = float(
                video_capture.get(
                    cv2.CAP_PROP_FPS
                )
            )

            total_frames = int(
                video_capture.get(
                    cv2.CAP_PROP_FRAME_COUNT
                )
            )

            frame_index = 0

            eye_closed_start_time = None
            yawn_start_time = None

            eye_alarm_triggered = False
            yawn_alarm_triggered = False

            maximum_eye_duration = 0.0
            maximum_yawn_duration = 0.0

            skip_video = False

            while video_capture.isOpened():

                (
                    success,
                    original_frame
                ) = video_capture.read()

                if not success:
                    break

                frame_index += 1

                frame_number = (
                    frame_index - 1
                )

                # 与最终评价程序一致，仅处理选中的帧
                if (
                    frame_number
                    % FRAME_STEP
                    != 0
                ):
                    continue

                video_time = get_video_time(
                    video_capture,
                    frame_index,
                    video_fps
                )

                # 检测、ROI裁剪和CNN推理均在原始视频帧上执行
                frame = original_frame.copy()

                (
                    frame_height,
                    frame_width
                ) = frame.shape[:2]

                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                rgb_frame.flags.writeable = (
                    False
                )

                detection = face_mesh.process(
                    rgb_frame
                )

                rgb_frame.flags.writeable = (
                    True
                )

                face_detected = bool(
                    detection.multi_face_landmarks
                )

                eye_status = (
                    "Eyes: Unknown"
                )

                mouth_status = (
                    "Mouth: Unknown"
                )

                eye_color = YELLOW
                mouth_color = YELLOW

                left_eye_probability = None
                right_eye_probability = None
                mouth_probability = None
                mouth_ratio = None

                closed_duration = 0.0
                yawn_duration = 0.0

                eye_alarm_now = False
                yawn_alarm_now = False

                if face_detected:

                    face_landmarks = (
                        detection
                        .multi_face_landmarks[0]
                    )

                    # ================================================
                    # 眼睛检测
                    # ================================================

                    left_eye_box = get_square_box(
                        face_landmarks,
                        LEFT_EYE,
                        frame_width,
                        frame_height,
                        EYE_EXPANSION
                    )

                    right_eye_box = get_square_box(
                        face_landmarks,
                        RIGHT_EYE,
                        frame_width,
                        frame_height,
                        EYE_EXPANSION
                    )

                    left_eye_image = (
                        preprocess_roi(
                            frame,
                            left_eye_box
                        )
                    )

                    right_eye_image = (
                        preprocess_roi(
                            frame,
                            right_eye_box
                        )
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
                        left_eye_probability
                        is not None
                        and right_eye_probability
                        is not None
                    ):

                        both_eyes_closed = (
                            left_eye_probability
                            < EYE_THRESHOLD
                            and right_eye_probability
                            < EYE_THRESHOLD
                        )

                        if both_eyes_closed:

                            eye_status = (
                                "Eyes: CLOSED"
                            )

                            eye_color = RED

                            if (
                                eye_closed_start_time
                                is None
                            ):

                                eye_closed_start_time = (
                                    video_time
                                )

                            closed_duration = max(
                                0.0,
                                video_time
                                - eye_closed_start_time
                            )

                            maximum_eye_duration = max(
                                maximum_eye_duration,
                                closed_duration
                            )

                        else:

                            eye_status = (
                                "Eyes: Open"
                            )

                            eye_color = GREEN

                            eye_closed_start_time = (
                                None
                            )

                    else:

                        eye_status = (
                            "Eyes: ROI Invalid"
                        )

                        eye_closed_start_time = (
                            None
                        )

                    draw_connections(
                        frame,
                        face_landmarks,
                        left_eye_connections,
                        frame_width,
                        frame_height,
                        eye_color
                    )

                    draw_connections(
                        frame,
                        face_landmarks,
                        right_eye_connections,
                        frame_width,
                        frame_height,
                        eye_color
                    )

                    draw_box(
                        frame,
                        left_eye_box,
                        eye_color
                    )

                    draw_box(
                        frame,
                        right_eye_box,
                        eye_color
                    )

                    # ================================================
                    # 嘴巴检测
                    # ================================================

                    mouth_box = get_square_box(
                        face_landmarks,
                        MOUTH,
                        frame_width,
                        frame_height,
                        MOUTH_EXPANSION
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

                    mouth_ratio = get_mouth_ratio(
                        face_landmarks,
                        frame_width,
                        frame_height
                    )

                    if mouth_probability is not None:

                        predicted_as_yawn = (
                            mouth_probability
                            >= MOUTH_THRESHOLD
                        )

                        if predicted_as_yawn:

                            likely_smile = (
                                mouth_ratio
                                is not None
                                and mouth_ratio
                                > SMILE_RATIO_THRESHOLD
                            )

                            if likely_smile:

                                mouth_status = (
                                    "Mouth: "
                                    "Smile - Ignored"
                                )

                                mouth_color = PINK

                                yawn_start_time = (
                                    None
                                )

                            else:

                                mouth_status = (
                                    "Mouth: YAWNING"
                                )

                                mouth_color = ORANGE

                                if (
                                    yawn_start_time
                                    is None
                                ):

                                    yawn_start_time = (
                                        video_time
                                    )

                                yawn_duration = max(
                                    0.0,
                                    video_time
                                    - yawn_start_time
                                )

                                maximum_yawn_duration = max(
                                    maximum_yawn_duration,
                                    yawn_duration
                                )

                        else:

                            mouth_status = (
                                "Mouth: Normal"
                            )

                            mouth_color = CYAN

                            yawn_start_time = (
                                None
                            )

                    else:

                        mouth_status = (
                            "Mouth: ROI Invalid"
                        )

                        yawn_start_time = None

                    draw_connections(
                        frame,
                        face_landmarks,
                        mouth_connections,
                        frame_width,
                        frame_height,
                        mouth_color
                    )

                    draw_box(
                        frame,
                        mouth_box,
                        mouth_color
                    )

                    eye_alarm_now = (
                        closed_duration
                        >= ALARM_SECONDS
                    )

                    yawn_alarm_now = (
                        yawn_duration
                        >= ALARM_SECONDS
                    )

                else:

                    # 人脸消失后不允许继续累计时间
                    eye_closed_start_time = None
                    yawn_start_time = None

                eye_alarm_triggered = (
                    eye_alarm_triggered
                    or eye_alarm_now
                )

                yawn_alarm_triggered = (
                    yawn_alarm_triggered
                    or yawn_alarm_now
                )

                # ================================================
                # HUD显示
                # ================================================

                draw_text(
                    frame,
                    (
                        "Face: Detected"
                        if face_detected
                        else "Face: Not Detected"
                    ),
                    (
                        20,
                        35
                    ),
                    (
                        GREEN
                        if face_detected
                        else YELLOW
                    ),
                    0.62,
                    2
                )

                draw_text(
                    frame,
                    eye_status,
                    (
                        20,
                        70
                    ),
                    eye_color,
                    0.62,
                    2
                )

                draw_text(
                    frame,
                    mouth_status,
                    (
                        20,
                        105
                    ),
                    mouth_color,
                    0.62,
                    2
                )

                if (
                    left_eye_probability
                    is not None
                    and right_eye_probability
                    is not None
                ):

                    eye_probability_text = (
                        f"Eye Open Prob. "
                        f"L={left_eye_probability:.3f} "
                        f"R={right_eye_probability:.3f}"
                    )

                else:

                    eye_probability_text = (
                        "Eye Open Prob. "
                        "L=-- R=--"
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

                if mouth_ratio is not None:

                    mouth_ratio_text = (
                        f"Mouth Ratio: "
                        f"{mouth_ratio:.3f}"
                    )

                else:

                    mouth_ratio_text = (
                        "Mouth Ratio: --"
                    )

                draw_text(
                    frame,
                    eye_probability_text,
                    (
                        20,
                        138
                    ),
                    WHITE
                )

                draw_text(
                    frame,
                    mouth_probability_text,
                    (
                        20,
                        168
                    ),
                    WHITE
                )

                draw_text(
                    frame,
                    mouth_ratio_text,
                    (
                        20,
                        198
                    ),
                    WHITE
                )

                draw_text(
                    frame,
                    (
                        f"Eye Closed Time: "
                        f"{closed_duration:.2f}s"
                    ),
                    (
                        20,
                        228
                    ),
                    (
                        RED
                        if closed_duration > 0
                        else WHITE
                    )
                )

                draw_text(
                    frame,
                    (
                        f"Yawn Time: "
                        f"{yawn_duration:.2f}s"
                    ),
                    (
                        20,
                        258
                    ),
                    (
                        ORANGE
                        if yawn_duration > 0
                        else WHITE
                    )
                )

                draw_text(
                    frame,
                    (
                        f"Video Time: "
                        f"{video_time:.2f}s"
                    ),
                    (
                        20,
                        288
                    ),
                    WHITE
                )

                draw_text(
                    frame,
                    (
                        f"Frame: "
                        f"{frame_index}/"
                        f"{total_frames}"
                    ),
                    (
                        20,
                        318
                    ),
                    WHITE
                )

                if eye_alarm_now:

                    draw_warning(
                        frame,
                        (
                            "WARNING: PROLONGED "
                            "EYE CLOSURE"
                        ),
                        frame_height // 2,
                        RED
                    )

                if yawn_alarm_now:

                    draw_warning(
                        frame,
                        (
                            "WARNING: "
                            "PROLONGED YAWNING"
                        ),
                        frame_height // 2 + 60,
                        ORANGE
                    )

                cv2.imshow(
                    "Offline Fatigue Test",
                    frame
                )

                pressed_key = (
                    cv2.waitKey(
                        PLAYBACK_DELAY_MS
                    )
                    & 0xFF
                )

                if pressed_key in (
                    ord("s"),
                    ord("S")
                ):

                    screenshot_count += 1

                    save_screenshot(
                        frame,
                        video_path,
                        video_time,
                        screenshot_count
                    )

                elif pressed_key in (
                    ord("d"),
                    ord("D")
                ):

                    skip_video = True

                    print(
                        "[INFO] 已跳过当前视频。"
                    )

                    break

                elif pressed_key in (
                    ord("q"),
                    ord("Q")
                ):

                    quit_program = True

                    print(
                        "[INFO] 正在退出测试程序。"
                    )

                    break

                elif pressed_key == 32:

                    while True:

                        paused_frame = (
                            frame.copy()
                        )

                        draw_text(
                            paused_frame,
                            "PAUSED",
                            (
                                frame_width - 120,
                                35
                            ),
                            YELLOW,
                            0.70,
                            2
                        )

                        cv2.imshow(
                            "Offline Fatigue Test",
                            paused_frame
                        )

                        pause_key = (
                            cv2.waitKey(30)
                            & 0xFF
                        )

                        if pause_key == 32:
                            break

                        if pause_key in (
                            ord("s"),
                            ord("S")
                        ):

                            screenshot_count += 1

                            save_screenshot(
                                paused_frame,
                                video_path,
                                video_time,
                                screenshot_count
                            )

                        elif pause_key in (
                            ord("d"),
                            ord("D")
                        ):

                            skip_video = True
                            break

                        elif pause_key in (
                            ord("q"),
                            ord("Q")
                        ):

                            quit_program = True
                            break

                    if (
                        skip_video
                        or quit_program
                    ):
                        break

            video_capture.release()

            print(
                f"闭眼报警是否触发："
                f"{eye_alarm_triggered}"
            )

            print(
                f"哈欠报警是否触发："
                f"{yawn_alarm_triggered}"
            )

            print(
                f"最大连续闭眼时间："
                f"{maximum_eye_duration:.3f}s"
            )

            print(
                f"最大连续哈欠时间："
                f"{maximum_yawn_duration:.3f}s"
            )

            if quit_program:
                break

    except KeyboardInterrupt:

        print(
            "\n[INFO] 测试已手动终止。"
        )

    finally:

        face_mesh.close()

        cv2.destroyAllWindows()

    print(
        "\n全部视频测试结束。"
    )

    print(
        f"截图保存目录：{SAVE_DIR}"
    )


if __name__ == "__main__":
    main()