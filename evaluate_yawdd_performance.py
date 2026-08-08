"""
YawDD Threshold-Based Baseline Evaluation

Method:
    Dlib 68-point facial landmarks
    EAR threshold for eye closure
    MAR threshold for yawning
    Consecutive-frame fatigue decision

Label definition:
    0 = Normal
    1 = Fatigue / Yawn

Important:
    Videos without an explicit label are not automatically treated as Normal.
    They are written to an unresolved-label CSV file instead.
"""

from pathlib import Path

import cv2
import dlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from imutils import face_utils
from scipy.spatial import distance as dist
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)


# ============================================================
# 1. 路径设置
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\DELL\PycharmProjects\pythonProject9"
    r"\BusDriverFatigueDetection"
)

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "archive123"
)

MANUAL_LABEL_PATH = (
    PROJECT_ROOT
    / "data"
    / "threshold_evaluation_labels.csv"
)

PREDICTOR_CANDIDATES = [
    PROJECT_ROOT
    / "models"
    / "shape_predictor_68_face_landmarks.dat",

    PROJECT_ROOT
    / "src"
    / "models"
    / "shape_predictor_68_face_landmarks.dat",

    Path(
        r"C:\Users\DELL\Desktop\pythonProject9"
        r"\shape_predictor_68_face_landmarks.dat"
        r"\shape_predictor_68_face_landmarks.dat"
    )
]

RESULT_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "threshold_baseline_results"
)

PER_VIDEO_RESULT_PATH = (
    RESULT_DIRECTORY
    / "threshold_baseline_video_results.csv"
)

REPORT_PATH = (
    RESULT_DIRECTORY
    / "threshold_baseline_report.txt"
)

CONFUSION_MATRIX_PATH = (
    RESULT_DIRECTORY
    / "threshold_baseline_confusion_matrix.png"
)

UNRESOLVED_LABEL_PATH = (
    RESULT_DIRECTORY
    / "threshold_unresolved_video_labels.csv"
)


# ============================================================
# 2. 判定参数
# ============================================================

EAR_THRESHOLD = 0.24
MAR_THRESHOLD = 0.75

EAR_CONSECUTIVE_FRAMES = 15
MAR_CONSECUTIVE_FRAMES = 15

PROCESSING_FRAME_WIDTH = 450

SUPPORTED_VIDEO_EXTENSIONS = {
    ".avi",
    ".mp4",
    ".mov",
    ".mkv"
}

NORMAL_LABEL = 0
FATIGUE_LABEL = 1

CLASS_NAMES = [
    "Normal",
    "Fatigue"
]


# ============================================================
# 3. EAR和MAR
# ============================================================

def eye_aspect_ratio(eye):
    """计算单只眼睛的EAR。"""

    if eye is None or len(eye) != 6:
        return None

    vertical_distance_1 = dist.euclidean(
        eye[1],
        eye[5]
    )

    vertical_distance_2 = dist.euclidean(
        eye[2],
        eye[4]
    )

    horizontal_distance = dist.euclidean(
        eye[0],
        eye[3]
    )

    if horizontal_distance < 1e-6:
        return None

    return float(
        (
            vertical_distance_1
            + vertical_distance_2
        )
        / (
            2.0
            * horizontal_distance
        )
    )


def mouth_aspect_ratio(mouth):
    """计算Dlib嘴部区域的MAR。"""

    if mouth is None or len(mouth) < 20:
        return None

    vertical_distance_1 = dist.euclidean(
        mouth[13],
        mouth[19]
    )

    vertical_distance_2 = dist.euclidean(
        mouth[14],
        mouth[18]
    )

    vertical_distance_3 = dist.euclidean(
        mouth[15],
        mouth[17]
    )

    horizontal_distance = dist.euclidean(
        mouth[12],
        mouth[16]
    )

    if horizontal_distance < 1e-6:
        return None

    return float(
        (
            vertical_distance_1
            + vertical_distance_2
            + vertical_distance_3
        )
        / (
            2.0
            * horizontal_distance
        )
    )


# ============================================================
# 4. 文件处理
# ============================================================

def find_predictor_path():
    """查找Dlib关键点模型。"""

    for candidate_path in PREDICTOR_CANDIDATES:

        if candidate_path.exists():
            return candidate_path

    return None


def collect_video_files(dataset_path):
    """递归读取数据集中的视频文件。"""

    video_files = [
        path
        for path in dataset_path.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower()
            in SUPPORTED_VIDEO_EXTENSIONS
        )
    ]

    video_files.sort(
        key=lambda path: str(path).lower()
    )

    return video_files


def create_relative_path(
    video_path,
    dataset_root
):
    """生成视频相对路径。"""

    try:

        relative_path = video_path.relative_to(
            dataset_root
        )

    except ValueError:

        relative_path = Path(
            video_path.name
        )

    return str(
        relative_path
    ).replace(
        "\\",
        "/"
    )


def create_video_id(
    video_path,
    dataset_root
):
    """生成唯一视频编号。"""

    relative_path = create_relative_path(
        video_path,
        dataset_root
    )

    return (
        str(
            Path(relative_path)
            .with_suffix("")
        )
        .replace("\\", "__")
        .replace("/", "__")
        .replace(" ", "_")
    )


def normalize_text(text):
    """统一字符串格式。"""

    return (
        str(text)
        .strip()
        .lower()
        .replace("\\", "/")
        .replace("-", "_")
        .replace(" ", "_")
    )


# ============================================================
# 5. 标签处理
# ============================================================

def load_manual_labels(label_file_path):
    """
    读取人工标签表。

    格式：
        RelativePath,Label
        Female/001.avi,0
        Female/002.avi,1
    """

    if not label_file_path.exists():
        return {}

    label_data = pd.read_csv(
        label_file_path
    )

    required_columns = {
        "RelativePath",
        "Label"
    }

    missing_columns = (
        required_columns
        .difference(
            label_data.columns
        )
    )

    if missing_columns:

        raise ValueError(
            "人工标签文件缺少字段："
            f"{sorted(missing_columns)}"
        )

    label_map = {}

    for _, row in label_data.iterrows():

        relative_path = normalize_text(
            row["RelativePath"]
        )

        try:

            label = int(
                row["Label"]
            )

        except (
            TypeError,
            ValueError
        ):

            continue

        if label not in (
            NORMAL_LABEL,
            FATIGUE_LABEL
        ):

            continue

        label_map[
            relative_path
        ] = label

    return label_map


def infer_label_from_path(
    video_path,
    dataset_root
):
    """
    根据明确的文件名或目录关键词判断标签。

    无法明确判断时返回None，
    不再默认标记为Normal。
    """

    relative_path = create_relative_path(
        video_path,
        dataset_root
    )

    path_text = normalize_text(
        relative_path
    )

    no_yawn_keywords = [
        "no_yawn",
        "noyawn",
        "not_yawn",
        "non_yawn",
        "without_yawn"
    ]

    fatigue_keywords = [
        "talking_yawning",
        "talkingyawning",
        "yawning",
        "fatigue",
        "drowsy",
        "sleepy"
    ]

    normal_keywords = [
        "normal",
        "talking",
        "alert",
        "awake"
    ]

    if any(
        keyword in path_text
        for keyword in no_yawn_keywords
    ):

        return NORMAL_LABEL

    if any(
        keyword in path_text
        for keyword in fatigue_keywords
    ):

        return FATIGUE_LABEL

    if any(
        keyword in path_text
        for keyword in normal_keywords
    ):

        return NORMAL_LABEL

    return None


def resolve_video_label(
    video_path,
    dataset_root,
    manual_label_map
):
    """优先使用人工标签，再使用文件名标签。"""

    relative_path = create_relative_path(
        video_path,
        dataset_root
    )

    normalized_path = normalize_text(
        relative_path
    )

    if normalized_path in manual_label_map:

        return (
            manual_label_map[
                normalized_path
            ],
            "Manual"
        )

    inferred_label = infer_label_from_path(
        video_path,
        dataset_root
    )

    if inferred_label is not None:

        return (
            inferred_label,
            "Filename"
        )

    return (
        None,
        "Unresolved"
    )


def save_unresolved_labels(
    unresolved_rows
):
    """保存无法自动判断标签的视频。"""

    if not unresolved_rows:

        if UNRESOLVED_LABEL_PATH.exists():
            UNRESOLVED_LABEL_PATH.unlink()

        return

    unresolved_data = pd.DataFrame(
        unresolved_rows
    )

    unresolved_data[
        "Label"
    ] = ""

    unresolved_data.to_csv(
        UNRESOLVED_LABEL_PATH,
        index=False,
        encoding="utf-8-sig"
    )


# ============================================================
# 6. 视频帧处理
# ============================================================

def resize_frame(
    frame,
    target_width
):
    """保持原比例缩放画面。"""

    height, width = frame.shape[:2]

    if width <= 0 or height <= 0:
        return frame

    resize_ratio = (
        target_width
        / float(width)
    )

    target_height = max(
        1,
        int(
            round(
                height
                * resize_ratio
            )
        )
    )

    return cv2.resize(
        frame,
        (
            target_width,
            target_height
        ),
        interpolation=cv2.INTER_AREA
    )


def select_largest_face(
    face_rectangles
):
    """选择画面中面积最大的人脸。"""

    if len(face_rectangles) == 0:
        return None

    return max(
        face_rectangles,
        key=lambda rectangle: (
            rectangle.width()
            * rectangle.height()
        )
    )


def get_video_timestamp(
    video_capture,
    frame_index,
    video_fps
):
    """获取当前帧在源视频中的时间。"""

    timestamp_ms = video_capture.get(
        cv2.CAP_PROP_POS_MSEC
    )

    if timestamp_ms >= 0:

        return float(
            timestamp_ms / 1000.0
        )

    if video_fps > 0:

        return float(
            frame_index
            / video_fps
        )

    return 0.0


# ============================================================
# 7. 单视频评估
# ============================================================

def evaluate_single_video(
    video_path,
    dataset_root,
    true_label,
    label_source,
    detector,
    predictor,
    landmark_indexes
):
    """评估一个视频。"""

    (
        left_eye_start,
        left_eye_end,
        right_eye_start,
        right_eye_end,
        mouth_start,
        mouth_end
    ) = landmark_indexes

    video_id = create_video_id(
        video_path,
        dataset_root
    )

    relative_path = create_relative_path(
        video_path,
        dataset_root
    )

    video_capture = cv2.VideoCapture(
        str(video_path)
    )

    if not video_capture.isOpened():

        return {
            "VideoID": video_id,
            "RelativePath": relative_path,
            "SourceFile": video_path.name,
            "TrueLabel": true_label,
            "LabelSource": label_source,
            "PredictedLabel": np.nan,
            "Correct": np.nan,
            "Status": "Open Failed"
        }

    video_fps = float(
        video_capture.get(
            cv2.CAP_PROP_FPS
        )
    )

    total_video_frames = int(
        video_capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    frame_index = 0
    valid_face_frames = 0
    no_face_frames = 0
    invalid_feature_frames = 0

    eye_closed_counter = 0
    yawn_counter = 0

    maximum_eye_counter = 0
    maximum_yawn_counter = 0

    minimum_ear = np.inf
    maximum_mar = -np.inf

    eye_alarm_detected = False
    yawn_alarm_detected = False

    first_eye_alarm_frame = np.nan
    first_yawn_alarm_frame = np.nan

    first_eye_alarm_time = np.nan
    first_yawn_alarm_time = np.nan

    while True:

        success, frame = (
            video_capture.read()
        )

        if not success:
            break

        frame_index += 1

        resized_frame = resize_frame(
            frame,
            PROCESSING_FRAME_WIDTH
        )

        gray_frame = cv2.cvtColor(
            resized_frame,
            cv2.COLOR_BGR2GRAY
        )

        face_rectangles = detector(
            gray_frame,
            0
        )

        face_rectangle = select_largest_face(
            face_rectangles
        )

        if face_rectangle is None:

            no_face_frames += 1

            eye_closed_counter = 0
            yawn_counter = 0

            continue

        try:

            shape = predictor(
                gray_frame,
                face_rectangle
            )

            shape = face_utils.shape_to_np(
                shape
            )

        except Exception:

            invalid_feature_frames += 1

            eye_closed_counter = 0
            yawn_counter = 0

            continue

        left_eye = shape[
            left_eye_start:left_eye_end
        ]

        right_eye = shape[
            right_eye_start:right_eye_end
        ]

        mouth = shape[
            mouth_start:mouth_end
        ]

        left_ear = eye_aspect_ratio(
            left_eye
        )

        right_ear = eye_aspect_ratio(
            right_eye
        )

        mar = mouth_aspect_ratio(
            mouth
        )

        if (
            left_ear is None
            or right_ear is None
            or mar is None
        ):

            invalid_feature_frames += 1

            eye_closed_counter = 0
            yawn_counter = 0

            continue

        valid_face_frames += 1

        average_ear = (
            left_ear
            + right_ear
        ) / 2.0

        minimum_ear = min(
            minimum_ear,
            average_ear
        )

        maximum_mar = max(
            maximum_mar,
            mar
        )

        if average_ear < EAR_THRESHOLD:

            eye_closed_counter += 1

        else:

            eye_closed_counter = 0

        if mar > MAR_THRESHOLD:

            yawn_counter += 1

        else:

            yawn_counter = 0

        maximum_eye_counter = max(
            maximum_eye_counter,
            eye_closed_counter
        )

        maximum_yawn_counter = max(
            maximum_yawn_counter,
            yawn_counter
        )

        current_timestamp = get_video_timestamp(
            video_capture,
            frame_index,
            video_fps
        )

        if (
            eye_closed_counter
            >= EAR_CONSECUTIVE_FRAMES
        ):

            if not eye_alarm_detected:

                eye_alarm_detected = True

                first_eye_alarm_frame = (
                    frame_index
                )

                first_eye_alarm_time = (
                    current_timestamp
                )

        if (
            yawn_counter
            >= MAR_CONSECUTIVE_FRAMES
        ):

            if not yawn_alarm_detected:

                yawn_alarm_detected = True

                first_yawn_alarm_frame = (
                    frame_index
                )

                first_yawn_alarm_time = (
                    current_timestamp
                )

    video_capture.release()

    predicted_label = (
        FATIGUE_LABEL
        if (
            eye_alarm_detected
            or yawn_alarm_detected
        )
        else NORMAL_LABEL
    )

    if (
        eye_alarm_detected
        and yawn_alarm_detected
    ):

        trigger_type = (
            "Eye Closure + Yawn"
        )

    elif eye_alarm_detected:

        trigger_type = (
            "Eye Closure"
        )

    elif yawn_alarm_detected:

        trigger_type = (
            "Yawn"
        )

    else:

        trigger_type = (
            "No Alarm"
        )

    alarm_times = [
        value
        for value in [
            first_eye_alarm_time,
            first_yawn_alarm_time
        ]
        if not pd.isna(value)
    ]

    first_alarm_time = (
        min(alarm_times)
        if alarm_times
        else np.nan
    )

    alarm_frames = [
        value
        for value in [
            first_eye_alarm_frame,
            first_yawn_alarm_frame
        ]
        if not pd.isna(value)
    ]

    first_alarm_frame = (
        min(alarm_frames)
        if alarm_frames
        else np.nan
    )

    correct = int(
        predicted_label
        == true_label
    )

    valid_face_rate = (
        valid_face_frames
        / frame_index
        if frame_index > 0
        else 0.0
    )

    if np.isinf(minimum_ear):
        minimum_ear = np.nan

    if np.isinf(maximum_mar):
        maximum_mar = np.nan

    return {
        "VideoID": video_id,
        "RelativePath": relative_path,
        "SourceFile": video_path.name,
        "TrueLabel": true_label,
        "LabelSource": label_source,
        "PredictedLabel": predicted_label,
        "Correct": correct,
        "Status": "Completed",
        "TriggerType": trigger_type,
        "FirstAlarmFrame": first_alarm_frame,
        "FirstAlarmTimeSec": first_alarm_time,
        "FirstEyeAlarmTimeSec":
            first_eye_alarm_time,
        "FirstYawnAlarmTimeSec":
            first_yawn_alarm_time,
        "VideoFPS": video_fps,
        "TotalVideoFrames": total_video_frames,
        "ProcessedFrames": frame_index,
        "ValidFaceFrames": valid_face_frames,
        "NoFaceFrames": no_face_frames,
        "InvalidFeatureFrames":
            invalid_feature_frames,
        "ValidFaceRate": valid_face_rate,
        "MinimumEAR": minimum_ear,
        "MaximumMAR": maximum_mar,
        "MaximumClosedEyeFrames":
            maximum_eye_counter,
        "MaximumYawnFrames":
            maximum_yawn_counter
    }


# ============================================================
# 8. 指标计算
# ============================================================

def safe_divide(
    numerator,
    denominator
):
    """避免除零。"""

    if denominator == 0:
        return 0.0

    return float(
        numerator
        / denominator
    )


def calculate_metrics(
    true_labels,
    predicted_labels
):
    """计算视频级二分类指标。"""

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=[
            NORMAL_LABEL,
            FATIGUE_LABEL
        ]
    )

    true_negative = int(
        matrix[0, 0]
    )

    false_positive = int(
        matrix[0, 1]
    )

    false_negative = int(
        matrix[1, 0]
    )

    true_positive = int(
        matrix[1, 1]
    )

    return {
        "Accuracy": accuracy_score(
            true_labels,
            predicted_labels
        ),

        "Precision": precision_score(
            true_labels,
            predicted_labels,
            pos_label=FATIGUE_LABEL,
            zero_division=0
        ),

        "Recall": recall_score(
            true_labels,
            predicted_labels,
            pos_label=FATIGUE_LABEL,
            zero_division=0
        ),

        "F1Score": f1_score(
            true_labels,
            predicted_labels,
            pos_label=FATIGUE_LABEL,
            zero_division=0
        ),

        "Specificity": safe_divide(
            true_negative,
            true_negative
            + false_positive
        ),

        "FalseAlarmRate": safe_divide(
            false_positive,
            true_negative
            + false_positive
        ),

        "MissRate": safe_divide(
            false_negative,
            true_positive
            + false_negative
        ),

        "TrueNegative": true_negative,
        "FalsePositive": false_positive,
        "FalseNegative": false_negative,
        "TruePositive": true_positive,
        "ConfusionMatrix": matrix
    }


# ============================================================
# 9. 保存结果
# ============================================================

def save_confusion_matrix(
    matrix
):
    """保存混淆矩阵图片。"""

    plt.figure(
        figsize=(6, 5)
    )

    plt.imshow(
        matrix,
        interpolation="nearest"
    )

    plt.title(
        "Threshold Baseline Confusion Matrix"
    )

    plt.colorbar()

    positions = np.arange(
        len(CLASS_NAMES)
    )

    plt.xticks(
        positions,
        CLASS_NAMES
    )

    plt.yticks(
        positions,
        CLASS_NAMES
    )

    plt.xlabel(
        "Predicted Label"
    )

    plt.ylabel(
        "True Label"
    )

    threshold = (
        matrix.max() / 2.0
        if matrix.max() > 0
        else 0
    )

    for row in range(
        matrix.shape[0]
    ):

        for column in range(
            matrix.shape[1]
        ):

            value = matrix[
                row,
                column
            ]

            plt.text(
                column,
                row,
                str(value),
                ha="center",
                va="center",
                color=(
                    "white"
                    if value > threshold
                    else "black"
                )
            )

    plt.tight_layout()

    plt.savefig(
        CONFUSION_MATRIX_PATH,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def save_report(
    evaluated_data,
    discovered_video_count,
    unresolved_video_count,
    failed_video_count,
    metrics,
    classification_text
):
    """保存评估报告。"""

    valid_face_rates = (
        evaluated_data[
            "ValidFaceRate"
        ]
        .dropna()
    )

    average_valid_face_rate = (
        valid_face_rates.mean()
        if len(valid_face_rates) > 0
        else 0.0
    )

    report_content = (
        "YawDD Threshold-Based Baseline Evaluation\n"
        "=========================================\n\n"

        "Parameters\n"
        "----------\n"
        f"EAR threshold: {EAR_THRESHOLD}\n"
        f"EAR consecutive frames: "
        f"{EAR_CONSECUTIVE_FRAMES}\n"
        f"MAR threshold: {MAR_THRESHOLD}\n"
        f"MAR consecutive frames: "
        f"{MAR_CONSECUTIVE_FRAMES}\n\n"

        "Dataset summary\n"
        "---------------\n"
        f"Discovered videos: "
        f"{discovered_video_count}\n"
        f"Evaluated labelled videos: "
        f"{len(evaluated_data)}\n"
        f"Unresolved labels: "
        f"{unresolved_video_count}\n"
        f"Failed videos: "
        f"{failed_video_count}\n"
        f"Normal videos: "
        f"{int((evaluated_data['TrueLabel'] == 0).sum())}\n"
        f"Fatigue videos: "
        f"{int((evaluated_data['TrueLabel'] == 1).sum())}\n"
        f"Average valid-face rate: "
        f"{average_valid_face_rate:.4f}\n\n"

        "Video-level performance\n"
        "-----------------------\n"
        f"Accuracy: "
        f"{metrics['Accuracy']:.4f}\n"
        f"Precision: "
        f"{metrics['Precision']:.4f}\n"
        f"Recall: "
        f"{metrics['Recall']:.4f}\n"
        f"Specificity: "
        f"{metrics['Specificity']:.4f}\n"
        f"F1-score: "
        f"{metrics['F1Score']:.4f}\n"
        f"False alarm rate: "
        f"{metrics['FalseAlarmRate']:.4f}\n"
        f"Miss rate: "
        f"{metrics['MissRate']:.4f}\n\n"

        "Confusion matrix\n"
        "----------------\n"
        f"{metrics['ConfusionMatrix']}\n\n"

        "Classification report\n"
        "---------------------\n"
        f"{classification_text}\n"
    )

    with open(
        REPORT_PATH,
        mode="w",
        encoding="utf-8"
    ) as report_file:

        report_file.write(
            report_content
        )


# ============================================================
# 10. 主程序
# ============================================================

def main():

    print("=" * 75)
    print(
        "YawDD Threshold-Based Baseline Evaluation"
    )
    print("=" * 75)

    RESULT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    if not DATASET_PATH.exists():

        print(
            "\n[ERROR] 找不到数据集目录："
        )

        print(
            DATASET_PATH
        )

        return

    predictor_path = find_predictor_path()

    if predictor_path is None:

        print(
            "\n[ERROR] 找不到Dlib关键点模型。"
        )

        return

    try:

        manual_label_map = (
            load_manual_labels(
                MANUAL_LABEL_PATH
            )
        )

    except ValueError as error:

        print(
            f"\n[ERROR] {error}"
        )

        return

    video_files = collect_video_files(
        DATASET_PATH
    )

    if not video_files:

        print(
            "\n[ERROR] 没有找到视频。"
        )

        return

    labelled_videos = []
    unresolved_rows = []

    for video_path in video_files:

        label, label_source = (
            resolve_video_label(
                video_path,
                DATASET_PATH,
                manual_label_map
            )
        )

        if label is None:

            unresolved_rows.append(
                {
                    "RelativePath":
                        create_relative_path(
                            video_path,
                            DATASET_PATH
                        ),

                    "SourceFile":
                        video_path.name
                }
            )

        else:

            labelled_videos.append(
                (
                    video_path,
                    label,
                    label_source
                )
            )

    save_unresolved_labels(
        unresolved_rows
    )

    normal_count = sum(
        label == NORMAL_LABEL
        for _, label, _ in labelled_videos
    )

    fatigue_count = sum(
        label == FATIGUE_LABEL
        for _, label, _ in labelled_videos
    )

    print("\n数据集统计")
    print("-" * 75)

    print(
        f"发现视频："
        f"{len(video_files)}"
    )

    print(
        f"有效标签视频："
        f"{len(labelled_videos)}"
    )

    print(
        f"Normal视频："
        f"{normal_count}"
    )

    print(
        f"Fatigue视频："
        f"{fatigue_count}"
    )

    print(
        f"标签不明确视频："
        f"{len(unresolved_rows)}"
    )

    print(
        f"EAR判定：EAR < {EAR_THRESHOLD}，"
        f"连续{EAR_CONSECUTIVE_FRAMES}帧"
    )

    print(
        f"MAR判定：MAR > {MAR_THRESHOLD}，"
        f"连续{MAR_CONSECUTIVE_FRAMES}帧"
    )

    if not labelled_videos:

        print(
            "\n[ERROR] 没有可用于评估的标签视频。"
        )

        return

    try:

        detector = (
            dlib.get_frontal_face_detector()
        )

        predictor = dlib.shape_predictor(
            str(predictor_path)
        )

    except Exception as error:

        print(
            f"\n[ERROR] Dlib初始化失败：{error}"
        )

        return

    (
        left_eye_start,
        left_eye_end
    ) = face_utils.FACIAL_LANDMARKS_IDXS[
        "left_eye"
    ]

    (
        right_eye_start,
        right_eye_end
    ) = face_utils.FACIAL_LANDMARKS_IDXS[
        "right_eye"
    ]

    (
        mouth_start,
        mouth_end
    ) = face_utils.FACIAL_LANDMARKS_IDXS[
        "mouth"
    ]

    landmark_indexes = (
        left_eye_start,
        left_eye_end,
        right_eye_start,
        right_eye_end,
        mouth_start,
        mouth_end
    )

    video_results = []

    for (
        video_number,
        (
            video_path,
            true_label,
            label_source
        )
    ) in enumerate(
        labelled_videos,
        start=1
    ):

        print(
            f"\n[{video_number}/"
            f"{len(labelled_videos)}] "
            f"{video_path.name}"
        )

        print(
            f"真实标签："
            f"{CLASS_NAMES[true_label]} "
            f"({label_source})"
        )

        result = evaluate_single_video(
            video_path=video_path,
            dataset_root=DATASET_PATH,
            true_label=true_label,
            label_source=label_source,
            detector=detector,
            predictor=predictor,
            landmark_indexes=landmark_indexes
        )

        video_results.append(
            result
        )

        if result["Status"] != "Completed":

            print(
                f"处理失败："
                f"{result['Status']}"
            )

            continue

        predicted_label = int(
            result["PredictedLabel"]
        )

        result_text = (
            "正确"
            if result["Correct"] == 1
            else "错误"
        )

        print(
            f"预测标签："
            f"{CLASS_NAMES[predicted_label]} "
            f"| {result_text}"
        )

        if predicted_label == FATIGUE_LABEL:

            print(
                f"触发类型："
                f"{result['TriggerType']} "
                f"| 首次报警时间："
                f"{result['FirstAlarmTimeSec']:.3f}s"
            )

    result_data = pd.DataFrame(
        video_results
    )

    result_data.to_csv(
        PER_VIDEO_RESULT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    completed_data = result_data[
        result_data["Status"]
        == "Completed"
    ].copy()

    failed_video_count = (
        len(result_data)
        - len(completed_data)
    )

    if completed_data.empty:

        print(
            "\n[ERROR] 没有完成评估的视频。"
        )

        return

    true_labels = (
        completed_data["TrueLabel"]
        .to_numpy(
            dtype=int
        )
    )

    predicted_labels = (
        completed_data["PredictedLabel"]
        .to_numpy(
            dtype=int
        )
    )

    metrics = calculate_metrics(
        true_labels,
        predicted_labels
    )

    classification_text = (
        classification_report(
            true_labels,
            predicted_labels,
            labels=[
                NORMAL_LABEL,
                FATIGUE_LABEL
            ],
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0
        )
    )

    save_confusion_matrix(
        metrics[
            "ConfusionMatrix"
        ]
    )

    save_report(
        evaluated_data=completed_data,
        discovered_video_count=len(
            video_files
        ),
        unresolved_video_count=len(
            unresolved_rows
        ),
        failed_video_count=failed_video_count,
        metrics=metrics,
        classification_text=classification_text
    )

    print("\n" + "=" * 75)
    print("Video-Level Evaluation Results")
    print("=" * 75)

    print(
        f"有效测试视频："
        f"{len(completed_data)}"
    )

    print(
        f"未使用的标签不明确视频："
        f"{len(unresolved_rows)}"
    )

    print(
        f"Accuracy："
        f"{metrics['Accuracy'] * 100:.2f}%"
    )

    print(
        f"Precision："
        f"{metrics['Precision']:.4f}"
    )

    print(
        f"Recall："
        f"{metrics['Recall']:.4f}"
    )

    print(
        f"Specificity："
        f"{metrics['Specificity']:.4f}"
    )

    print(
        f"F1-score："
        f"{metrics['F1Score']:.4f}"
    )

    print(
        f"False Alarm Rate："
        f"{metrics['FalseAlarmRate']:.4f}"
    )

    print(
        f"Miss Rate："
        f"{metrics['MissRate']:.4f}"
    )

    print("\nConfusion Matrix")

    print(
        metrics[
            "ConfusionMatrix"
        ]
    )

    print("\n结果保存位置")
    print("-" * 75)

    print(
        f"逐视频结果："
        f"{PER_VIDEO_RESULT_PATH}"
    )

    print(
        f"混淆矩阵："
        f"{CONFUSION_MATRIX_PATH}"
    )

    print(
        f"评估报告："
        f"{REPORT_PATH}"
    )

    if unresolved_rows:

        print(
            f"待人工标注视频："
            f"{UNRESOLVED_LABEL_PATH}"
        )


if __name__ == "__main__":
    main()