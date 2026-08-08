"""
Facial Feature Extraction

功能：
    1. 使用 Dlib 68点面部关键点检测器定位眼睛和嘴巴；
    2. 从视频中提取 EAR 和 MAR；
    3. 保存视频编号、帧编号和视频时间戳；
    4. 为后续传统机器学习模型提供基础特征。

标签定义：
    0 = Normal / No Yawn
    1 = Fatigue / Yawn
"""

import csv
import os
from pathlib import Path

import cv2
import dlib
import numpy as np

from imutils import face_utils
from scipy.spatial import distance as dist


# ============================================================
# 1. 路径设置
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\DELL\PycharmProjects\pythonProject9"
    r"\BusDriverFatigueDetection"
)

YAWDD_PATH = PROJECT_ROOT / "data" / "archive123"

OUTPUT_CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "fatigue_features.csv"
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


# ============================================================
# 2. 提取参数
# ============================================================

# 每隔多少帧提取一次特征
FRAME_SAMPLE_INTERVAL = 5

# 处理时统一缩放后的画面宽度
PROCESSING_FRAME_WIDTH = 450

SUPPORTED_VIDEO_EXTENSIONS = {
    ".avi",
    ".mp4",
    ".mov",
    ".mkv"
}

NORMAL_LABEL = 0
FATIGUE_LABEL = 1


# ============================================================
# 3. EAR和MAR计算
# ============================================================

def eye_aspect_ratio(eye):
    """
    计算单只眼睛的 Eye Aspect Ratio。

    参数：
        eye: 由6个眼睛关键点组成的数组

    返回：
        EAR浮点值
    """

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

    ear = (
        vertical_distance_1
        + vertical_distance_2
    ) / (
        2.0 * horizontal_distance
    )

    return float(ear)


def mouth_aspect_ratio(mouth):
    """
    计算 Mouth Aspect Ratio。

    mouth为Dlib嘴部区域的20个关键点，
    其中索引12至19对应内部嘴唇轮廓。

    公式：
        MAR = (A + B + C) / (2D)
    """

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

    mar = (
        vertical_distance_1
        + vertical_distance_2
        + vertical_distance_3
    ) / (
        2.0 * horizontal_distance
    )

    return float(mar)


# ============================================================
# 4. 文件和标签处理
# ============================================================

def find_predictor_path():
    """从候选路径中查找Dlib关键点模型。"""

    for predictor_path in PREDICTOR_CANDIDATES:

        if predictor_path.exists():
            return predictor_path

    return None


def collect_video_files(dataset_path):
    """递归查找数据集中的视频文件。"""

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


def normalize_text(text):
    """统一文件名格式，便于判断标签。"""

    return (
        str(text)
        .strip()
        .lower()
        .replace("\\", "/")
        .replace("-", "_")
        .replace(" ", "_")
    )


def infer_video_label(video_path, dataset_root):
    """
    根据文件名和所在目录判断视频标签。

    优先判断非哈欠关键词，防止 no_yawn
    因为包含 yawn 而被错误标为疲劳。
    """

    try:
        relative_path = video_path.relative_to(
            dataset_root
        )
    except ValueError:
        relative_path = video_path

    path_text = normalize_text(
        relative_path
    )

    normal_keywords = [
        "no_yawn",
        "noyawn",
        "not_yawn",
        "non_yawn",
        "without_yawn",
        "normal",
        "alert",
        "awake"
    ]

    fatigue_keywords = [
        "yawn",
        "fatigue",
        "drowsy",
        "sleepy"
    ]

    if any(
        keyword in path_text
        for keyword in normal_keywords
    ):
        return NORMAL_LABEL

    if any(
        keyword in path_text
        for keyword in fatigue_keywords
    ):
        return FATIGUE_LABEL

    # 未发现明确疲劳关键词时，默认按正常视频处理
    return NORMAL_LABEL


def create_video_id(video_path, dataset_root):
    """
    使用相对路径生成唯一的视频编号。

    例如：
        Male/01_yawn.avi
    转换为：
        Male__01_yawn
    """

    try:
        relative_path = video_path.relative_to(
            dataset_root
        )
    except ValueError:
        relative_path = video_path.name

    relative_path = Path(relative_path)

    video_id = str(
        relative_path.with_suffix("")
    )

    video_id = (
        video_id
        .replace("\\", "__")
        .replace("/", "__")
        .replace(" ", "_")
    )

    return video_id


# ============================================================
# 5. 图像处理工具
# ============================================================

def resize_frame(frame, target_width):
    """保持原始比例缩放视频帧。"""

    height, width = frame.shape[:2]

    if width <= 0 or height <= 0:
        return frame

    resize_ratio = target_width / float(width)

    target_height = int(
        height * resize_ratio
    )

    resized_frame = cv2.resize(
        frame,
        (target_width, target_height),
        interpolation=cv2.INTER_AREA
    )

    return resized_frame


def select_largest_face(face_rectangles):
    """
    多人画面中选择面积最大的人脸。

    驾驶员通常位于画面主体位置，
    使用最大人脸可减少乘客或背景人脸干扰。
    """

    if not face_rectangles:
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
    """
    获取当前帧在原始视频中的时间。

    优先使用视频自带时间戳；
    如果时间戳不可用，则使用帧编号除以FPS。
    """

    timestamp_ms = video_capture.get(
        cv2.CAP_PROP_POS_MSEC
    )

    if timestamp_ms >= 0:
        return float(
            timestamp_ms / 1000.0
        )

    if video_fps > 0:
        return float(
            frame_index / video_fps
        )

    return 0.0


# ============================================================
# 6. 单个视频特征提取
# ============================================================

def process_video(
    video_path,
    dataset_root,
    detector,
    predictor,
    writer,
    landmark_indexes
):
    """
    处理一个视频并将特征写入CSV。

    返回：
        该视频的处理统计信息
    """

    (
        left_eye_start,
        left_eye_end,
        right_eye_start,
        right_eye_end,
        mouth_start,
        mouth_end
    ) = landmark_indexes

    video_capture = cv2.VideoCapture(
        str(video_path)
    )

    if not video_capture.isOpened():

        print(
            f"[WARNING] 无法打开视频：{video_path}"
        )

        return {
            "rows": 0,
            "frames": 0,
            "sampled_frames": 0,
            "face_failures": 0,
            "open_failed": True
        }

    video_id = create_video_id(
        video_path,
        dataset_root
    )

    label = infer_video_label(
        video_path,
        dataset_root
    )

    video_fps = video_capture.get(
        cv2.CAP_PROP_FPS
    )

    frame_index = 0
    sampled_frames = 0
    written_rows = 0
    face_failures = 0

    while True:

        success, frame = video_capture.read()

        if not success:
            break

        frame_index += 1

        # 按设定间隔采样
        if (
            frame_index
            % FRAME_SAMPLE_INTERVAL
            != 0
        ):
            continue

        sampled_frames += 1

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

            face_failures += 1
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

            face_failures += 1
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
            continue

        average_ear = (
            left_ear + right_ear
        ) / 2.0

        timestamp_seconds = get_video_timestamp(
            video_capture,
            frame_index,
            video_fps
        )

        writer.writerow(
            [
                video_id,
                video_path.name,
                frame_index,
                f"{timestamp_seconds:.6f}",
                f"{average_ear:.8f}",
                f"{mar:.8f}",
                label
            ]
        )

        written_rows += 1

    video_capture.release()

    return {
        "rows": written_rows,
        "frames": frame_index,
        "sampled_frames": sampled_frames,
        "face_failures": face_failures,
        "open_failed": False
    }


# ============================================================
# 7. 主程序
# ============================================================

def main():

    print("=" * 70)
    print("Facial Feature Extraction")
    print("=" * 70)

    # --------------------------------------------------------
    # 检查数据集
    # --------------------------------------------------------

    if not YAWDD_PATH.exists():

        print("\n[ERROR] 找不到视频数据集：")
        print(YAWDD_PATH)
        return

    # --------------------------------------------------------
    # 查找Dlib关键点模型
    # --------------------------------------------------------

    predictor_path = find_predictor_path()

    if predictor_path is None:

        print(
            "\n[ERROR] 找不到 "
            "shape_predictor_68_face_landmarks.dat"
        )

        print("\n已检查以下路径：")

        for candidate_path in PREDICTOR_CANDIDATES:
            print(candidate_path)

        return

    print("\nDlib关键点模型：")
    print(predictor_path)

    # --------------------------------------------------------
    # 查找全部视频
    # --------------------------------------------------------

    video_files = collect_video_files(
        YAWDD_PATH
    )

    if len(video_files) == 0:

        print("\n[ERROR] 数据集目录中没有找到视频。")
        print(YAWDD_PATH)
        return

    normal_video_count = sum(
        infer_video_label(
            video_path,
            YAWDD_PATH
        ) == NORMAL_LABEL
        for video_path in video_files
    )

    fatigue_video_count = (
        len(video_files)
        - normal_video_count
    )

    print("\n视频统计")
    print("-" * 70)
    print(f"视频总数：       {len(video_files)}")
    print(f"Normal视频：     {normal_video_count}")
    print(f"Fatigue视频：    {fatigue_video_count}")
    print(
        f"采样间隔：       每"
        f"{FRAME_SAMPLE_INTERVAL}帧提取一次"
    )

    # --------------------------------------------------------
    # 初始化Dlib
    # --------------------------------------------------------

    try:
        detector = dlib.get_frontal_face_detector()

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

    # --------------------------------------------------------
    # 创建输出目录
    # --------------------------------------------------------

    OUTPUT_CSV_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    total_rows = 0
    total_frames = 0
    total_sampled_frames = 0
    total_face_failures = 0
    failed_videos = 0

    # --------------------------------------------------------
    # 开始提取
    # --------------------------------------------------------

    with open(
        OUTPUT_CSV_PATH,
        mode="w",
        newline="",
        encoding="utf-8-sig"
    ) as csv_file:

        writer = csv.writer(
            csv_file
        )

        writer.writerow(
            [
                "VideoID",
                "SourceFile",
                "FrameIndex",
                "TimestampSec",
                "EAR",
                "MAR",
                "Label"
            ]
        )

        for video_number, video_path in enumerate(
            video_files,
            start=1
        ):

            label = infer_video_label(
                video_path,
                YAWDD_PATH
            )

            label_name = (
                "Fatigue"
                if label == FATIGUE_LABEL
                else "Normal"
            )

            print(
                f"\n[{video_number}/{len(video_files)}] "
                f"{video_path.name}"
            )

            print(
                f"标签：{label_name}"
            )

            statistics = process_video(
                video_path=video_path,
                dataset_root=YAWDD_PATH,
                detector=detector,
                predictor=predictor,
                writer=writer,
                landmark_indexes=landmark_indexes
            )

            total_rows += statistics["rows"]
            total_frames += statistics["frames"]
            total_sampled_frames += (
                statistics["sampled_frames"]
            )
            total_face_failures += (
                statistics["face_failures"]
            )

            if statistics["open_failed"]:
                failed_videos += 1

            print(
                f"写入特征：{statistics['rows']} 行 "
                f"| 人脸未检测："
                f"{statistics['face_failures']} 帧"
            )

    # --------------------------------------------------------
    # 输出最终结果
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("Feature Extraction Completed")
    print("=" * 70)

    print(f"处理视频：       {len(video_files)}")
    print(f"打开失败视频：   {failed_videos}")
    print(f"读取总帧数：     {total_frames}")
    print(f"采样帧数：       {total_sampled_frames}")
    print(f"人脸检测失败：   {total_face_failures}")
    print(f"有效特征行数：   {total_rows}")
    print(f"输出文件：       {OUTPUT_CSV_PATH}")

    if total_rows == 0:

        print(
            "\n[WARNING] CSV已经创建，"
            "但没有写入有效特征。"
        )

        print(
            "请检查视频内容、Dlib模型"
            "以及视频中人脸是否清晰可见。"
        )

    else:

        face_detection_rate = (
            total_rows
            / total_sampled_frames
            * 100
            if total_sampled_frames > 0
            else 0.0
        )

        print(
            f"有效特征提取率： "
            f"{face_detection_rate:.2f}%"
        )


if __name__ == "__main__":
    main()