"""
Eye-State 2D-CNN Training Script

Label definition:
    0 = Closed eyes
    1 = Open eyes

Dataset split:
    70% training
    15% validation
    15% independent testing
"""

import os

# 减少TensorFlow启动时无关的INFO提示
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)
from sklearn.model_selection import train_test_split

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)
from tensorflow.keras.layers import (
    Input,
    Rescaling,
    Conv2D,
    BatchNormalization,
    MaxPooling2D,
    Dropout,
    Flatten,
    Dense
)
from tensorflow.keras.models import Sequential


# ============================================================
# 1. 路径设置
# ============================================================

DATASET_PATH = Path(
    r"C:\Users\DELL\PycharmProjects\pythonProject9"
    r"\BusDriverFatigueDetection\data\archive\mrleyedataset"
)

MODEL_SAVE_PATH = Path(
    r"C:\Users\DELL\PycharmProjects\pythonProject9"
    r"\BusDriverFatigueDetection\models\best_eye_2dcnn.h5"
)

TRAINING_CURVE_PATH = Path(
    r"C:\Users\DELL\PycharmProjects\pythonProject9"
    r"\BusDriverFatigueDetection\data\eye_cnn_training_curve.png"
)

CONFUSION_MATRIX_PATH = Path(
    r"C:\Users\DELL\PycharmProjects\pythonProject9"
    r"\BusDriverFatigueDetection\data\eye_cnn_confusion_matrix.png"
)

TEST_REPORT_PATH = Path(
    r"C:\Users\DELL\PycharmProjects\pythonProject9"
    r"\BusDriverFatigueDetection\data\eye_cnn_test_report.txt"
)


# ============================================================
# 2. 训练参数
# ============================================================

IMG_HEIGHT = 80
IMG_WIDTH = 80

BATCH_SIZE = 64
EPOCHS = 30
RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp"
}

# 标签含义固定如下
CLOSED_LABEL = 0
OPEN_LABEL = 1

CLASS_NAMES = [
    "Closed Eyes",
    "Open Eyes"
]


# ============================================================
# 3. 基础设置
# ============================================================

def set_random_seed(seed):
    """固定随机种子，尽量保证实验可以复现。"""

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def create_output_directories():
    """创建模型、图片和报告的保存目录。"""

    output_paths = [
        MODEL_SAVE_PATH,
        TRAINING_CURVE_PATH,
        CONFUSION_MATRIX_PATH,
        TEST_REPORT_PATH
    ]

    for output_path in output_paths:
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# 4. 数据集目录识别
# ============================================================

def find_class_directories(dataset_path):
    """
    查找闭眼和睁眼文件夹。

    支持以下目录形式：

        0 / 1
        closed / open
        close / open

    返回顺序固定为：

        closed_dir
        open_dir
    """

    class_directories = [
        path
        for path in dataset_path.iterdir()
        if path.is_dir()
    ]

    if len(class_directories) == 0:
        raise ValueError(
            f"数据集目录下没有找到类别文件夹：{dataset_path}"
        )

    folder_map = {
        path.name.strip().lower(): path
        for path in class_directories
    }

    # 优先识别数字文件夹
    if "0" in folder_map and "1" in folder_map:

        closed_dir = folder_map["0"]
        open_dir = folder_map["1"]

        print("\n[INFO] 检测到数字类别目录")
        print(f"       0 = Closed Eyes：{closed_dir}")
        print(f"       1 = Open Eyes：  {open_dir}")

        return closed_dir, open_dir

    # 兼容英文名称
    closed_candidates = [
        path
        for path in class_directories
        if "clos" in path.name.lower()
    ]

    open_candidates = [
        path
        for path in class_directories
        if "open" in path.name.lower()
    ]

    if (
        len(closed_candidates) == 1
        and len(open_candidates) == 1
    ):
        closed_dir = closed_candidates[0]
        open_dir = open_candidates[0]

        print("\n[INFO] 检测到英文类别目录")
        print(f"       Closed Eyes：{closed_dir}")
        print(f"       Open Eyes：  {open_dir}")

        return closed_dir, open_dir

    folder_names = [
        path.name
        for path in class_directories
    ]

    raise ValueError(
        "无法确定闭眼和睁眼类别文件夹。\n"
        "支持的文件夹形式为：0/1 或 closed/open。\n"
        f"当前检测到的文件夹：{folder_names}"
    )


def collect_images_from_directory(directory):
    """递归读取指定文件夹中的所有图像。"""

    image_paths = [
        str(path)
        for path in directory.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
    ]

    image_paths.sort()

    return image_paths


def collect_image_paths(dataset_path):
    """
    收集闭眼和睁眼图像路径。

    标签固定为：
        Closed Eyes = 0
        Open Eyes = 1
    """

    closed_dir, open_dir = find_class_directories(
        dataset_path
    )

    closed_images = collect_images_from_directory(
        closed_dir
    )

    open_images = collect_images_from_directory(
        open_dir
    )

    if len(closed_images) == 0:
        raise ValueError(
            f"闭眼文件夹中没有找到图像：{closed_dir}"
        )

    if len(open_images) == 0:
        raise ValueError(
            f"睁眼文件夹中没有找到图像：{open_dir}"
        )

    image_paths = (
        closed_images
        + open_images
    )

    labels = (
        [CLOSED_LABEL] * len(closed_images)
        + [OPEN_LABEL] * len(open_images)
    )

    print("\n原始数据统计")
    print("-" * 60)
    print(f"Closed Eyes：{len(closed_images)} 张")
    print(f"Open Eyes：  {len(open_images)} 张")
    print(f"Total：      {len(image_paths)} 张")

    return image_paths, labels


# ============================================================
# 5. 数据集划分
# ============================================================

def split_dataset(image_paths, labels):
    """
    按照70%、15%、15%划分训练集、验证集和测试集。
    每部分均保持闭眼、睁眼的类别比例。
    """

    total_ratio = (
        TRAIN_RATIO
        + VALIDATION_RATIO
        + TEST_RATIO
    )

    if not np.isclose(total_ratio, 1.0):
        raise ValueError(
            "训练集、验证集和测试集比例之和必须等于1。"
        )

    # 先分出70%训练集
    (
        train_paths,
        remaining_paths,
        train_labels,
        remaining_labels
    ) = train_test_split(
        image_paths,
        labels,
        test_size=1.0 - TRAIN_RATIO,
        random_state=RANDOM_SEED,
        stratify=labels
    )

    # 剩余30%平均分成验证集和测试集
    relative_test_ratio = (
        TEST_RATIO
        / (VALIDATION_RATIO + TEST_RATIO)
    )

    (
        validation_paths,
        test_paths,
        validation_labels,
        test_labels
    ) = train_test_split(
        remaining_paths,
        remaining_labels,
        test_size=relative_test_ratio,
        random_state=RANDOM_SEED,
        stratify=remaining_labels
    )

    return (
        train_paths,
        validation_paths,
        test_paths,
        train_labels,
        validation_labels,
        test_labels
    )


def count_labels(labels):
    """统计闭眼和睁眼样本数量。"""

    labels_array = np.asarray(labels)

    closed_count = int(
        np.sum(labels_array == CLOSED_LABEL)
    )

    open_count = int(
        np.sum(labels_array == OPEN_LABEL)
    )

    return closed_count, open_count


def print_dataset_summary(
    train_labels,
    validation_labels,
    test_labels
):
    """打印数据集划分结果。"""

    print("\n数据集划分结果")
    print("-" * 60)

    dataset_groups = [
        ("Training", train_labels),
        ("Validation", validation_labels),
        ("Testing", test_labels)
    ]

    for dataset_name, dataset_labels in dataset_groups:

        closed_count, open_count = count_labels(
            dataset_labels
        )

        print(
            f"{dataset_name:<10}: "
            f"{len(dataset_labels):>6} 张 "
            f"| Closed: {closed_count:>6} "
            f"| Open: {open_count:>6}"
        )


# ============================================================
# 6. TensorFlow数据读取
# ============================================================

def load_and_preprocess_image(image_path, label):
    """
    读取图像并进行预处理。

    处理步骤：
        1. 读取图像；
        2. 转换为单通道灰度图；
        3. 缩放到80×80；
        4. 转换为float32。
    """

    image_data = tf.io.read_file(
        image_path
    )

    image = tf.io.decode_image(
        image_data,
        channels=1,
        expand_animations=False
    )

    image.set_shape(
        [None, None, 1]
    )

    image = tf.image.resize(
        image,
        [IMG_HEIGHT, IMG_WIDTH],
        method="bilinear"
    )

    image = tf.cast(
        image,
        tf.float32
    )

    label = tf.cast(
        label,
        tf.float32
    )

    return image, label


def create_tf_dataset(
    image_paths,
    labels,
    training=False
):
    """创建TensorFlow输入数据集。"""

    dataset = tf.data.Dataset.from_tensor_slices(
        (image_paths, labels)
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(image_paths),
            seed=RANDOM_SEED,
            reshuffle_each_iteration=True
        )

    dataset = dataset.map(
        load_and_preprocess_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(
        BATCH_SIZE
    )

    dataset = dataset.prefetch(
        buffer_size=tf.data.AUTOTUNE
    )

    return dataset


# ============================================================
# 7. 构建2D-CNN模型
# ============================================================

def build_eye_model():
    """构建闭眼和睁眼二分类模型。"""

    model = Sequential(
        [
            Input(
                shape=(IMG_HEIGHT, IMG_WIDTH, 1)
            ),

            # 像素归一化到0至1
            Rescaling(
                scale=1.0 / 255.0
            ),

            # 第一组卷积：80×80×1 -> 80×80×32
            Conv2D(
                filters=32,
                kernel_size=(3, 3),
                padding="same",
                activation="relu"
            ),
            BatchNormalization(),
            MaxPooling2D(
                pool_size=(2, 2)
            ),

            # 第二组卷积：40×40×32 -> 40×40×64
            Conv2D(
                filters=64,
                kernel_size=(3, 3),
                padding="same",
                activation="relu"
            ),
            BatchNormalization(),
            MaxPooling2D(
                pool_size=(2, 2)
            ),
            Dropout(0.20),

            # 第三组卷积：20×20×64 -> 20×20×128
            Conv2D(
                filters=128,
                kernel_size=(3, 3),
                padding="same",
                activation="relu"
            ),
            BatchNormalization(),
            MaxPooling2D(
                pool_size=(2, 2)
            ),
            Dropout(0.30),

            # 10×10×128 = 12800
            Flatten(),

            Dense(
                units=128,
                activation="relu"
            ),

            Dropout(0.50),

            # 预测值接近0表示闭眼，接近1表示睁眼
            Dense(
                units=1,
                activation="sigmoid"
            )
        ],
        name="eye_state_2dcnn"
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=1e-3
    )

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# 8. 训练回调
# ============================================================

def create_callbacks():
    """创建模型训练过程中使用的回调函数。"""

    return [
        EarlyStopping(
            monitor="val_loss",
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),

        ModelCheckpoint(
            filepath=str(MODEL_SAVE_PATH),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),

        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.20,
            patience=3,
            min_lr=1e-5,
            verbose=1
        )
    ]


# ============================================================
# 9. 保存训练曲线
# ============================================================

def save_training_curves(history):
    """保存准确率曲线和损失曲线。"""

    epochs = range(
        1,
        len(history.history["loss"]) + 1
    )

    plt.figure(
        figsize=(12, 5)
    )

    # 准确率
    plt.subplot(
        1,
        2,
        1
    )

    plt.plot(
        epochs,
        history.history["accuracy"],
        label="Training Accuracy",
        linewidth=2
    )

    plt.plot(
        epochs,
        history.history["val_accuracy"],
        label="Validation Accuracy",
        linewidth=2
    )

    plt.title(
        "Eye-State 2D-CNN Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)

    # 损失
    plt.subplot(
        1,
        2,
        2
    )

    plt.plot(
        epochs,
        history.history["loss"],
        label="Training Loss",
        linewidth=2
    )

    plt.plot(
        epochs,
        history.history["val_loss"],
        label="Validation Loss",
        linewidth=2
    )

    plt.title(
        "Eye-State 2D-CNN Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        TRAINING_CURVE_PATH,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 10. 保存混淆矩阵
# ============================================================

def save_confusion_matrix(matrix):
    """保存独立测试集混淆矩阵。"""

    plt.figure(
        figsize=(6, 5)
    )

    plt.imshow(
        matrix,
        interpolation="nearest"
    )

    plt.title(
        "Eye-State 2D-CNN Confusion Matrix"
    )

    plt.colorbar()

    positions = np.arange(
        len(CLASS_NAMES)
    )

    plt.xticks(
        positions,
        CLASS_NAMES,
        rotation=20
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
                horizontalalignment="center",
                verticalalignment="center",
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


# ============================================================
# 11. 独立测试集评估
# ============================================================

def evaluate_test_dataset(
    model,
    test_dataset
):
    """在独立测试集上计算模型性能。"""

    true_labels = []
    predicted_probabilities = []

    for images, labels in test_dataset:

        probabilities = model.predict(
            images,
            verbose=0
        ).reshape(-1)

        predicted_probabilities.extend(
            probabilities.tolist()
        )

        true_labels.extend(
            labels.numpy()
            .astype(int)
            .tolist()
        )

    true_labels = np.asarray(
        true_labels,
        dtype=np.int32
    )

    predicted_probabilities = np.asarray(
        predicted_probabilities,
        dtype=np.float32
    )

    predicted_labels = (
        predicted_probabilities >= 0.5
    ).astype(np.int32)

    accuracy = accuracy_score(
        true_labels,
        predicted_labels
    )

    precision = precision_score(
        true_labels,
        predicted_labels,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        true_labels,
        predicted_labels,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        true_labels,
        predicted_labels,
        average="macro",
        zero_division=0
    )

    try:
        roc_auc = roc_auc_score(
            true_labels,
            predicted_probabilities
        )
    except ValueError:
        roc_auc = float("nan")

    report = classification_report(
        true_labels,
        predicted_labels,
        labels=[0, 1],
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=[0, 1]
    )

    save_confusion_matrix(
        matrix
    )

    report_content = (
        "Eye-State 2D-CNN Independent Test Report\n"
        "========================================\n\n"
        "Label definition:\n"
        "0 = Closed Eyes\n"
        "1 = Open Eyes\n\n"
        f"Number of test images: {len(true_labels)}\n\n"
        f"Accuracy:        {accuracy:.4f}\n"
        f"Macro Precision: {precision:.4f}\n"
        f"Macro Recall:    {recall:.4f}\n"
        f"Macro F1-score:  {f1:.4f}\n"
        f"ROC-AUC:         {roc_auc:.4f}\n\n"
        f"{report}\n"
        f"Confusion Matrix:\n{matrix}\n"
    )

    with open(
        TEST_REPORT_PATH,
        mode="w",
        encoding="utf-8"
    ) as report_file:
        report_file.write(
            report_content
        )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": matrix
    }


# ============================================================
# 12. 主程序
# ============================================================

def main():

    print("=" * 65)
    print("Eye-State 2D-CNN Training")
    print("=" * 65)

    set_random_seed(
        RANDOM_SEED
    )

    create_output_directories()

    if not DATASET_PATH.exists():
        print(
            "\n[ERROR] 数据集目录不存在："
        )
        print(
            DATASET_PATH
        )
        return

    try:
        image_paths, labels = collect_image_paths(
            DATASET_PATH
        )

    except ValueError as error:
        print(
            f"\n[ERROR] {error}"
        )
        return

    print("\n标签对应关系")
    print("-" * 60)
    print("0 = Closed Eyes")
    print("1 = Open Eyes")

    try:
        (
            train_paths,
            validation_paths,
            test_paths,
            train_labels,
            validation_labels,
            test_labels
        ) = split_dataset(
            image_paths,
            labels
        )

    except ValueError as error:
        print(
            f"\n[ERROR] 数据集划分失败：{error}"
        )
        return

    print_dataset_summary(
        train_labels,
        validation_labels,
        test_labels
    )

    train_dataset = create_tf_dataset(
        train_paths,
        train_labels,
        training=True
    )

    validation_dataset = create_tf_dataset(
        validation_paths,
        validation_labels,
        training=False
    )

    test_dataset = create_tf_dataset(
        test_paths,
        test_labels,
        training=False
    )

    model = build_eye_model()

    print("\n模型结构")
    print("-" * 60)

    model.summary()

    print("\n开始训练眼睛状态识别模型……")

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS,
        callbacks=create_callbacks(),
        verbose=1
    )

    save_training_curves(
        history
    )

    print("\n正在加载验证集损失最低的模型……")

    best_model = tf.keras.models.load_model(
        str(MODEL_SAVE_PATH)
    )

    print("正在进行独立测试集评估……")

    test_results = evaluate_test_dataset(
        best_model,
        test_dataset
    )

    print("\n" + "=" * 65)
    print("Independent Test Results")
    print("=" * 65)

    print(
        f"Accuracy        : "
        f"{test_results['accuracy'] * 100:.2f}%"
    )

    print(
        f"Macro Precision : "
        f"{test_results['precision']:.4f}"
    )

    print(
        f"Macro Recall    : "
        f"{test_results['recall']:.4f}"
    )

    print(
        f"Macro F1-score  : "
        f"{test_results['f1']:.4f}"
    )

    print(
        f"ROC-AUC         : "
        f"{test_results['roc_auc']:.4f}"
    )

    print("\nConfusion Matrix")
    print(
        test_results["confusion_matrix"]
    )

    print("\n结果保存位置")
    print("-" * 65)
    print(f"模型：      {MODEL_SAVE_PATH}")
    print(f"训练曲线：  {TRAINING_CURVE_PATH}")
    print(f"混淆矩阵：  {CONFUSION_MATRIX_PATH}")
    print(f"测试报告：  {TEST_REPORT_PATH}")

    print("\n训练和测试全部完成。")


if __name__ == "__main__":
    main()