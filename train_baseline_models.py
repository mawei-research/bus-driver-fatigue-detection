"""
Traditional machine-learning baseline training script.

Input CSV columns:
    VideoID, SourceFile, FrameIndex, TimestampSec, EAR, MAR, Label

Label definition:
    0 = Normal
    1 = Fatigue / Yawn

The split is performed at video level so that frames from the same video
cannot appear in both training and evaluation sets.
"""

import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

from imblearn.over_sampling import SMOTE
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
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
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# ============================================================
# 1. 路径设置
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\DELL\PycharmProjects\pythonProject9"
    r"\BusDriverFatigueDetection"
)

CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "fatigue_features.csv"
)

MODEL_SAVE_DIR = (
    PROJECT_ROOT
    / "models"
)

RESULT_SAVE_DIR = (
    PROJECT_ROOT
    / "data"
)

BEST_MODEL_PATH = (
    MODEL_SAVE_DIR
    / "best_baseline_model.pkl"
)

SCALER_PATH = (
    MODEL_SAVE_DIR
    / "best_baseline_scaler.pkl"
)

MODEL_BUNDLE_PATH = (
    MODEL_SAVE_DIR
    / "best_baseline_bundle.pkl"
)

MODEL_COMPARISON_CHART_PATH = (
    RESULT_SAVE_DIR
    / "baseline_model_comparison.png"
)

CONFUSION_MATRIX_PATH = (
    RESULT_SAVE_DIR
    / "baseline_test_confusion_matrix.png"
)

VALIDATION_RESULTS_PATH = (
    RESULT_SAVE_DIR
    / "baseline_validation_results.csv"
)

TEST_PREDICTIONS_PATH = (
    RESULT_SAVE_DIR
    / "baseline_test_predictions.csv"
)

VIDEO_TEST_PREDICTIONS_PATH = (
    RESULT_SAVE_DIR
    / "baseline_video_test_predictions.csv"
)

DATA_SPLIT_PATH = (
    RESULT_SAVE_DIR
    / "baseline_video_split.csv"
)

TEST_REPORT_PATH = (
    RESULT_SAVE_DIR
    / "baseline_test_report.txt"
)


# ============================================================
# 2. 实验参数
# ============================================================

RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

DECISION_THRESHOLD = 0.50

REQUIRED_COLUMNS = {
    "VideoID",
    "SourceFile",
    "FrameIndex",
    "TimestampSec",
    "EAR",
    "MAR",
    "Label"
}

FEATURE_COLUMNS = [
    "EAR",
    "MAR",
    "EAR_mean_5",
    "EAR_std_5",
    "EAR_mean_15",
    "EAR_std_15",
    "EAR_mean_30",
    "EAR_std_30",
    "MAR_mean_15",
    "MAR_std_15",
    "EAR_diff",
    "MAR_diff"
]

CLASS_NAMES = [
    "Normal",
    "Fatigue"
]


# ============================================================
# 3. 创建输出目录
# ============================================================

def create_output_directories():
    """创建模型和实验结果目录。"""

    MODEL_SAVE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    RESULT_SAVE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# 4. 读取并检查特征文件
# ============================================================

def load_feature_csv(csv_path):
    """读取新版特征文件并完成基础检查。"""

    if not csv_path.exists():

        raise FileNotFoundError(
            f"找不到特征文件：{csv_path}\n"
            "请先运行修改后的 extract_features.py。"
        )

    data = pd.read_csv(
        csv_path
    )

    if data.empty:

        raise ValueError(
            "特征文件为空，没有可用于训练的数据。"
        )

    missing_columns = (
        REQUIRED_COLUMNS
        .difference(data.columns)
    )

    if missing_columns:

        raise ValueError(
            "特征文件缺少以下字段："
            f"{sorted(missing_columns)}\n"
            "请使用修改后的 extract_features.py "
            "重新生成 fatigue_features.csv。"
        )

    numeric_columns = [
        "FrameIndex",
        "TimestampSec",
        "EAR",
        "MAR",
        "Label"
    ]

    for column in numeric_columns:

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    before_drop = len(data)

    data = data.dropna(
        subset=[
            "VideoID",
            "SourceFile",
            "FrameIndex",
            "EAR",
            "MAR",
            "Label"
        ]
    ).copy()

    removed_rows = (
        before_drop
        - len(data)
    )

    if removed_rows > 0:

        print(
            f"[INFO] 已删除 {removed_rows} 行无效数据。"
        )

    data["VideoID"] = (
        data["VideoID"]
        .astype(str)
    )

    data["SourceFile"] = (
        data["SourceFile"]
        .astype(str)
    )

    data["FrameIndex"] = (
        data["FrameIndex"]
        .astype(int)
    )

    data["Label"] = (
        data["Label"]
        .astype(int)
    )

    unexpected_labels = sorted(
        set(data["Label"].unique())
        .difference({0, 1})
    )

    if unexpected_labels:

        raise ValueError(
            "Label列中发现非0/1标签："
            f"{unexpected_labels}"
        )

    # 检查同一视频是否存在多个标签
    label_counts_per_video = (
        data.groupby("VideoID")["Label"]
        .nunique()
    )

    inconsistent_videos = (
        label_counts_per_video[
            label_counts_per_video > 1
        ]
    )

    if not inconsistent_videos.empty:

        raise ValueError(
            "以下视频存在多个不同标签，无法继续训练：\n"
            + "\n".join(
                inconsistent_videos.index.tolist()
            )
        )

    data = data.sort_values(
        [
            "VideoID",
            "FrameIndex"
        ]
    ).reset_index(
        drop=True
    )

    return data


# ============================================================
# 5. 时序特征工程
# ============================================================

def add_features_to_one_video(video_data):
    """在单个视频内部计算滑动统计特征。"""

    video_data = video_data.sort_values(
        "FrameIndex"
    ).copy()

    ear_series = (
        video_data["EAR"]
        .astype(float)
    )

    mar_series = (
        video_data["MAR"]
        .astype(float)
    )

    # EAR多尺度滑动统计特征
    for window_size in (
        5,
        15,
        30
    ):

        video_data[
            f"EAR_mean_{window_size}"
        ] = ear_series.rolling(
            window=window_size,
            min_periods=1
        ).mean()

        video_data[
            f"EAR_std_{window_size}"
        ] = ear_series.rolling(
            window=window_size,
            min_periods=1
        ).std(
            ddof=0
        )

    # MAR滑动统计特征
    video_data[
        "MAR_mean_15"
    ] = mar_series.rolling(
        window=15,
        min_periods=1
    ).mean()

    video_data[
        "MAR_std_15"
    ] = mar_series.rolling(
        window=15,
        min_periods=1
    ).std(
        ddof=0
    )

    # 一阶变化量
    video_data[
        "EAR_diff"
    ] = ear_series.diff().fillna(
        0.0
    )

    video_data[
        "MAR_diff"
    ] = mar_series.diff().fillna(
        0.0
    )

    return video_data


def extract_time_series_features(data):
    """
    分视频计算时序特征。

    每个视频单独计算滑动窗口，避免不同视频之间
    的数据被错误连接。
    """

    processed_videos = []

    for _, video_data in data.groupby(
        "VideoID",
        sort=False
    ):

        processed_video = (
            add_features_to_one_video(
                video_data
            )
        )

        processed_videos.append(
            processed_video
        )

    featured_data = pd.concat(
        processed_videos,
        axis=0,
        ignore_index=True
    )

    feature_values = (
        featured_data[FEATURE_COLUMNS]
        .to_numpy(dtype=float)
    )

    invalid_mask = (
        ~np.isfinite(feature_values)
        .all(axis=1)
    )

    invalid_count = int(
        invalid_mask.sum()
    )

    if invalid_count > 0:

        print(
            f"[INFO] 已删除 {invalid_count} 行非有限特征。"
        )

        featured_data = (
            featured_data
            .loc[~invalid_mask]
            .copy()
        )

    if featured_data.empty:

        raise ValueError(
            "特征工程完成后没有剩余有效数据。"
        )

    return featured_data.reset_index(
        drop=True
    )


# ============================================================
# 6. 创建视频索引表
# ============================================================

def build_video_table(data):
    """生成每个视频一行的视频级索引表。"""

    video_table = (
        data.groupby(
            "VideoID",
            as_index=False
        )
        .agg(
            SourceFile=(
                "SourceFile",
                "first"
            ),
            Label=(
                "Label",
                "first"
            ),
            SampledFrames=(
                "FrameIndex",
                "size"
            )
        )
    )

    return video_table


# ============================================================
# 7. 按视频划分数据集
# ============================================================

def split_videos(video_table):
    """
    按视频进行70%、15%、15%的分层划分。

    同一个视频的帧不会同时出现在训练集和测试集。
    """

    total_ratio = (
        TRAIN_RATIO
        + VALIDATION_RATIO
        + TEST_RATIO
    )

    if not np.isclose(
        total_ratio,
        1.0
    ):

        raise ValueError(
            "训练、验证和测试比例之和必须等于1。"
        )

    class_video_counts = (
        video_table["Label"]
        .value_counts()
        .sort_index()
    )

    if len(class_video_counts) < 2:

        raise ValueError(
            "视频数据只包含一个类别，"
            "无法训练二分类模型。"
        )

    if class_video_counts.min() < 4:

        raise ValueError(
            "每个类别至少需要4个不同视频，"
            "才能完成训练、验证和测试划分。\n"
            f"当前视频数量："
            f"{class_video_counts.to_dict()}"
        )

    # 第一次划分：70%训练，30%剩余
    (
        train_videos,
        remaining_videos
    ) = train_test_split(
        video_table,
        test_size=1.0 - TRAIN_RATIO,
        random_state=RANDOM_SEED,
        stratify=video_table["Label"]
    )

    # 第二次划分：剩余数据平均分为验证和测试
    relative_test_ratio = (
        TEST_RATIO
        / (
            VALIDATION_RATIO
            + TEST_RATIO
        )
    )

    (
        validation_videos,
        test_videos
    ) = train_test_split(
        remaining_videos,
        test_size=relative_test_ratio,
        random_state=RANDOM_SEED,
        stratify=remaining_videos["Label"]
    )

    return (
        train_videos.reset_index(
            drop=True
        ),
        validation_videos.reset_index(
            drop=True
        ),
        test_videos.reset_index(
            drop=True
        )
    )


def save_video_split(
    train_videos,
    validation_videos,
    test_videos
):
    """保存视频级数据划分结果。"""

    split_frames = []

    split_groups = [
        (
            "Training",
            train_videos
        ),
        (
            "Validation",
            validation_videos
        ),
        (
            "Testing",
            test_videos
        )
    ]

    for split_name, split_data in split_groups:

        current_data = (
            split_data.copy()
        )

        current_data[
            "Split"
        ] = split_name

        split_frames.append(
            current_data
        )

    split_table = pd.concat(
        split_frames,
        ignore_index=True
    )

    split_table.to_csv(
        DATA_SPLIT_PATH,
        index=False,
        encoding="utf-8-sig"
    )


def select_rows_by_videos(
    data,
    videos
):
    """根据视频编号选出相应的帧特征。"""

    video_ids = set(
        videos["VideoID"]
        .tolist()
    )

    selected_data = data[
        data["VideoID"].isin(
            video_ids
        )
    ].copy()

    return selected_data.reset_index(
        drop=True
    )


def print_split_summary(
    split_name,
    video_data,
    frame_data
):
    """打印数据划分统计。"""

    video_counts = (
        video_data["Label"]
        .value_counts()
        .to_dict()
    )

    frame_counts = (
        frame_data["Label"]
        .value_counts()
        .to_dict()
    )

    print(
        f"{split_name:<10} | "
        f"Videos: {len(video_data):>4} "
        f"(Normal={video_counts.get(0, 0)}, "
        f"Fatigue={video_counts.get(1, 0)}) "
        f"| Rows: {len(frame_data):>7} "
        f"(Normal={frame_counts.get(0, 0)}, "
        f"Fatigue={frame_counts.get(1, 0)})"
    )


# ============================================================
# 8. SMOTE样本平衡
# ============================================================

def apply_smote(
    features,
    labels
):
    """仅对训练数据使用SMOTE。"""

    label_counts = (
        pd.Series(labels)
        .value_counts()
    )

    if len(label_counts) < 2:

        raise ValueError(
            "训练数据只包含一个类别，"
            "无法执行SMOTE。"
        )

    minority_count = int(
        label_counts.min()
    )

    if minority_count < 2:

        print(
            "[WARNING] 少数类样本不足2个，"
            "本次不使用SMOTE。"
        )

        return (
            features,
            labels
        )

    k_neighbors = min(
        5,
        minority_count - 1
    )

    smote = SMOTE(
        random_state=RANDOM_SEED,
        k_neighbors=k_neighbors
    )

    (
        resampled_features,
        resampled_labels
    ) = smote.fit_resample(
        features,
        labels
    )

    return (
        resampled_features,
        resampled_labels
    )


# ============================================================
# 9. 模型定义
# ============================================================

def build_models():
    """建立传统机器学习基线模型池。"""

    models = {
        "SVM (RBF)": SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            probability=True,
            random_state=RANDOM_SEED
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=2,
            random_state=RANDOM_SEED,
            n_jobs=-1
        ),

        "MLP": MLPClassifier(
            hidden_layer_sizes=(
                64,
                32
            ),
            activation="relu",
            solver="adam",
            learning_rate_init=1e-3,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.10,
            n_iter_no_change=20,
            random_state=RANDOM_SEED
        ),

        "XGBoost": xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.80,
            colsample_bytree=0.80,
            eval_metric="logloss",
            random_state=RANDOM_SEED,
            n_jobs=-1
        )
    }

    return models


def get_positive_probabilities(
    model,
    features
):
    """获取类别1的预测概率。"""

    if hasattr(
        model,
        "predict_proba"
    ):

        return model.predict_proba(
            features
        )[:, 1]

    if hasattr(
        model,
        "decision_function"
    ):

        scores = model.decision_function(
            features
        )

        return (
            1.0
            / (
                1.0
                + np.exp(-scores)
            )
        )

    return model.predict(
        features
    ).astype(float)


# ============================================================
# 10. 指标计算
# ============================================================

def calculate_binary_metrics(
    true_labels,
    predicted_labels,
    probabilities
):
    """计算二分类评价指标。"""

    metrics = {
        "Accuracy": accuracy_score(
            true_labels,
            predicted_labels
        ),

        "MacroPrecision": precision_score(
            true_labels,
            predicted_labels,
            average="macro",
            zero_division=0
        ),

        "MacroRecall": recall_score(
            true_labels,
            predicted_labels,
            average="macro",
            zero_division=0
        ),

        "MacroF1": f1_score(
            true_labels,
            predicted_labels,
            average="macro",
            zero_division=0
        )
    }

    try:

        metrics[
            "ROCAUC"
        ] = roc_auc_score(
            true_labels,
            probabilities
        )

    except ValueError:

        metrics[
            "ROCAUC"
        ] = np.nan

    return metrics


def aggregate_video_predictions(
    frame_data,
    probabilities
):
    """
    将帧级预测结果按视频进行平均。

    每个视频最终得到一个疲劳概率和一个预测标签。
    """

    prediction_data = frame_data[
        [
            "VideoID",
            "SourceFile",
            "Label"
        ]
    ].copy()

    prediction_data[
        "FatigueProbability"
    ] = probabilities

    video_predictions = (
        prediction_data.groupby(
            "VideoID",
            as_index=False
        )
        .agg(
            SourceFile=(
                "SourceFile",
                "first"
            ),
            TrueLabel=(
                "Label",
                "first"
            ),
            MeanFatigueProbability=(
                "FatigueProbability",
                "mean"
            ),
            SampledFrames=(
                "FatigueProbability",
                "size"
            )
        )
    )

    video_predictions[
        "PredictedLabel"
    ] = (
        video_predictions[
            "MeanFatigueProbability"
        ]
        >= DECISION_THRESHOLD
    ).astype(int)

    video_predictions[
        "Correct"
    ] = (
        video_predictions[
            "TrueLabel"
        ]
        == video_predictions[
            "PredictedLabel"
        ]
    ).astype(int)

    return video_predictions


def evaluate_predictions(
    frame_data,
    probabilities
):
    """同时计算帧级和视频级评价指标。"""

    true_frame_labels = (
        frame_data["Label"]
        .to_numpy(dtype=int)
    )

    predicted_frame_labels = (
        probabilities
        >= DECISION_THRESHOLD
    ).astype(int)

    frame_metrics = (
        calculate_binary_metrics(
            true_frame_labels,
            predicted_frame_labels,
            probabilities
        )
    )

    video_predictions = (
        aggregate_video_predictions(
            frame_data,
            probabilities
        )
    )

    video_metrics = (
        calculate_binary_metrics(
            video_predictions[
                "TrueLabel"
            ].to_numpy(
                dtype=int
            ),
            video_predictions[
                "PredictedLabel"
            ].to_numpy(
                dtype=int
            ),
            video_predictions[
                "MeanFatigueProbability"
            ].to_numpy(
                dtype=float
            )
        )
    )

    return (
        frame_metrics,
        video_metrics,
        predicted_frame_labels,
        video_predictions
    )


# ============================================================
# 11. 训练并比较基线模型
# ============================================================

def train_and_compare_models(
    models,
    train_features,
    train_labels,
    validation_features,
    validation_data
):
    """训练全部模型，并使用验证集选择最佳模型。"""

    validation_rows = []

    print("\n开始训练和比较基线模型")
    print("-" * 90)

    for model_name, model in models.items():

        start_time = (
            time.perf_counter()
        )

        model.fit(
            train_features,
            train_labels
        )

        training_seconds = (
            time.perf_counter()
            - start_time
        )

        probabilities = (
            get_positive_probabilities(
                model,
                validation_features
            )
        )

        (
            frame_metrics,
            video_metrics,
            _,
            _
        ) = evaluate_predictions(
            validation_data,
            probabilities
        )

        validation_rows.append(
            {
                "Model": model_name,
                "TrainingSeconds": training_seconds,

                "FrameAccuracy":
                    frame_metrics[
                        "Accuracy"
                    ],

                "FrameMacroPrecision":
                    frame_metrics[
                        "MacroPrecision"
                    ],

                "FrameMacroRecall":
                    frame_metrics[
                        "MacroRecall"
                    ],

                "FrameMacroF1":
                    frame_metrics[
                        "MacroF1"
                    ],

                "FrameROCAUC":
                    frame_metrics[
                        "ROCAUC"
                    ],

                "VideoAccuracy":
                    video_metrics[
                        "Accuracy"
                    ],

                "VideoMacroPrecision":
                    video_metrics[
                        "MacroPrecision"
                    ],

                "VideoMacroRecall":
                    video_metrics[
                        "MacroRecall"
                    ],

                "VideoMacroF1":
                    video_metrics[
                        "MacroF1"
                    ],

                "VideoROCAUC":
                    video_metrics[
                        "ROCAUC"
                    ]
            }
        )

        print(
            f"{model_name:<16} | "
            f"Video F1: "
            f"{video_metrics['MacroF1']:.4f} | "
            f"Video Acc: "
            f"{video_metrics['Accuracy']:.4f} | "
            f"Frame F1: "
            f"{frame_metrics['MacroF1']:.4f} | "
            f"Time: "
            f"{training_seconds:.2f}s"
        )

    validation_results = pd.DataFrame(
        validation_rows
    )

    validation_results = (
        validation_results.sort_values(
            by=[
                "VideoMacroF1",
                "VideoAccuracy",
                "FrameMacroF1"
            ],
            ascending=[
                False,
                False,
                False
            ]
        )
        .reset_index(
            drop=True
        )
    )

    best_model_name = (
        validation_results.loc[
            0,
            "Model"
        ]
    )

    validation_results.to_csv(
        VALIDATION_RESULTS_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    return (
        best_model_name,
        validation_results
    )


# ============================================================
# 12. 保存模型对比图
# ============================================================

def save_model_comparison_chart(
    validation_results
):
    """保存验证集视频级模型性能对比图。"""

    model_names = (
        validation_results[
            "Model"
        ].tolist()
    )

    accuracies = (
        validation_results[
            "VideoAccuracy"
        ].to_numpy()
        * 100
    )

    f1_scores = (
        validation_results[
            "VideoMacroF1"
        ].to_numpy()
        * 100
    )

    positions = np.arange(
        len(model_names)
    )

    bar_width = 0.36

    plt.figure(
        figsize=(11, 6)
    )

    accuracy_bars = plt.bar(
        positions
        - bar_width / 2,
        accuracies,
        width=bar_width,
        label="Video Accuracy"
    )

    f1_bars = plt.bar(
        positions
        + bar_width / 2,
        f1_scores,
        width=bar_width,
        label="Video Macro F1"
    )

    for bars in (
        accuracy_bars,
        f1_bars
    ):

        for bar in bars:

            value = bar.get_height()

            plt.text(
                bar.get_x()
                + bar.get_width() / 2,
                value + 0.5,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=9
            )

    lower_limit = max(
        0.0,
        min(
            np.min(accuracies),
            np.min(f1_scores)
        ) - 8.0
    )

    plt.ylim(
        lower_limit,
        105
    )

    plt.xticks(
        positions,
        model_names
    )

    plt.xlabel(
        "Classifier"
    )

    plt.ylabel(
        "Performance (%)"
    )

    plt.title(
        "Baseline Model Validation Performance"
    )

    plt.legend()

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        MODEL_COMPARISON_CHART_PATH,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 13. 保存混淆矩阵
# ============================================================

def save_confusion_matrix(matrix):
    """保存最佳模型的视频级测试混淆矩阵。"""

    plt.figure(
        figsize=(6, 5)
    )

    plt.imshow(
        matrix,
        interpolation="nearest"
    )

    plt.title(
        "Best Baseline Model: "
        "Video-Level Test Confusion Matrix"
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


# ============================================================
# 14. 保存帧级测试预测
# ============================================================

def save_frame_test_predictions(
    frame_data,
    probabilities,
    predicted_labels
):
    """保存测试集逐帧预测结果。"""

    output_data = frame_data[
        [
            "VideoID",
            "SourceFile",
            "FrameIndex",
            "TimestampSec",
            "EAR",
            "MAR",
            "Label"
        ]
    ].copy()

    output_data = output_data.rename(
        columns={
            "Label": "TrueLabel"
        }
    )

    output_data[
        "FatigueProbability"
    ] = probabilities

    output_data[
        "PredictedLabel"
    ] = predicted_labels

    output_data[
        "Correct"
    ] = (
        output_data[
            "TrueLabel"
        ]
        == output_data[
            "PredictedLabel"
        ]
    ).astype(int)

    output_data.to_csv(
        TEST_PREDICTIONS_PATH,
        index=False,
        encoding="utf-8-sig"
    )


# ============================================================
# 15. 保存最终测试报告
# ============================================================

def save_test_report(
    best_model_name,
    frame_metrics,
    video_metrics,
    frame_report,
    video_report,
    video_confusion_matrix,
    train_video_count,
    validation_video_count,
    test_video_count,
    train_row_count,
    validation_row_count,
    test_row_count
):
    """保存最佳模型的独立测试报告。"""

    report_text = (
        "Traditional Machine-Learning Baseline Test Report\n"
        "=================================================\n\n"

        f"Selected model: {best_model_name}\n"
        f"Decision threshold: {DECISION_THRESHOLD:.2f}\n"
        f"Feature count: {len(FEATURE_COLUMNS)}\n"
        f"Features: {', '.join(FEATURE_COLUMNS)}\n\n"

        "Video-level split\n"
        "-----------------\n"
        f"Training videos: {train_video_count}\n"
        f"Validation videos: {validation_video_count}\n"
        f"Testing videos: {test_video_count}\n\n"

        "Frame-feature rows\n"
        "------------------\n"
        f"Training rows: {train_row_count}\n"
        f"Validation rows: {validation_row_count}\n"
        f"Testing rows: {test_row_count}\n\n"

        "Frame-level test metrics\n"
        "------------------------\n"
        f"Accuracy: "
        f"{frame_metrics['Accuracy']:.4f}\n"
        f"Macro Precision: "
        f"{frame_metrics['MacroPrecision']:.4f}\n"
        f"Macro Recall: "
        f"{frame_metrics['MacroRecall']:.4f}\n"
        f"Macro F1: "
        f"{frame_metrics['MacroF1']:.4f}\n"
        f"ROC-AUC: "
        f"{frame_metrics['ROCAUC']:.4f}\n\n"

        f"{frame_report}\n"

        "Video-level test metrics\n"
        "------------------------\n"
        f"Accuracy: "
        f"{video_metrics['Accuracy']:.4f}\n"
        f"Macro Precision: "
        f"{video_metrics['MacroPrecision']:.4f}\n"
        f"Macro Recall: "
        f"{video_metrics['MacroRecall']:.4f}\n"
        f"Macro F1: "
        f"{video_metrics['MacroF1']:.4f}\n"
        f"ROC-AUC: "
        f"{video_metrics['ROCAUC']:.4f}\n\n"

        f"{video_report}\n"

        "Video-level confusion matrix\n"
        "----------------------------\n"
        f"{video_confusion_matrix}\n"
    )

    with open(
        TEST_REPORT_PATH,
        mode="w",
        encoding="utf-8"
    ) as report_file:

        report_file.write(
            report_text
        )


# ============================================================
# 16. 主程序
# ============================================================

def main():

    print("=" * 90)
    print(
        "Traditional Machine-Learning "
        "Baseline Training"
    )
    print("=" * 90)

    create_output_directories()

    try:

        raw_data = load_feature_csv(
            CSV_PATH
        )

        data = extract_time_series_features(
            raw_data
        )

        video_table = build_video_table(
            data
        )

        (
            train_videos,
            validation_videos,
            test_videos
        ) = split_videos(
            video_table
        )

    except (
        FileNotFoundError,
        ValueError
    ) as error:

        print(
            f"\n[ERROR] {error}"
        )

        return

    save_video_split(
        train_videos,
        validation_videos,
        test_videos
    )

    train_data = select_rows_by_videos(
        data,
        train_videos
    )

    validation_data = select_rows_by_videos(
        data,
        validation_videos
    )

    test_data = select_rows_by_videos(
        data,
        test_videos
    )

    print("\n数据划分结果")
    print("-" * 90)

    print_split_summary(
        "Training",
        train_videos,
        train_data
    )

    print_split_summary(
        "Validation",
        validation_videos,
        validation_data
    )

    print_split_summary(
        "Testing",
        test_videos,
        test_data
    )

    # --------------------------------------------------------
    # 准备训练和验证特征
    # --------------------------------------------------------

    train_x = train_data[
        FEATURE_COLUMNS
    ].to_numpy(
        dtype=np.float32
    )

    train_y = train_data[
        "Label"
    ].to_numpy(
        dtype=np.int32
    )

    validation_x = validation_data[
        FEATURE_COLUMNS
    ].to_numpy(
        dtype=np.float32
    )

    test_x = test_data[
        FEATURE_COLUMNS
    ].to_numpy(
        dtype=np.float32
    )

    # 标准化器仅使用训练集拟合
    selection_scaler = StandardScaler()

    train_x_scaled = (
        selection_scaler.fit_transform(
            train_x
        )
    )

    validation_x_scaled = (
        selection_scaler.transform(
            validation_x
        )
    )

    try:

        (
            train_x_balanced,
            train_y_balanced
        ) = apply_smote(
            train_x_scaled,
            train_y
        )

    except ValueError as error:

        print(
            f"\n[ERROR] {error}"
        )

        return

    print("\n训练集SMOTE前后")
    print("-" * 90)

    print(
        "Before: "
        f"{pd.Series(train_y).value_counts().sort_index().to_dict()}"
    )

    print(
        "After:  "
        f"{pd.Series(train_y_balanced).value_counts().sort_index().to_dict()}"
    )

    # --------------------------------------------------------
    # 验证集模型比较
    # --------------------------------------------------------

    models = build_models()

    (
        best_model_name,
        validation_results
    ) = train_and_compare_models(
        models=models,
        train_features=train_x_balanced,
        train_labels=train_y_balanced,
        validation_features=validation_x_scaled,
        validation_data=validation_data
    )

    save_model_comparison_chart(
        validation_results
    )

    print("\n验证集选出的最佳模型")
    print("-" * 90)
    print(best_model_name)

    # --------------------------------------------------------
    # 合并训练集和验证集，重新训练最佳模型
    # --------------------------------------------------------

    final_train_data = pd.concat(
        [
            train_data,
            validation_data
        ],
        axis=0,
        ignore_index=True
    )

    final_train_x = final_train_data[
        FEATURE_COLUMNS
    ].to_numpy(
        dtype=np.float32
    )

    final_train_y = final_train_data[
        "Label"
    ].to_numpy(
        dtype=np.int32
    )

    final_scaler = StandardScaler()

    final_train_x_scaled = (
        final_scaler.fit_transform(
            final_train_x
        )
    )

    test_x_scaled = (
        final_scaler.transform(
            test_x
        )
    )

    try:

        (
            final_train_x_balanced,
            final_train_y_balanced
        ) = apply_smote(
            final_train_x_scaled,
            final_train_y
        )

    except ValueError as error:

        print(
            f"\n[ERROR] {error}"
        )

        return

    final_model = clone(
        models[
            best_model_name
        ]
    )

    print(
        "\n使用训练集和验证集重新训练最佳模型……"
    )

    final_model.fit(
        final_train_x_balanced,
        final_train_y_balanced
    )

    # --------------------------------------------------------
    # 独立测试集评估
    # --------------------------------------------------------

    test_probabilities = (
        get_positive_probabilities(
            final_model,
            test_x_scaled
        )
    )

    (
        frame_metrics,
        video_metrics,
        predicted_frame_labels,
        video_predictions
    ) = evaluate_predictions(
        test_data,
        test_probabilities
    )

    true_frame_labels = (
        test_data["Label"]
        .to_numpy(dtype=int)
    )

    frame_report = classification_report(
        true_frame_labels,
        predicted_frame_labels,
        labels=[
            0,
            1
        ],
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )

    true_video_labels = (
        video_predictions[
            "TrueLabel"
        ].to_numpy(
            dtype=int
        )
    )

    predicted_video_labels = (
        video_predictions[
            "PredictedLabel"
        ].to_numpy(
            dtype=int
        )
    )

    video_report = classification_report(
        true_video_labels,
        predicted_video_labels,
        labels=[
            0,
            1
        ],
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )

    video_matrix = confusion_matrix(
        true_video_labels,
        predicted_video_labels,
        labels=[
            0,
            1
        ]
    )

    save_confusion_matrix(
        video_matrix
    )

    save_frame_test_predictions(
        test_data,
        test_probabilities,
        predicted_frame_labels
    )

    video_predictions.to_csv(
        VIDEO_TEST_PREDICTIONS_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    save_test_report(
        best_model_name=best_model_name,
        frame_metrics=frame_metrics,
        video_metrics=video_metrics,
        frame_report=frame_report,
        video_report=video_report,
        video_confusion_matrix=video_matrix,
        train_video_count=len(
            train_videos
        ),
        validation_video_count=len(
            validation_videos
        ),
        test_video_count=len(
            test_videos
        ),
        train_row_count=len(
            train_data
        ),
        validation_row_count=len(
            validation_data
        ),
        test_row_count=len(
            test_data
        )
    )

    # --------------------------------------------------------
    # 保存模型和标准化器
    # --------------------------------------------------------

    joblib.dump(
        final_model,
        BEST_MODEL_PATH
    )

    joblib.dump(
        final_scaler,
        SCALER_PATH
    )

    model_bundle = {
        "model_name":
            best_model_name,

        "model":
            final_model,

        "scaler":
            final_scaler,

        "feature_columns":
            FEATURE_COLUMNS,

        "decision_threshold":
            DECISION_THRESHOLD,

        "label_definition": {
            0: "Normal",
            1: "Fatigue"
        },

        "random_seed":
            RANDOM_SEED
    }

    joblib.dump(
        model_bundle,
        MODEL_BUNDLE_PATH
    )

    # --------------------------------------------------------
    # 控制台输出
    # --------------------------------------------------------

    print("\n" + "=" * 90)
    print("Independent Test Results")
    print("=" * 90)

    print(
        f"Selected model       : "
        f"{best_model_name}"
    )

    print(
        f"Frame Accuracy       : "
        f"{frame_metrics['Accuracy'] * 100:.2f}%"
    )

    print(
        f"Frame Macro F1       : "
        f"{frame_metrics['MacroF1']:.4f}"
    )

    print(
        f"Video Accuracy       : "
        f"{video_metrics['Accuracy'] * 100:.2f}%"
    )

    print(
        f"Video Macro F1       : "
        f"{video_metrics['MacroF1']:.4f}"
    )

    print(
        f"Video ROC-AUC        : "
        f"{video_metrics['ROCAUC']:.4f}"
    )

    print("\nVideo-level Confusion Matrix")

    print(
        video_matrix
    )

    print("\n结果保存位置")
    print("-" * 90)

    print(
        f"最佳模型：           "
        f"{BEST_MODEL_PATH}"
    )

    print(
        f"标准化器：           "
        f"{SCALER_PATH}"
    )

    print(
        f"模型包：             "
        f"{MODEL_BUNDLE_PATH}"
    )

    print(
        f"模型比较图：         "
        f"{MODEL_COMPARISON_CHART_PATH}"
    )

    print(
        f"测试混淆矩阵：       "
        f"{CONFUSION_MATRIX_PATH}"
    )

    print(
        f"验证集结果：         "
        f"{VALIDATION_RESULTS_PATH}"
    )

    print(
        f"帧级测试预测：       "
        f"{TEST_PREDICTIONS_PATH}"
    )

    print(
        f"视频级测试预测：     "
        f"{VIDEO_TEST_PREDICTIONS_PATH}"
    )

    print(
        f"视频划分记录：       "
        f"{DATA_SPLIT_PATH}"
    )

    print(
        f"测试报告：           "
        f"{TEST_REPORT_PATH}"
    )


if __name__ == "__main__":
    main()