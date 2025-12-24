from pathlib import Path
from typing import cast

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch

from py1.knn import knn_predict


_CHOSEN_FONT: str | None = None


def _configure_chinese_font() -> str | None:
    """Find a font with Chinese glyphs and register it if needed."""

    font_candidates = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "NSimSun",
        "PingFang SC",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
    ]

    installed = {f.name for f in fm.fontManager.ttflist}
    chosen = next((name for name in font_candidates if name in installed), None)

    if not chosen:
        fallback_files = [
            Path(r"C:\\Windows\\Fonts\\msyh.ttc"),
            Path(r"C:\\Windows\\Fonts\\msyh.ttf"),
            Path(r"C:\\Windows\\Fonts\\simhei.ttf"),
            Path(r"C:\\Windows\\Fonts\\simsun.ttc"),
        ]
        for font_path in fallback_files:
            if font_path.exists():
                props = fm.FontProperties(fname=str(font_path))
                fm.fontManager.addfont(str(font_path))
                chosen = props.get_name()
                break

    return chosen


def _apply_font(font_name: str | None) -> None:
    """Apply the chosen font to matplotlib rcParams if available."""

    if font_name:
        matplotlib.rcParams["font.family"] = [font_name]
        matplotlib.rcParams["font.sans-serif"] = [font_name]
        matplotlib.rcParams["axes.unicode_minus"] = False
    else:
        print("未检测到中文字体，建议安装 'Microsoft YaHei' 或 'SimHei' 以避免中文显示警告。")


_CHOSEN_FONT = _configure_chinese_font()
_apply_font(_CHOSEN_FONT)

def run_experiment(k: int = 3, visualize: bool = True) -> None:
    """
    载入Iris数据集，划分数据，调用自定义KNN，并输出准确率与可视化。
    """
    print("正在加载 Iris 数据集...")
    iris = cast(Bunch, load_iris())  # 强制类型转换
    X, y = iris.data, iris.target
    feature_names = iris.feature_names

    print(f"数据集加载完成。样本数: {X.shape[0]}, 特征数: {X.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=114514)

    print(f"开始运行 KNN (k={k})...")
    y_pred = knn_predict(X_train, y_train, X_test, k=k)

    acc = np.mean(y_pred == y_test)
    print(f"KNN 预测准确率: {acc:.2f}")

    if visualize:
        visualize_data_and_results(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            y_pred=y_pred,
            feature_names=feature_names,
        )


def visualize_data_and_results(
    *,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    feature_names: list[str],
) -> None:
    """用 seaborn 对特征分布与预测结果进行可视化。"""

    sns.set_theme(style="ticks", palette="deep")
    # seaborn.set_theme can reset font settings, so re-apply after theme.
    _apply_font(_CHOSEN_FONT)

    # 训练集与测试集的前两个特征散点图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.scatterplot(
        ax=axes[0],
        x=X_train[:, 0],
        y=X_train[:, 1],
        hue=y_train,
        alpha=0.7,
        edgecolor="white",
        s=60,
        legend="brief",
    )
    axes[0].set_title("训练集分布")
    axes[0].set_xlabel(feature_names[0])
    axes[0].set_ylabel(feature_names[1])

    correct = y_pred == y_test
    sns.scatterplot(
        ax=axes[1],
        x=X_test[:, 0],
        y=X_test[:, 1],
        hue=y_pred,
        style=correct,
        markers={True: "o", False: "X"},
        alpha=0.85,
        edgecolor="white",
        s=90,
        legend="brief",
    )
    axes[1].set_title("测试集预测(圈=正确, X=错误)")
    axes[1].set_xlabel(feature_names[0])
    axes[1].set_ylabel(feature_names[1])

    plt.tight_layout()

    # 混淆矩阵热力图
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("预测标签")
    plt.ylabel("真实标签")
    plt.title("KNN 混淆矩阵")

    plt.show()
