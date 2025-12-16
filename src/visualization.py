import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# helper function
def bar_chart(values, counts, title_name: str = "", xlabel: str = "Giá trị", ylabel: str = "Tần suất"):
    """Vẽ bar chart đơn giản từ danh sách giá trị và tần suất."""
    plt.figure(figsize=(8, 5))
    sns.barplot(x=values, y=counts, color="#4C78A8")
    plt.title(title_name)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()


def countplot_from_array(arr, title_name: str = ""):
    """Đếm tần suất trong mảng phân loại và vẽ bar chart."""
    values, counts = np.unique(arr, return_counts=True)
    bar_chart(values, counts, title_name=title_name, xlabel="Nhóm", ylabel="Số lượng")


def plot_hist(data, bins: int = 20, title_name: str = "", xlabel: str = "Giá trị"):
    data = np.array(data)
    plt.figure(figsize=(8, 5))
    sns.histplot(data, bins=bins, kde=True, edgecolor="black", color="#72B7B2")
    plt.title(title_name)
    plt.xlabel(xlabel)
    plt.ylabel("Tần suất")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_box(data, title_name: str = "", ylabel: str = "Giá trị"):
    data = np.array(data)
    plt.figure(figsize=(6, 5))
    sns.boxplot(y=data, color="#F58518")
    plt.title(title_name)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()


def hist_and_box(data, bins: int = 20, title: str = "", xlabel: str = "Giá trị"):
    """Vẽ histogram và boxplot song song để xem phân phối + outlier."""
    data = np.array(data)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(data, bins=bins, kde=True, edgecolor="black", color="#72B7B2", ax=axes[0])
    axes[0].set_title(f"Hist - {title}")
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel("Tần suất")
    axes[0].grid(alpha=0.3)

    sns.boxplot(y=data, color="#F58518", ax=axes[1])
    axes[1].set_title(f"Boxplot - {title}")
    axes[1].set_ylabel(xlabel)

    fig.tight_layout()
    plt.show()


def heatmap_corr(corr_matrix, labels, title: str = "Heatmap tương quan"):
    """Vẽ heatmap tương quan cho ma trận corr và danh sách nhãn."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, xticklabels=labels, yticklabels=labels, cmap="coolwarm", center=0, annot=False)
    plt.title(title)
    plt.tight_layout()
    plt.show()
