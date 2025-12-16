import os
import numpy as np


def load_raw_data(path):
    """
    Đọc file CSV thô bằng numpy.
    Trả về structured array với tên cột.
    """
    data = np.genfromtxt(
        path,
        delimiter=",",
        dtype=None,
        names=True,
        encoding="utf-8",
    )
    return data


def clean_strings(data):
    """
    Loại bỏ dấu ngoặc kép thừa trong các cột kiểu chuỗi.
    Trả về bản sao mới của data.
    """
    cleaned_rows = []
    for row in data:
        new_row = []
        for v in row:
            if isinstance(v, str):
                new_row.append(v.replace('"', ""))
            else:
                new_row.append(v)
        cleaned_rows.append(tuple(new_row))
    return np.array(cleaned_rows, dtype=data.dtype)


def drop_leakage_and_id(data):
    """
    Bỏ cột ID và 2 cột Naive_Bayes_* (data leakage),
    giống như trong 01_data_exploration.
    """
    leak_cols = [
        "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1",
        "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2",
    ]
    drop_cols = ["CLIENTNUM"] + leak_cols
    keep_cols = [c for c in data.dtype.names if c not in drop_cols]
    return data[keep_cols]


def split_numeric_categorical(data):
    """
    Tách tên cột số và cột phân loại dựa vào dtype.kind.
    """
    numeric_cols = [name for name in data.dtype.names if data[name].dtype.kind in {"i", "f"}]
    cat_cols = [name for name in data.dtype.names if data[name].dtype.kind not in {"i", "f"}]
    return numeric_cols, cat_cols


def detect_outliers_iqr(col_values, factor=1.5):
    """
    Phát hiện outlier bằng IQR cho 1 vector số.
    Trả về mask boolean (True nếu là outlier).
    """
    x = col_values.astype(float)
    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    mask_outlier = (x < lower) | (x > upper)
    return mask_outlier, lower, upper


def remove_outliers_iqr(data, col_names, factor=1.5):
    """
    Loại bỏ hàng có outlier (theo IQR) trên một số cột được chọn.
    Chỉ nên dùng nếu thật sự cần (ví dụ loại bỏ vài outlier cực đoan).
    """
    mask_keep = np.ones(len(data), dtype=bool)
    for name in col_names:
        mask_out, _, _ = detect_outliers_iqr(data[name], factor=factor)
        mask_keep &= ~mask_out
    return data[mask_keep]


def min_max_normalize(col_values):
    """
    Chuẩn hóa min-max cho 1 vector số về [0, 1].
    """
    x = col_values.astype(float)
    min_v = np.min(x)
    max_v = np.max(x)
    if max_v == min_v:
        return np.zeros_like(x)
    return (x - min_v) / (max_v - min_v)


def log_transform(col_values, eps=1e-6):
    """
    Log transform đơn giản cho dữ liệu dương (thêm eps để tránh log(0)).
    """
    x = col_values.astype(float)
    x = np.where(x < 0, 0, x)
    return np.log(x + eps)


def decimal_scaling(col_values):
    """
    Decimal scaling: chia cho 10^k sao cho giá trị tuyệt đối < 1.
    """
    x = col_values.astype(float)
    max_abs = np.max(np.abs(x))
    if max_abs == 0:
        return x
    k = len(str(int(max_abs)))
    return x / (10 ** k)


def standardize_zscore(col_values):
    """
    Chuẩn hóa z-score: (x - mean) / std.
    """
    x = col_values.astype(float)
    mean = np.mean(x)
    std = np.std(x)
    if std == 0:
        return np.zeros_like(x)
    return (x - mean) / std


def preprocess_full(path_csv):
    """
    Pipeline đơn giản:
    - Load
    - Clean string
    - Drop ID + leakage
    - Tách numeric / categorical
    - Trả về data đã clean + list cột numeric/categorical
    """
    data = load_raw_data(path_csv)
    data = clean_strings(data)
    data = drop_leakage_and_id(data)
    numeric_cols, cat_cols = split_numeric_categorical(data)
    return data, numeric_cols, cat_cols


