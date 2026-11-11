
from pathlib import Path

import numpy as np
import streamlit as st


# MODEL CONFIGURATION

MODEL_DIR = Path(__file__).resolve().parent

WEIGHTS_PATH = MODEL_DIR / "weights_softmax_regression.npy"

INPUT_SIZE = 784
NUM_CLASSES = 10
EXPECTED_WEIGHT_SHAPE = (785, 10)


# LOAD MODEL

@st.cache_resource
def load_model():
    """
    Nạp trọng số Softmax Regression đã huấn luyện.

    Trả về:
        weights: numpy array có shape (785, 10).
    """

    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file trọng số: {WEIGHTS_PATH}"
        )

    weights = np.load(WEIGHTS_PATH)

    if weights.shape != EXPECTED_WEIGHT_SHAPE:
        raise ValueError(
            f"Kích thước weights không hợp lệ: {weights.shape}. "
            f"Yêu cầu: {EXPECTED_WEIGHT_SHAPE}"
        )

    if not np.isfinite(weights).all():
        raise ValueError(
            "File trọng số chứa NaN hoặc Inf."
        )

    return weights


# SOFTMAX

def softmax(z):
    """
    Tính Softmax ổn định số học.

    Đầu vào:
        z: vector hoặc ma trận logits.

    Trả về:
        Xác suất các lớp.
    """

    z = np.asarray(z, dtype=np.float64)

    z = z - np.max(z, axis=-1, keepdims=True)

    exp_z = np.exp(z)

    return exp_z / np.sum(
        exp_z,
        axis=-1,
        keepdims=True
    )


# PREDICT

def predict(weights, x):
    """
    Dự đoán chữ số từ vector ảnh đã tiền xử lý.

    Tham số:
        weights:
            Ma trận trọng số shape (785, 10).

        x:
            Vector ảnh shape (1, 784) hoặc (784,).
            Giá trị pixel đã chuẩn hóa về [0, 1].

    Trả về:
        probabilities:
            Vector shape (10,), xác suất của các chữ số 0-9.
    """

    weights = np.asarray(
        weights,
        dtype=np.float64
    )

    x = np.asarray(
        x,
        dtype=np.float64
    )

    # Kiểm tra trọng số
    if weights.shape != EXPECTED_WEIGHT_SHAPE:
        raise ValueError(
            f"Weights phải có shape {EXPECTED_WEIGHT_SHAPE}, "
            f"nhận được {weights.shape}"
        )

    # Đưa input về dạng ma trận (1, 784)
    if x.ndim == 1:
        x = x.reshape(1, -1)

    if x.shape != (1, INPUT_SIZE):
        raise ValueError(
            f"Input phải có shape (1, 784), nhận được {x.shape}"
        )

    if not np.isfinite(x).all():
        raise ValueError(
            "Dữ liệu ảnh chứa NaN hoặc Inf."
        )

    if np.any(x < 0) or np.any(x > 1):
        raise ValueError(
            "Pixel đầu vào phải được chuẩn hóa trong khoảng [0, 1]."
        )

    # Thêm bias vào đầu vector
    bias = np.ones((1, 1), dtype=np.float64)

    x_with_bias = np.concatenate(
        [bias, x],
        axis=1
    )

    # Tính logits
    logits = x_with_bias @ weights

    # Tính xác suất Softmax
    probabilities = softmax(logits)

    # Trả về vector 10 xác suất
    return probabilities.reshape(NUM_CLASSES)


# PREDICT FROM 28x28 IMAGE (OPTIONAL)

def predict_digit(image_28x28):
    """
    Hàm tiện ích nếu muốn dự đoán trực tiếp từ ảnh 28x28.

    Đầu vào:
        Ảnh grayscale 28x28, pixel [0, 255].

    Trả về:
        predicted_digit: chữ số dự đoán.
        probabilities: xác suất 10 lớp.
    """

    image = np.asarray(image_28x28)

    if image.shape != (28, 28):
        raise ValueError(
            f"Ảnh phải có kích thước (28, 28), "
            f"nhận được {image.shape}"
        )

    if not np.isfinite(image).all():
        raise ValueError(
            "Ảnh chứa giá trị không hợp lệ."
        )

    image = image.astype(np.float64) / 255.0

    x = image.reshape(1, INPUT_SIZE)

    weights = load_model()

    probabilities = predict(weights, x)

    predicted_digit = int(np.argmax(probabilities))

    return predicted_digit, probabilities
