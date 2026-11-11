
from pathlib import Path
import numpy as np


# =========================================================
# MODEL CONFIGURATION
# =========================================================

# Thư mục chứa file model
MODEL_DIR = Path(__file__).resolve().parent

# Đường dẫn đến file trọng số
WEIGHTS_PATH = MODEL_DIR / "weights_softmax_regression.npy"


# =========================================================
# INPUT CONFIGURATION
# =========================================================

IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28

INPUT_SIZE = IMAGE_HEIGHT * IMAGE_WIDTH  # 784
NUM_CLASSES = 10

# Model có thêm 1 bias ở đầu vector
EXPECTED_WEIGHT_SHAPE = (INPUT_SIZE + 1, NUM_CLASSES)


# =========================================================
# MODEL INFORMATION
# =========================================================

MODEL_NAME = "Softmax Regression"
MODEL_VERSION = "1.0"

MODEL_DESCRIPTION = (
    "Mô hình nhận diện chữ số viết tay từ 0 đến 9 "
    "sử dụng thuật toán Softmax Regression huấn luyện trên bộ dữ liệu MNIST."
)

MODEL_INFO = {
    "Thuật toán": "Softmax Regression (Multiclass Logistic Regression)",
    "Bộ dữ liệu": "MNIST (60,000 huấn luyện/validation, 10,000 kiểm thử)",
    "Độ chính xác (Test Accuracy)": "89.04% trên tập Testing độc lập",
    "F1-Score trung bình": "0.886 (Macro) / 0.886 (Weighted)",
    "Kích thước đầu vào": "28 × 28 Grayscale (784 pixel + 1 bias = 785 đặc trưng)",
    "Số lớp phân loại": "10 chữ số (từ 0 đến 9)",
    "Hàm kích hoạt": "Softmax ổn định số học (trừ max logits)",
    "Hàm mất mát": "Cross-Entropy Loss (hội tụ về ~0.47)",
    "Phương pháp tối ưu": "Gradient Descent (Learning Rate = 0.15, 150 epochs)",
    "Chuẩn hóa dữ liệu": "Pixel uint8 [0, 255] chuẩn hóa về float [0, 1]"
}

# =========================================================
# MODEL STATUS
# =========================================================

def check_model_ready() -> bool:
    """
    Kiểm tra file trọng số có tồn tại và đúng định dạng không.

    Trả về:
        True: Model sẵn sàng.
        False: Model chưa sẵn sàng.
    """

    if not WEIGHTS_PATH.is_file():
        return False

    try:
        weights = np.load(
            WEIGHTS_PATH,
            allow_pickle=False,
            mmap_mode="r"
        )

        if weights.shape != EXPECTED_WEIGHT_SHAPE:
            return False

        if not np.issubdtype(weights.dtype, np.number):
            return False

        if not np.isfinite(weights).all():
            return False

        return True

    except (OSError, ValueError, TypeError):
        return False


# Kiểm tra trạng thái model khi ứng dụng khởi động
MODEL_READY = check_model_ready()
