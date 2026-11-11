
import numpy as np

from PIL import (
    Image,
    ImageOps,
    UnidentifiedImageError
)


# =========================================================
# CẤU HÌNH
# =========================================================

ALLOWED_TYPES = ("png", "jpg", "jpeg")
MAX_SIZE_MB = 5

IMAGE_SIZE = 28
DIGIT_SIZE = 20
THRESHOLD_VALUE = 30


# =========================================================
# CUSTOM EXCEPTION
# =========================================================

class InvalidImageError(Exception):
    """Lỗi ảnh đầu vào không hợp lệ."""


# =========================================================
# ĐỌC VÀ KIỂM TRA ẢNH UPLOAD
# =========================================================

def load_uploaded_image(uploaded_file) -> Image.Image:
    """
    Kiểm tra và đọc ảnh người dùng tải lên.

    Hỗ trợ:
        PNG, JPG, JPEG.
        Dung lượng tối đa 5MB.

    Trả về:
        Ảnh PIL ở chế độ RGBA.
    """

    if uploaded_file is None:
        raise InvalidImageError(
            "Bạn chưa tải ảnh lên."
        )

    # Kiểm tra định dạng file
    filename = getattr(uploaded_file, "name", "")

    ext = (
        filename.rsplit(".", 1)[-1].lower()
        if "." in filename
        else ""
    )

    if ext not in ALLOWED_TYPES:
        raise InvalidImageError(
            "Ảnh sai định dạng. Chỉ hỗ trợ PNG, JPG, JPEG."
        )

    # Kiểm tra dung lượng
    file_size = getattr(uploaded_file, "size", 0)

    if file_size > MAX_SIZE_MB * 1024 * 1024:
        raise InvalidImageError(
            f"Ảnh quá lớn (tối đa {MAX_SIZE_MB}MB)."
        )

    try:
        uploaded_file.seek(0)

        img = Image.open(uploaded_file)

        # Áp dụng hướng ảnh từ EXIF nếu có
        img = ImageOps.exif_transpose(img)

        # Đọc toàn bộ dữ liệu ảnh
        img.load()

        return img.convert("RGBA")

    except (
        UnidentifiedImageError,
        OSError,
        ValueError
    ) as e:
        raise InvalidImageError(
            "Không đọc được ảnh. File có thể bị hỏng."
        ) from e


# =========================================================
# CHUYỂN DỮ LIỆU CANVAS THÀNH ẢNH PIL
# =========================================================

def canvas_to_image(image_data) -> Image.Image:
    """
    Chuyển dữ liệu canvas dạng numpy RGBA sang PIL.

    Kiểm tra vùng vẽ có chứa nét chữ hay chưa.
    """

    if image_data is None:
        raise InvalidImageError(
            "Vùng vẽ chưa có nội dung. Hãy vẽ một chữ số."
        )

    arr = np.asarray(image_data)

    if arr.ndim != 3 or arr.shape[2] < 3:
        raise InvalidImageError(
            "Dữ liệu canvas không hợp lệ."
        )

    arr = np.clip(arr, 0, 255).astype(np.uint8)

    # Chỉ lấy 3 kênh RGB
    rgb = arr[:, :, :3]

    # Kiểm tra sự khác biệt giữa các pixel và nền.
    # Canvas thông thường có nền đen và nét vẽ màu trắng.
    brightness = rgb.mean(axis=2)

    if brightness.max() < 30:
        raise InvalidImageError(
            "Vùng vẽ chưa có nội dung. Hãy vẽ một chữ số."
        )

    return Image.fromarray(rgb).convert("RGBA")


# =========================================================
# TIỀN XỬ LÝ ẢNH THEO CHUẨN MNIST
# =========================================================

def preprocess_image(
    img: Image.Image,
    from_canvas: bool = False
):
    """
    Tiền xử lý ảnh để đưa vào mô hình Softmax Regression.

    Các bước:
        1. Chuyển ảnh về grayscale.
        2. Xử lý nền trắng/nền đen.
        3. Đảo màu nếu cần để có chữ trắng, nền đen.
        4. Crop vùng chứa chữ số.
        5. Resize giữ tỷ lệ trong vùng 20x20.
        6. Căn giữa trên canvas 28x28.
        7. Chuẩn hóa pixel về [0, 1].
        8. Flatten thành vector 784 phần tử.

    Tham số:
        img:
            Ảnh PIL đầu vào.

        from_canvas:
            True nếu ảnh đến từ vùng vẽ canvas.

    Trả về:
        image_vector:
            numpy array có shape (1, 784), giá trị [0, 1].

        processed_image:
            Ảnh PIL grayscale kích thước 28x28.
    """

    if img is None:
        raise InvalidImageError(
            "Không có ảnh đầu vào."
        )

    if not isinstance(img, Image.Image):
        raise InvalidImageError(
            "Dữ liệu đầu vào không phải ảnh PIL."
        )

    try:
        # -------------------------------------------------
        # 1. Xử lý ảnh RGBA và nền trong suốt
        # -------------------------------------------------

        if img.mode == "RGBA":

            if from_canvas:
                # Canvas có nền đen trong suốt hoặc đen.
                # Ghép lên nền đen để giữ nét trắng.
                background = Image.new(
                    "RGBA",
                    img.size,
                    (0, 0, 0, 255)
                )

            else:
                # Ảnh upload PNG trong suốt:
                # ghép lên nền trắng.
                background = Image.new(
                    "RGBA",
                    img.size,
                    (255, 255, 255, 255)
                )

            img = Image.alpha_composite(
                background,
                img
            )

        else:
            img = img.convert("RGB")

        # -------------------------------------------------
        # 2. Chuyển sang grayscale
        # -------------------------------------------------

        gray = img.convert("L")

        arr = np.asarray(
            gray,
            dtype=np.uint8
        )

        # -------------------------------------------------
        # 3. Xác định màu nền và đảo màu nếu cần
        # -------------------------------------------------

        # Lấy các pixel ở vùng viền ảnh
        border = np.concatenate([
            arr[0, :],
            arr[-1, :],
            arr[:, 0],
            arr[:, -1]
        ])

        background_brightness = np.median(border)

        # MNIST: nền đen, chữ trắng.
        # Nếu nền sáng thì đảo màu.
        if background_brightness > 127:
            gray = ImageOps.invert(gray)

        arr = np.asarray(
            gray,
            dtype=np.uint8
        )

        # -------------------------------------------------
        # 4. Tìm vùng chứa chữ số và căn chỉnh chuẩn MNIST
        # -------------------------------------------------

        if gray.size == (IMAGE_SIZE, IMAGE_SIZE):
            canvas = gray
        else:
            ys, xs = np.where(
                arr > THRESHOLD_VALUE
            )

            if len(xs) == 0 or len(ys) == 0:
                raise InvalidImageError(
                    "Không tìm thấy nét chữ số trong ảnh."
                )

            # Crop sát vùng chữ số
            x_min = int(xs.min())
            x_max = int(xs.max() + 1)

            y_min = int(ys.min())
            y_max = int(ys.max() + 1)

            w = x_max - x_min
            h = y_max - y_min

            digit = gray.crop((x_min, y_min, x_max, y_max))

            # -------------------------------------------------
            # 5. Resize giữ nguyên tỷ lệ trong vùng 20x20
            # -------------------------------------------------
            if w > h:
                new_w = DIGIT_SIZE
                new_h = max(1, int(round(h * float(DIGIT_SIZE) / float(w))))
            else:
                new_h = DIGIT_SIZE
                new_w = max(1, int(round(w * float(DIGIT_SIZE) / float(h))))

            digit = digit.resize((new_w, new_h), Image.Resampling.BICUBIC)

            # -------------------------------------------------
            # 6. Đặt chữ số vào giữa canvas 28x28
            # -------------------------------------------------
            canvas = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), color=0)
            x_offset = (IMAGE_SIZE - new_w) // 2
            y_offset = (IMAGE_SIZE - new_h) // 2
            canvas.paste(digit, (x_offset, y_offset))

            # Tinh chỉnh vị trí theo Trọng tâm (Center of Mass) chuẩn MNIST
            c_arr = np.asarray(canvas, dtype=np.float32)
            total = float(np.sum(c_arr))
            if total > 0:
                y_idx, x_idx = np.indices(c_arr.shape)
                cy = float(np.sum(y_idx * c_arr) / total)
                cx = float(np.sum(x_idx * c_arr) / total)
                shift_y = int(round(13.5 - cy))
                shift_x = int(round(13.5 - cx))

                if abs(shift_y) <= 4 and abs(shift_x) <= 4 and (shift_y != 0 or shift_x != 0):
                    shifted = np.zeros_like(c_arr)
                    sy1, sy2 = max(0, -shift_y), min(IMAGE_SIZE, IMAGE_SIZE - shift_y)
                    sx1, sx2 = max(0, -shift_x), min(IMAGE_SIZE, IMAGE_SIZE - shift_x)
                    dy1, dy2 = max(0, shift_y), min(IMAGE_SIZE, IMAGE_SIZE + shift_y)
                    dx1, dx2 = max(0, shift_x), min(IMAGE_SIZE, IMAGE_SIZE + shift_x)
                    h_overlap = min(sy2 - sy1, dy2 - dy1)
                    w_overlap = min(sx2 - sx1, dx2 - dx1)
                    if h_overlap > 0 and w_overlap > 0:
                        shifted[dy1:dy1 + h_overlap, dx1:dx1 + w_overlap] = c_arr[sy1:sy1 + h_overlap, sx1:sx1 + w_overlap]
                        canvas = Image.fromarray(np.clip(shifted, 0, 255).astype(np.uint8))

        # -------------------------------------------------
        # 7. Chuẩn hóa pixel về [0, 1]
        # -------------------------------------------------

        normalized = (
            np.asarray(
                canvas,
                dtype=np.float64
            ) / 255.0
        )

        # -------------------------------------------------
        # 8. Flatten thành vector 784 đặc trưng
        # -------------------------------------------------

        image_vector = normalized.reshape(
            1,
            IMAGE_SIZE * IMAGE_SIZE
        )

        return image_vector, canvas

    except InvalidImageError:
        raise

    except Exception as e:
        raise InvalidImageError(
            f"Không thể tiền xử lý ảnh: {str(e)}"
        ) from e
