
import cv2
import numpy as np

from PIL import (
    Image,
    ImageOps,
    UnidentifiedImageError
)


# CẤU HÌNH

ALLOWED_TYPES = ("png", "jpg", "jpeg")
MAX_SIZE_MB = 5

IMAGE_SIZE = 28
DIGIT_SIZE = 20
THRESHOLD_VALUE = 30


# CUSTOM EXCEPTION

class InvalidImageError(Exception):
    """Lỗi ảnh đầu vào không hợp lệ."""


# ĐỌC VÀ KIỂM TRA ẢNH UPLOAD

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


# CHUYỂN DỮ LIỆU CANVAS THÀNH ẢNH PIL

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


# TIỀN XỬ LÝ ẢNH THEO CHUẨN MNIST

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
        # 1. Xử lý ảnh RGBA và nền trong suốt
        if img.mode == "RGBA":
            if from_canvas:
                bg = Image.new("RGBA", img.size, (0, 0, 0, 255))
            else:
                bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(bg, img)
        
        # Chuyển đổi sang định dạng numpy Grayscale
        rgb_arr = np.array(img.convert("RGB"))
        gray = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape

        # 2. Xử lý trường hợp ảnh mẫu chuẩn kích thước 28x28
        if (w, h) == (IMAGE_SIZE, IMAGE_SIZE):
            border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
            if np.median(border) > 127:
                gray = 255 - gray
            canvas = gray
        else:
            # 3. Phân đoạn nét chữ số (Segmentation) bằng OpenCV
            if from_canvas:
                # Bảng vẽ Canvas: Nền đen (0), nét vẽ trắng (255)
                border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
                if np.median(border) > 127:
                    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
                else:
                    binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)[1]
                
                # Khử nhiễu các chấm nhỏ lẻ
                kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_clean)

            else:
                # Ảnh chụp tải lên từ điện thoại / máy quét / máy ảnh
                border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
                med_border = np.median(border)

                if med_border > 90:
                    # Nền giấy sáng (chữ tối trên giấy trắng):
                    # Khử bóng đổ và cân bằng ánh sáng bằng Morphological Closing
                    ksize = max(15, (min(w, h) // 8) * 2 + 1)
                    kernel_bg = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
                    bg = cv2.morphologyEx(gray, cv2.MORPH_DILATE, kernel_bg)
                    norm = np.clip((gray.astype(np.float32) / np.maximum(bg.astype(np.float32), 1.0)) * 255.0, 0, 255).astype(np.uint8)

                    # Phân ngưỡng thích nghi (Adaptive Gaussian Thresholding)
                    bsize = max(11, (min(w, h) // 16) * 2 + 1)
                    binary = cv2.adaptiveThreshold(
                        norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, bsize, 12
                    )
                else:
                    # Ảnh nền tối
                    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

                # Loại bỏ viền sát mép ảnh (tránh mép bàn hoặc bóng viền ảnh)
                margin = max(2, min(w, h) // 100)
                binary[:margin, :] = 0
                binary[-margin:, :] = 0
                binary[:, :margin] = 0
                binary[:, -margin:] = 0

                # Lọc thành phần liên thông (Connected Components)
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
                clean_binary = np.zeros_like(binary)
                total_area = w * h

                for i in range(1, num_labels):
                    x, y, sw, sh, area = stats[i]
                    # Bỏ các vệt nhiễu quá nhỏ hoặc nền chiếm quá nửa bức ảnh
                    if area < 15 or area > 0.45 * total_area:
                        continue
                    aspect = sw / float(sh) if sh > 0 else 0
                    # Loại bỏ các thanh dài nằm ngang hoặc dọc sát rìa (mép bàn, cạnh giấy)
                    if sw > 0.65 * w or sh > 0.85 * h:
                        continue
                    if aspect > 5.5 or aspect < 0.08:
                        continue
                    clean_binary[labels == i] = 255

                if np.sum(clean_binary > 0) > 30:
                    binary = clean_binary

            # 4. Tìm khung bao quanh chữ số (Bounding Box)
            ys, xs = np.where(binary > 30)

            if len(xs) == 0 or len(ys) == 0:
                # Thử lại với Otsu đơn giản nếu chưa tìm thấy
                _, fallback = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                ys, xs = np.where(fallback > 30)
                if len(xs) == 0 or len(ys) == 0:
                    raise InvalidImageError("Không tìm thấy nét chữ số rõ ràng trong ảnh.")
                binary = fallback

            x_min = int(xs.min())
            x_max = int(xs.max() + 1)
            y_min = int(ys.min())
            y_max = int(ys.max() + 1)

            digit_patch = binary[y_min:y_max, x_min:x_max]
            pw = digit_patch.shape[1]
            ph = digit_patch.shape[0]

            # 5. Resize giữ nguyên tỷ lệ trong khung 20x20
            if pw > ph:
                new_w = DIGIT_SIZE
                new_h = max(1, int(round(ph * float(DIGIT_SIZE) / float(pw))))
            else:
                new_h = DIGIT_SIZE
                new_w = max(1, int(round(pw * float(DIGIT_SIZE) / float(ph))))

            # Dùng INTER_AREA để làm mượt cạnh nét vẽ (anti-aliasing)
            resized = cv2.resize(digit_patch, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Tự động cân bằng độ đậm nét chữ (Stroke Thickness Normalization)
            # Chuẩn MNIST có khoảng 50 - 110 pixel sáng trong vùng 20x20
            stroke_pixels = np.sum(resized > 30)
            if stroke_pixels < 55:
                # Nét bút quá mảnh (ví dụ bút bi trên ảnh độ phân giải cao), giãn nhẹ nét
                resized = cv2.dilate(resized, np.ones((2, 2), np.uint8), iterations=1)

            # 6. Đặt chữ số vào giữa canvas 28x28
            canvas = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
            x_offset = (IMAGE_SIZE - new_w) // 2
            y_offset = (IMAGE_SIZE - new_h) // 2
            canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

            # Tinh chỉnh vị trí theo Trọng tâm (Center of Mass) chuẩn MNIST
            total_mass = float(np.sum(canvas))
            if total_mass > 0:
                y_idx, x_idx = np.indices(canvas.shape)
                cy = float(np.sum(y_idx * canvas) / total_mass)
                cx = float(np.sum(x_idx * canvas) / total_mass)
                shift_y = int(round(13.5 - cy))
                shift_x = int(round(13.5 - cx))

                # Giới hạn dịch chuyển tối đa 3 pixel để không làm méo đặc trưng của 4 và 9
                shift_y = max(-3, min(3, shift_y))
                shift_x = max(-3, min(3, shift_x))

                if shift_y != 0 or shift_x != 0:
                    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
                    canvas = cv2.warpAffine(canvas, M, (IMAGE_SIZE, IMAGE_SIZE))

        # 7. Chuẩn hóa pixel về [0, 1]
        normalized = canvas.astype(np.float64) / 255.0

        # 8. Flatten thành vector 784 đặc trưng
        image_vector = normalized.reshape(1, IMAGE_SIZE * IMAGE_SIZE)
        processed_pil = Image.fromarray(canvas)

        return image_vector, processed_pil

    except InvalidImageError:
        raise

    except Exception as e:
        raise InvalidImageError(
            f"Không thể tiền xử lý ảnh: {str(e)}"
        ) from e
