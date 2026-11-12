import os
from pathlib import Path
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

try:
    from streamlit_drawable_canvas import st_canvas
except Exception:
    st_canvas = None

from model.config import MODEL_READY, WEIGHTS_PATH
from model.predict import load_model, predict
from utils.image_processing import (
    InvalidImageError,
    canvas_to_image,
    load_uploaded_image,
    preprocess_image,
)
from utils.styles import header, icon_title


# Các cặp chữ số thường bị nhận diện nhầm từ Confusion Matrix trong notebook
CONFUSED_PAIRS = {
    (4, 9): "Số 4 và số 9 có hình thái nét thẳng phía trên rất tương đồng với vòng khép kín trong không gian tuyến tính.",
    (9, 4): "Số 9 dễ bị mô hình tuyến tính phân loại thành số 4 do nét vòng phía trên không phân tách rõ với nét sổ thẳng.",
    (5, 3): "Số 5 và số 3 có chung phần bụng cong phía dưới, dễ gây nhầm lẫn khi nét ngang trên bị mờ.",
    (3, 5): "Số 3 và số 5 có kích thước và vị trí điểm ảnh kích hoạt ở nửa dưới rất giống nhau.",
    (5, 8): "Số 5 và số 8 đều có phần cong ở thân và đáy, dễ nhầm khi nét số 5 khép hẹp.",
    (8, 5): "Số 8 bị hở một bên dễ bị mô hình dự đoán thành số 5.",
    (2, 8): "Số 2 và số 8 có các điểm ảnh giao cắt ở vùng trung tâm và nét ngang phía dưới.",
    (7, 9): "Số 7 và số 9 dễ nhầm lẫn nếu nét gạch ngang trên của số 7 nối liền với nét xổ cong.",
    (7, 1): "Số 7 vẽ thẳng không có nét gạch ngang giữa rất dễ bị nhận diện thành số 1."
}


def render():
    header(
        "DIGIT RECOGNITION",
        "Nhận diện chữ số viết tay",
        "Tải ảnh từ máy, tự tay vẽ lên bảng tương tác hoặc chọn ảnh mẫu MNIST để trải nghiệm mô hình AI."
    )

    if not MODEL_READY:
        st.error(
            f"Không tìm thấy file trọng số mô hình tại `{WEIGHTS_PATH}`. "
            "Vui lòng đảm bảo file `weights_softmax_regression.npy` đã được huấn luyện và đặt trong thư mục `model/`.",
            icon=":material/error:"
        )
        return

    # Nạp mô hình trọng số
    try:
        weights = load_model()
    except Exception as e:
        st.error(f"Lỗi nạp mô hình: {e}", icon=":material/error:")
        return

    col_input, col_result = st.columns([1.1, 1], gap="large")

    with col_input:
        with st.container(border=True):
            icon_title("input-cursor-text", "1. Cung cấp chữ số")
            st.markdown(
                "<span style='color:#64748B;font-size:13px;'>"
                "Chọn 1 trong 3 phương thức bên dưới để đưa chữ số vào hệ thống:"
                "</span>",
                unsafe_allow_html=True
            )

            input_mode = st.radio(
                "Chọn phương thức nhập:",
                options=["Tải ảnh lên", "Vẽ trực tiếp", "Chọn ảnh mẫu"],
                horizontal=True,
                label_visibility="collapsed"
            )

            raw_image = None
            is_from_canvas = False
            source_description = ""


            # PHƯƠNG THỨC 1: UPLOAD ẢNH

            if input_mode == "Tải ảnh lên":
                st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
                uploaded_file = st.file_uploader(
                    "Chọn file ảnh chứa 1 chữ số (PNG, JPG, JPEG):",
                    type=["png", "jpg", "jpeg"],
                    key="digit_uploader",
                    help="Hỗ trợ tối đa 5MB. Ảnh rõ nét, chữ số nằm ở trung tâm."
                )
                if uploaded_file is not None:
                    # Kiểm tra định dạng tệp hợp lệ trước khi xử lý
                    allowed_ext = {"png", "jpg", "jpeg"}
                    file_ext = uploaded_file.name.lower().split('.')[-1]
                    if file_ext not in allowed_ext:
                        st.warning(
                            f"Định dạng tệp `{uploaded_file.name}` không hợp lệ. Vui lòng tải ảnh PNG, JPG hoặc JPEG.",
                            icon=":material/warning:"
                        )
                    else:
                        try:
                            raw_image = load_uploaded_image(uploaded_file)
                            source_description = f"Ảnh tải lên: `{uploaded_file.name}`"
                            st.image(raw_image, caption="Ảnh gốc tải lên", width=200)
                        except InvalidImageError as err:
                            st.warning(str(err), icon=":material/warning:")


            # PHƯƠNG THỨC 2: VẼ TRỰC TIẾP (CANVAS)

            elif input_mode == "Vẽ trực tiếp":
                st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
                current_canvas = st_canvas
                if current_canvas is None:
                    try:
                        from streamlit_drawable_canvas import st_canvas as current_canvas
                    except Exception:
                        current_canvas = None

                if current_canvas is None:
                    st.warning("Bảng vẽ tương tác chưa thể tải trên môi trường hiện tại.", icon=":material/warning:")
                else:
                    st.markdown(
                        "<span style='font-size:13px;color:#475569;'>"
                        "Vẽ chữ số màu trắng trên nền đen bằng chuột hoặc màn hình cảm ứng:"
                        "</span>",
                        unsafe_allow_html=True
                    )

                    c_slider, c_btn = st.columns([2, 1])
                    with c_slider:
                        stroke_width = st.slider(
                            "Độ dày nét cọ (px):",
                            min_value=14,
                            max_value=32,
                            value=22,
                            step=2,
                            help="Khuyến nghị 20–24px để nét chữ khi thu nhỏ về 28×28 đạt độ dày chuẩn MNIST.",
                            key="canvas_stroke_width"
                        )
                    with c_btn:
                        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
                        clear_canvas = st.button("Xóa nét vẽ", icon=":material/delete:", key="btn_clear_canvas", use_container_width=True)

                    st.caption("Khuyến nghị: Để nét cọ **20–24px**, vẽ chữ số to rõ ở giữa khung hình để mô hình nhận diện chính xác nhất.")

                    if "canvas_version" not in st.session_state:
                        st.session_state["canvas_version"] = 0

                    if clear_canvas:
                        st.session_state["canvas_version"] += 1
                        st.session_state["canvas_predict_clicked"] = False
                        st.rerun()

                    canvas_key = f"digit_canvas_v_{st.session_state['canvas_version']}"

                    canvas_result = current_canvas(
                        fill_color="rgba(255, 255, 255, 0)",
                        stroke_width=stroke_width,
                        stroke_color="#FFFFFF",
                        background_color="#000000",
                        height=280,
                        width=280,
                        drawing_mode="freedraw",
                        key=canvas_key,
                        return_image_data=True,
                    )

                    canvas_img_data = None
                    if canvas_result is not None:
                        try:
                            canvas_img_data = canvas_result.image_data
                        except (RuntimeError, AttributeError):
                            canvas_img_data = None

                    has_strokes = False
                    if canvas_img_data is not None:
                        arr = np.asarray(canvas_img_data)
                        if arr.ndim == 3 and arr[:, :, :3].mean() > 5:
                            has_strokes = True
                            try:
                                raw_image = canvas_to_image(canvas_img_data)
                                is_from_canvas = True
                                source_description = "Bảng vẽ trực tiếp (Interactive Canvas)"
                            except InvalidImageError:
                                raw_image = None

                    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
                    btn_predict = st.button(
                        "Bắt đầu nhận diện",
                        icon=":material/play_arrow:",
                        type="primary",
                        key="btn_predict_canvas",
                        use_container_width=True,
                        disabled=not has_strokes
                    )
                    if btn_predict:
                        st.session_state["canvas_predict_clicked"] = True


            # PHƯƠNG THỨC 3: CHỌN ẢNH MẪU MNIST

            elif input_mode == "Chọn ảnh mẫu":
                st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
                st.markdown(
                    "<span style='font-size:13px;color:#475569;'>"
                    "Chọn một chữ số chuẩn từ bộ dữ liệu kiểm thử MNIST để thử nghiệm nhanh:"
                    "</span>",
                    unsafe_allow_html=True
                )

                sample_dir = Path("assets/sample_images")
                selected_digit = st.selectbox(
                    "Chọn chữ số mẫu:",
                    options=list(range(10)),
                    format_func=lambda x: f"Chữ số mẫu: {x}",
                    key="sample_digit_select"
                )

                sample_path = sample_dir / f"digit_{selected_digit}.png"
                if sample_path.exists():
                    sample_pil = Image.open(sample_path).convert("RGBA")
                    st.image(
                        sample_pil,
                        caption=f"Ảnh mẫu chữ số {selected_digit} (MNIST Testing Set)",
                        width=140
                    )
                    raw_image = sample_pil
                    source_description = f"Ảnh mẫu chuẩn MNIST: Chữ số {selected_digit}"

            st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
            with st.expander("Mẹo nhận diện chính xác nhất (Đặc biệt số 4 & 9)", expanded=False):
                st.markdown(
                    """
                    <div style="font-size:13px; line-height:1.6; color:#334155;">
                        <p style="margin-bottom:6px;"><strong><i class="bi bi-lightbulb" style="color:#F59E0B;"></i> Đặc trưng mô hình Hồi quy Tuyến tính:</strong><br>
                        Softmax Regression phân loại dựa trên trọng số từng pixel. Chữ số 4 và 9 có độ tương đồng không gian rất cao:
                        </p>
                        <ul style="padding-left:18px; margin-bottom:4px;">
                            <li><strong>Chữ số 4:</strong> Nên viết <em>hở đỉnh</em> (nét gập góc và nét sổ đứng cắt qua). Tránh khép kín chóp tam giác vì đỉnh nhọn kín sẽ kích hoạt vùng trọng số vòng lặp của số 9.</li>
                            <li><strong>Chữ số 9:</strong> Viết vòng tròn khép kín ở trên và <em>nét sổ thẳng đứng xuống</em> (tránh uốn cong đuôi sang trái như chữ 'g' vì dễ kích hoạt số 3).</li>
                            <li><strong>Ảnh chụp điện thoại:</strong> Hệ thống đã tự động khử bóng đổ và viền mép bàn, nên chụp chữ viết rõ nét và đủ sáng.</li>
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


    # CỘT PHẢI: KẾT QUẢ DỰ ĐOÁN & PHÂN TÍCH

    with col_result:
        with st.container(border=True):
            icon_title("cpu-fill", "2. Kết quả nhận diện & Phân tích")

            if input_mode == "Vẽ trực tiếp" and not st.session_state.get("canvas_predict_clicked", False):
                if raw_image is not None:
                    st.info(
                        "Bạn đang vẽ chữ số... Hãy tiếp tục vẽ đầy đủ các nét (ví dụ số 4, 5, 7...), sau đó nhấn nút **Bắt đầu nhận diện** ở bên dưới bảng vẽ để xem kết quả.",
                        icon=":material/draw:"
                    )
                else:
                    st.info(
                        "Hãy vẽ một chữ số lên bảng vẽ bên trái và nhấn **Bắt đầu nhận diện**.",
                        icon=":material/draw:"
                    )
                return

            if raw_image is None:
                st.info(
                    "Hãy tải ảnh lên, vẽ chữ số trên bảng vẽ hoặc chọn ảnh mẫu để hệ thống bắt đầu nhận diện.",
                    icon=":material/info:"
                )
                return

            st.caption(f"Nguồn dữ liệu: **{source_description}**")

            # Tiền xử lý ảnh
            with st.spinner("Đang tiền xử lý ảnh và căn chỉnh chuẩn MNIST..."):
                try:
                    img_vector, processed_canvas = preprocess_image(
                        raw_image,
                        from_canvas=is_from_canvas
                    )
                except InvalidImageError as e:
                    st.warning(str(e), icon=":material/warning:")
                    return
                except Exception as e:
                    st.error(f"Lỗi tiền xử lý ảnh: {e}", icon=":material/error:")
                    return

            # Dự đoán
            try:
                probabilities = predict(weights, img_vector)
                predicted_digit = int(np.argmax(probabilities))
                confidence = float(probabilities[predicted_digit] * 100.0)
            except Exception as e:
                st.error(f"Lỗi trong quá trình dự đoán mô hình: {e}")
                return

            # Hiển thị kết quả chính
            r1, r2 = st.columns([1.1, 0.9], gap="medium")
            with r1:
                st.markdown(
                    f'<div class="result-card-container">'
                    f'<div class="result-glow-tag">CHỮ SỐ DỰ ĐOÁN</div>'
                    f'<div class="result-glow-digit">{predicted_digit}</div>'
                    f'<div class="result-confidence-pill">'
                    f'<i class="bi bi-shield-check" style="font-size:15px;"></i> Độ tin cậy: {confidence:.2f}%'
                    f'</div></div>',
                    unsafe_allow_html=True
                )

            with r2:
                st.markdown(
                    "<div style='font-size:12px;font-weight:700;color:#64748B;letter-spacing:0.5px;margin-bottom:8px;text-transform:uppercase;'>"
                    "<i class='bi bi-bounding-box' style='margin-right:6px;'></i>Ảnh chuẩn hóa 28×28"
                    "</div>",
                    unsafe_allow_html=True
                )
                st.image(
                    processed_canvas,
                    caption="Định dạng Grayscale chuẩn MNIST",
                    width=135
                )

            st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

            # Top 3 ứng viên
            top3_indices = np.argsort(probabilities)[::-1][:3]
            st.markdown(
                "<div style='font-size:13px;font-weight:700;color:#1E293B;margin-bottom:8px;display:flex;align-items:center;'>"
                "<i class='bi bi-trophy' style='color:#F59E0B;margin-right:8px;'></i>Top 3 chữ số tiềm năng nhất:"
                "</div>",
                unsafe_allow_html=True
            )
            top_cols = st.columns(3)
            for rank, (col, idx) in enumerate(zip(top_cols, top3_indices), start=1):
                prob_val = probabilities[idx] * 100
                with col:
                    is_winner = (rank == 1)
                    badge_bg = "linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%)" if is_winner else "#F8FAFC"
                    badge_border = "#BFDBFE" if is_winner else "#E2E8F0"
                    digit_color = "#1D4ED8" if is_winner else "#475569"
                    rank_color = "#2563EB" if is_winner else "#94A3B8"
                    st.markdown(
                        f'<div style="background:{badge_bg};border:1.5px solid {badge_border};border-radius:12px;padding:10px 8px;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,0.03);">'
                        f'<span style="font-size:11px;color:{rank_color};font-weight:800;letter-spacing:0.5px;">HẠNG #{rank}</span>'
                        f'<div style="font-size:24px;font-weight:800;color:{digit_color};margin:2px 0;">Số {idx}</div>'
                        f'<div style="font-size:12.5px;color:#334155;font-weight:700;">{prob_val:.2f}%</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

            st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)

            # Biểu đồ phân phối xác suất 10 lớp
            icon_title("bar-chart-steps", "Phân phối xác suất chi tiết 10 lớp (0–9)", size="h4")
            df_probs = pd.DataFrame({
                "Chữ số": [f"Số {i}" for i in range(10)],
                "Xác suất": [round(float(p * 100), 2) for p in probabilities],
                "Phân loại": ["Dự đoán chính" if i == predicted_digit else "Lớp khác" for i in range(10)]
            })

            bar_chart = alt.Chart(df_probs).mark_bar(
                cornerRadiusTopRight=6,
                cornerRadiusBottomRight=6
            ).encode(
                y=alt.Y("Chữ số:N", sort=None, title=None, axis=alt.Axis(labelFontWeight="bold", labelFontSize=12)),
                x=alt.X("Xác suất:Q", title="Xác suất (%)", scale=alt.Scale(domain=[0, 100])),
                color=alt.Color(
                    "Phân loại:N",
                    scale=alt.Scale(
                        domain=["Dự đoán chính", "Lớp khác"],
                        range=["#2563EB", "#CBD5E1"]
                    ),
                    legend=None
                ),
                tooltip=[
                    alt.Tooltip("Chữ số:N", title="Chữ số"),
                    alt.Tooltip("Xác suất:Q", title="Xác suất (%)", format=".2f")
                ]
            ).properties(height=260)

            st.altair_chart(bar_chart, use_container_width=True)

            # Phân tích chẩn đoán chuyên sâu dựa trên Notebook
            second_digit = top3_indices[1]
            second_prob = probabilities[second_digit] * 100
            confused_key = (predicted_digit, second_digit)

            if confused_key in CONFUSED_PAIRS and second_prob > 12.0:
                st.warning(
                    f"**Phân tích cặp chữ số dễ nhầm ({predicted_digit} vs {second_digit}):**\n\n"
                    f"{CONFUSED_PAIRS[confused_key]}\n\n"
                    f"*(Xác suất số {second_digit} đạt {second_prob:.1f}%)*",
                    icon=":material/lightbulb:"
                )
            elif confidence < 60.0:
                st.info(
                    f"**Độ tin cậy ở mức trung bình ({confidence:.1f}%):** "
                    "Do Softmax Regression là mô hình phân loại tuyến tính (Linear Classifier), "
                    "nó nhạy cảm với nét vẽ bị lệch tâm, xoay góc hoặc độ dày nét không đều. "
                    "Bạn có thể vẽ lại với nét đậm hơn hoặc căn giữa khung vẽ để tăng độ chính xác.",
                    icon=":material/info:"
                )
