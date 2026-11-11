import streamlit as st

from model.config import MODEL_INFO
from utils.styles import feature_card, header, icon_title, kpi_card, step


def render():
    header(
        "MACHINE LEARNING SHOWCASE",
        "Nhận diện chữ số viết tay",
        "Ứng dụng trí tuệ nhân tạo phân loại chữ số 0–9 thời gian thực dựa trên thuật toán Softmax Regression."
    )

    # 4 thẻ thống kê nổi bật (KPIs)
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi_card("Độ chính xác", "88.77%", "10.000 mẫu Test độc lập")
    with k2:
        kpi_card("Tập huấn luyện", "60.000", "Ảnh chuẩn MNIST")
    with k3:
        kpi_card("Số lớp phân loại", "10 chữ số", "Từ 0 đến 9")
    with k4:
        kpi_card("Thời gian suy luận", "< 5 ms", "Thuật toán tối ưu hóa vector")

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    with st.container(border=True):
        icon_title("stars", "Tổng quan dự án")
        st.markdown(
            """
            Ứng dụng xây dựng trên nền tảng **Softmax Regression (Multiclass Logistic Regression)** 
            được lập trình và tối ưu hóa từ đầu bằng thư viện **NumPy**. 
            Mô hình giải quyết bài toán phân loại đa lớp trên không gian 784 đặc trưng (ảnh xám 28×28 pixel), 
            sử dụng hàm kích hoạt **Softmax ổn định số học** và phương pháp tối ưu hóa **Gradient Descent** qua 150 vòng lặp.
            """
        )

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        feature_card(
            "cloud-arrow-up",
            "Tải ảnh từ máy",
            "Hỗ trợ tải file ảnh PNG, JPG, JPEG. Hệ thống tự động phát hiện màu nền và tiền xử lý chuẩn hóa."
        )
    with c2:
        feature_card(
            "pencil-square",
            "Vẽ trực tiếp (Canvas)",
            "Vẽ chữ số tùy ý bằng chuột hoặc bút cảm ứng. Hỗ trợ đầy đủ các số nhiều nét như 4, 5, 7."
        )
    with c3:
        feature_card(
            "bar-chart-line",
            "Phân tích xác suất 10 lớp",
            "Hiển thị chữ số dự đoán nổi bật, độ tin cậy và biểu đồ cột so sánh xác suất của toàn bộ 10 lớp."
        )

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
    left, right = st.columns([1.1, 1], gap="large")
    with left:
        with st.container(border=True):
            icon_title("cpu", "Thông số kỹ thuật mô hình")
            table_html = """
            <table style="width:100%; border-collapse:collapse; font-size:13.5px;">
            """
            for k, v in MODEL_INFO.items():
                table_html += f"""
                <tr style="border-bottom:1px solid #F1F5F9;">
                    <td style="padding:10px 4px; color:#64748B; font-weight:600; width:40%;">{k}</td>
                    <td style="padding:10px 4px; color:#0F172A; font-weight:700;">{v}</td>
                </tr>
                """
            table_html += "</table>"
            st.markdown(table_html, unsafe_allow_html=True)

    with right:
        with st.container(border=True):
            icon_title("rocket-takeoff", "Quy trình sử dụng nhanh")
            step(1, "Chuyển sang mục <b>Nhận diện chữ số</b> ở menu bên trái.")
            step(2, "Chọn 1 trong 3 phương thức: <b>Tải ảnh lên</b>, <b>Vẽ trực tiếp</b> hoặc chọn <b>Ảnh mẫu</b>.")
            step(3, "Nhấn <b>Bắt đầu nhận diện</b> để xem kết quả dự đoán cùng biểu đồ phân phối xác suất.")

            st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
            icon_title("graph-up", "Báo cáo thực nghiệm chuyên sâu")
            st.info(
                "Khám phá trang **Đánh giá & Hiệu năng** để xem biểu đồ hội tụ Loss 150 epochs, "
                "ma trận nhầm lẫn Confusion Matrix 10×10 và phân tích các trường hợp sai lệch thực tế.",
                icon=":material/analytics:"
            )
