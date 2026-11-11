import streamlit as st

from utils.styles import header, icon_title, step


def render():
    header(
        "USER GUIDE & BEST PRACTICES",
        "Hướng dẫn sử dụng & Mẹo tối ưu",
        "Chi tiết quy trình và các lưu ý quan trọng để mô hình AI nhận diện với độ chính xác cao nhất."
    )

    with st.container(border=True):
        icon_title("list-ol", "Quy trình sử dụng ứng dụng")
        step(1, "Vào mục <b>Nhận diện chữ số</b> trên thanh điều hướng bên trái.")
        step(2, "Chọn 1 trong 3 phương thức cung cấp chữ số:")
        st.markdown(
            """
- **Tải ảnh lên**: Tải file ảnh có sẵn từ máy tính (`.png`, `.jpg`, `.jpeg`).
- **Vẽ trực tiếp**: Dùng chuột hoặc bút cảm ứng vẽ nét chữ số tùy ý lên bảng vẽ tương tác đen.
- **Chọn ảnh mẫu**: Chọn nhanh một chữ số mẫu chuẩn từ 0 đến 9 đã chuẩn bị sẵn từ tập test MNIST.
            """
        )
        step(3, "Nhấn nút <b>Bắt đầu nhận diện</b> để khởi chạy mô hình.")
        step(4, "Xem chữ số dự đoán nổi bật, độ tin cậy (%), ảnh sau tiền xử lý 28×28 và phân phối xác suất 10 lớp.")

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        with st.container(border=True):
            icon_title("check-circle-fill", "Nên làm (Khuyến nghị)", color="#16A34A")
            st.markdown(
                '<div style="font-size:13.5px;color:#334155;line-height:1.7;">'
                '<div style="display:flex;gap:8px;margin-bottom:8px;"><i class="bi bi-check2" style="color:#16A34A;font-weight:bold;font-size:16px;"></i><span><b>Chỉ 1 chữ số duy nhất</b> trong mỗi khung hình hoặc nét vẽ.</span></div>'
                '<div style="display:flex;gap:8px;margin-bottom:8px;"><i class="bi bi-check2" style="color:#16A34A;font-weight:bold;font-size:16px;"></i><span><b>Độ dày nét cọ vừa phải</b> (khoảng 16–22px) để sau khi thu nhỏ nét không bị đứt.</span></div>'
                '<div style="display:flex;gap:8px;margin-bottom:8px;"><i class="bi bi-check2" style="color:#16A34A;font-weight:bold;font-size:16px;"></i><span>Đối với ảnh chụp: Chọn góc chụp thẳng đứng, chữ số tương phản cao với nền giấy trắng.</span></div>'
                '<div style="display:flex;gap:8px;margin-bottom:8px;"><i class="bi bi-check2" style="color:#16A34A;font-weight:bold;font-size:16px;"></i><span>Căn chữ số nằm ở khu vực trung tâm, kích thước chiếm khoảng 70–80% khung hình.</span></div>'
                '<div style="display:flex;gap:8px;"><i class="bi bi-check2" style="color:#16A34A;font-weight:bold;font-size:16px;"></i><span>Quan sát ô ảnh 28×28 Grayscale ở kết quả để kiểm tra AI có thấy nét vẽ rõ không.</span></div>'
                '</div>',
                unsafe_allow_html=True
            )

    with c2:
        with st.container(border=True):
            icon_title("x-circle-fill", "Cần tránh (Lưu ý)", color="#DC2626")
            st.markdown(
                '<div style="font-size:13.5px;color:#334155;line-height:1.7;">'
                '<div style="display:flex;gap:8px;margin-bottom:8px;"><i class="bi bi-x-lg" style="color:#DC2626;font-weight:bold;font-size:14px;margin-top:4px;"></i><span><b>Tránh ảnh chứa nhiều chữ số</b> hoặc văn bản xung quanh (mô hình chỉ nhận 1 số).</span></div>'
                '<div style="display:flex;gap:8px;margin-bottom:8px;"><i class="bi bi-x-lg" style="color:#DC2626;font-weight:bold;font-size:14px;margin-top:4px;"></i><span><b>Tránh vẽ nét quá mảnh</b> (&lt; 8px) khiến pixel bị mờ khi rescale về 28×28.</span></div>'
                '<div style="display:flex;gap:8px;margin-bottom:8px;"><i class="bi bi-x-lg" style="color:#DC2626;font-weight:bold;font-size:14px;margin-top:4px;"></i><span>Tránh ảnh bị lóa sáng, đổ bóng đen hoặc góc chụp bị nghiêng/méo quá mức.</span></div>'
                '<div style="display:flex;gap:8px;margin-bottom:8px;"><i class="bi bi-x-lg" style="color:#DC2626;font-weight:bold;font-size:14px;margin-top:4px;"></i><span><b>Lưu ý các cặp số dễ nhầm:</b> 4 & 9 (vòng khép kín), 5 & 3 (phần bụng cong), 2 & 8 (vùng giao cắt).</span></div>'
                '<div style="display:flex;gap:8px;"><i class="bi bi-x-lg" style="color:#DC2626;font-weight:bold;font-size:14px;margin-top:4px;"></i><span>Tránh các định dạng file lạ ngoài `.png`, `.jpg`, `.jpeg`.</span></div>'
                '</div>',
                unsafe_allow_html=True
            )

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
    with st.container(border=True):
        icon_title("gear-wide-connected", "Cơ chế tiền xử lý ảnh tự động (MNIST Standard Pipeline)", color="#2563EB")
        st.markdown(
            "Toàn bộ ảnh đầu vào (dù vẽ trực tiếp hay tải từ file) đều đi qua quy trình tiền xử lý 5 bước chuẩn quốc tế:"
        )
        p1, p2, p3, p4, p5 = st.columns(5)
        with p1:
            st.markdown(
                '<div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:12px;padding:12px 10px;text-align:center;height:100%;">'
                '<div style="font-weight:800;color:#2563EB;font-size:14px;margin-bottom:4px;">Bước 1</div>'
                '<div style="font-weight:700;color:#0F172A;font-size:12.5px;margin-bottom:4px;">Nhận diện màu nền</div>'
                '<div style="color:#64748B;font-size:11.5px;">Tự động đảo màu đưa về chữ trắng nền đen</div>'
                '</div>',
                unsafe_allow_html=True
            )
        with p2:
            st.markdown(
                '<div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:12px;padding:12px 10px;text-align:center;height:100%;">'
                '<div style="font-weight:800;color:#2563EB;font-size:14px;margin-bottom:4px;">Bước 2</div>'
                '<div style="font-weight:700;color:#0F172A;font-size:12.5px;margin-bottom:4px;">Cắt viền (Crop)</div>'
                '<div style="color:#64748B;font-size:11.5px;">Loại bỏ khoảng trống thừa bám sát nét chữ</div>'
                '</div>',
                unsafe_allow_html=True
            )
        with p3:
            st.markdown(
                '<div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:12px;padding:12px 10px;text-align:center;height:100%;">'
                '<div style="font-weight:800;color:#2563EB;font-size:14px;margin-bottom:4px;">Bước 3</div>'
                '<div style="font-weight:700;color:#0F172A;font-size:12.5px;margin-bottom:4px;">Co giãn tỷ lệ</div>'
                '<div style="color:#64748B;font-size:11.5px;">Resize chữ số vừa vặn khung 20×20 px</div>'
                '</div>',
                unsafe_allow_html=True
            )
        with p4:
            st.markdown(
                '<div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:12px;padding:12px 10px;text-align:center;height:100%;">'
                '<div style="font-weight:800;color:#2563EB;font-size:14px;margin-bottom:4px;">Bước 4</div>'
                '<div style="font-weight:700;color:#0F172A;font-size:12.5px;margin-bottom:4px;">Căn trọng tâm</div>'
                '<div style="color:#64748B;font-size:11.5px;">Căn giữa theo Center of Mass trên khung 28×28</div>'
                '</div>',
                unsafe_allow_html=True
            )
        with p5:
            st.markdown(
                '<div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:12px;padding:12px 10px;text-align:center;height:100%;">'
                '<div style="font-weight:800;color:#2563EB;font-size:14px;margin-bottom:4px;">Bước 5</div>'
                '<div style="font-weight:700;color:#0F172A;font-size:12.5px;margin-bottom:4px;">Chuẩn hóa Pixel</div>'
                '<div style="color:#64748B;font-size:11.5px;">Chia pixel cho 255 về khoảng [0, 1]</div>'
                '</div>',
                unsafe_allow_html=True
            )
