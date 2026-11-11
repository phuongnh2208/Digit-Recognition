import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from utils.styles import header, icon_title, kpi_card


METRICS_PATH = Path("model/evaluation_metrics.json")


def render():
    header(
        "MODEL EVALUATION & BENCHMARK",
        "Đánh giá & Hiệu năng mô hình",
        "Kết quả đánh giá thực nghiệm toàn diện trên 10.000 mẫu kiểm thử độc lập từ bộ dữ liệu chuẩn MNIST."
    )

    if not METRICS_PATH.exists():
        st.error("Không tìm thấy file dữ liệu đánh giá `model/evaluation_metrics.json`.", icon=":material/error:")
        return

    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    # ------------------------------------------------------------------
    # 1. CÁC CHỈ SỐ TỔNG QUAN (KPIs)
    # ------------------------------------------------------------------
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi_card(
            "Độ chính xác (Accuracy)",
            f"{metrics['accuracy']}%",
            f"{metrics['correct_samples']} / {metrics['total_samples']} mẫu chính xác"
        )
    with k2:
        kpi_card(
            "Macro F1-Score",
            f"{metrics['macro_f1']:.4f}",
            "Cân bằng Precision & Recall"
        )
    with k3:
        kpi_card(
            "Weighted F1-Score",
            f"{metrics['weighted_f1']:.4f}",
            "Tính trọng số theo phân bố mẫu"
        )
    with k4:
        kpi_card(
            "Tập kiểm thử (Test Set)",
            "10.000",
            "Mẫu độc lập hoàn toàn"
        )

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # 2. TIẾN TRÌNH HUẤN LUYỆN & HỘI TỤ (LOSS CURVES)
    # ------------------------------------------------------------------
    with st.container(border=True):
        icon_title("graph-up", "Tiến trình hội tụ Loss qua 150 Epochs")
        st.markdown(
            "<span style='color:#64748B;font-size:13px;'>"
            "Theo dõi sự suy giảm của hàm mất mát Cross-Entropy trên tập Huấn luyện (55.000 mẫu) và Validation (5.000 mẫu):"
            "</span>",
            unsafe_allow_html=True
        )

        loss_data = metrics.get("loss_history", {})
        if loss_data:
            df_loss = pd.DataFrame({
                "Epoch": loss_data["epochs"],
                "Training Loss": loss_data["train_loss"],
                "Validation Loss": loss_data["val_loss"]
            }).set_index("Epoch")

            c_chart, c_note = st.columns([1.8, 1.2], gap="large")
            with c_chart:
                st.line_chart(df_loss, height=280)

            with c_note:
                st.markdown(
                    """
                    <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:10px;padding:16px;">
                        <div style="font-weight:700;color:#1E293B;margin-bottom:8px;">Nhận xét thực nghiệm từ Notebook:</div>
                        <ul style="color:#475569;font-size:13px;padding-left:18px;margin:0;">
                            <li><b>Tốc độ hội tụ:</b> Cả hai đường giảm rất nhanh trong 40 epoch đầu tiên (từ 2.30 xuống ~0.7), sau đó giảm dần và đạt mức ổn định ~0.47 ở epoch 150.</li>
                            <li style="margin-top:6px;"><b>Đánh giá Overfitting:</b> Đường <code>Validation Loss</code> bám rất sát đường <code>Training Loss</code> và không hề có dấu hiệu ngóc lên ở các vòng lặp sau.</li>
                            <li style="margin-top:6px;"><b>Kết luận:</b> Mô hình hoạt động ổn định, không bị học vẹt hay quá khớp trên dữ liệu kiểm tra.</li>
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    st.markdown("<br>", unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # 3. MA TRẬN NHẦM LẪN (CONFUSION MATRIX)
    # ------------------------------------------------------------------
    c_cm, c_misclass = st.columns([1.2, 1], gap="large")

    with c_cm:
        with st.container(border=True):
            icon_title("grid-3x3-gap-fill", "Ma trận nhầm lẫn (Confusion Matrix)")
            st.markdown(
                "<span style='color:#64748B;font-size:13px;'>"
                "Trục tung: Nhãn thực tế (True Label) | Trục hoành: Nhãn dự đoán (Predicted Label)"
                "</span>",
                unsafe_allow_html=True
            )

            cm_array = np.array(metrics["confusion_matrix"])

            fig, ax = plt.subplots(figsize=(6.5, 5.2))
            im = ax.imshow(cm_array, cmap="Blues", interpolation="nearest")
            max_val = cm_array.max()

            for i in range(10):
                for j in range(10):
                    val = int(cm_array[i, j])
                    color = "white" if val > (max_val * 0.45) else "#1E293B"
                    ax.text(
                        j, i, str(val),
                        ha="center", va="center",
                        color=color, fontsize=8.5, fontweight="bold"
                    )

            ax.set_xticks(range(10))
            ax.set_yticks(range(10))
            ax.set_xticklabels(range(10), fontweight="bold")
            ax.set_yticklabels(range(10), fontweight="bold")
            ax.set_xlabel("Nhãn dự đoán (Predicted Label)", fontsize=10, fontweight="bold", labelpad=8)
            ax.set_ylabel("Nhãn thực tế (True Label)", fontsize=10, fontweight="bold", labelpad=8)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    with c_misclass:
        with st.container(border=True):
            icon_title("exclamation-triangle-fill", "Top các cặp chữ số dễ nhầm lẫn nhất", color="#F59E0B")
            st.markdown(
                "<span style='color:#64748B;font-size:13px;'>"
                "Dữ liệu trích xuất tự động từ ma trận nhầm lẫn:"
                "</span>",
                unsafe_allow_html=True
            )

            top_mis = metrics.get("top_misclassified", [])
            df_mis = pd.DataFrame([
                {
                    "Thực tế": f"Số {item['true_digit']}",
                    "Đoán nhầm": f"Số {item['pred_digit']}",
                    "Số lần": item["count"],
                    "Tỷ lệ lớp": f"{item['rate_pct']}%"
                }
                for item in top_mis[:6]
            ])
            st.dataframe(df_mis, use_container_width=True, hide_index=True)

            st.markdown(
                """
                <div style="font-size:13px;color:#334155;margin-top:10px;">
                    <b>Nguyên nhân chính:</b>
                    <ul style="padding-left:18px;margin-top:6px;">
                        <li><b>(4, 9):</b> Nét sổ đứng thẳng của số 4 dễ bị nhầm với nét khép kín của số 9.</li>
                        <li><b>(5, 3) & (5, 8):</b> Chung phần thân cong phía dưới khi chiếu lên không gian pixel.</li>
                        <li><b>(2, 8):</b> Giao cắt pixel ở vùng trọng tâm giữa ảnh.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # 4. BÁO CÁO PHÂN LOẠI CHI TIẾT (PER-CLASS REPORT)
    # ------------------------------------------------------------------
    with st.container(border=True):
        icon_title("table", "Báo cáo phân loại chi tiết từng chữ số (0 đến 9)")
        st.markdown(
            "<span style='color:#64748B;font-size:13px;'>"
            "Đo lường năng lực nhận diện của từng lớp chữ số độc lập:"
            "</span>",
            unsafe_allow_html=True
        )

        class_rows = []
        for c in metrics["per_class_metrics"]:
            d = c["digit"]
            class_rows.append({
                "Chữ số": f"Số {d}",
                "Precision": f"{c['precision']:.4f}",
                "Recall": f"{c['recall']:.4f}",
                "F1-Score": f"{c['f1_score']:.4f}",
                "Số mẫu (Support)": c["support"],
                "Đánh giá hiệu năng": "Xuất sắc (F1 > 94%)" if c["f1_score"] >= 0.94 else (
                    "Khá tốt (F1 > 88%)" if c["f1_score"] >= 0.88 else "Dễ nhầm lẫn (F1 ~83-85%)"
                )
            })

        df_classes = pd.DataFrame(class_rows)
        st.dataframe(df_classes, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # 5. CƠ SỞ TOÁN HỌC & ĐÁNH GIÁ MÔ HÌNH (TỪ NOTEBOOK)
    # ------------------------------------------------------------------
    with st.container(border=True):
        icon_title("journal-code", "Cơ sở lý thuyết & Đánh giá mô hình")

        t1, t2 = st.tabs(["Công thức toán học", "Ưu điểm & Giới hạn của mô hình"])

        with t1:
            st.markdown("### 1. Hàm kích hoạt Softmax (Ổn định số học)")
            st.latex(r"\hat{y}_i = P(y = i \mid x) = \frac{e^{z_i - \max(Z)}}{\sum_{j=0}^{K-1} e^{z_j - \max(Z)}}")
            st.caption("Trừ đi max(Z) để triệt tiêu hiện tượng tràn số (numerical overflow) khi số mũ lớn.")

            st.markdown("### 2. Hàm mất mát Cross-Entropy Loss")
            st.latex(r"\mathcal{L}(Y, \hat{Y}) = -\frac{1}{N} \sum_{n=1}^{N} \sum_{k=0}^{K-1} y_{n, k} \ln(\hat{y}_{n, k} + 10^{-15})")
            st.caption("Cực tiểu hóa sai số giữa phân phối xác suất dự đoán và vector One-Hot thực tế.")

            st.markdown("### 3. Gradient và Cập nhật trọng số Gradient Descent")
            st.latex(r"\nabla_{W} \mathcal{L} = \frac{1}{N} X^T (\hat{Y} - Y), \quad W \leftarrow W - \eta \cdot \nabla_{W} \mathcal{L}")
            st.caption(f"Hệ số học tập learning_rate = 0.15, số epoch = 150.")

        with t2:
            col_pro, col_con = st.columns(2)
            with col_pro:
                st.markdown('#### <i class="bi bi-check2-circle" style="color:#16A34A;margin-right:8px;"></i>Ưu điểm của Softmax Regression:', unsafe_allow_html=True)
                st.markdown(
                    """
                    - **Toán học tường minh:** Đạo hàm trực tiếp, không cần backpropagation qua nhiều tầng ẩn.
                    - **Tốc độ suy luận tức thì:** Chỉ gồm 1 phép nhân ma trận $Z = X \\cdot W$ và Softmax, cực nhẹ khi triển khai Web.
                    - **Không bị quá khớp (Overfitting):** Khoảng cách Train/Val/Test Loss rất nhỏ.
                    - **Mô hình đối chứng vững chắc (Baseline):** Đạt ~89% độ chính xác cho bài toán 10 lớp.
                    """
                )
            with col_con:
                st.markdown('#### <i class="bi bi-exclamation-triangle" style="color:#F59E0B;margin-right:8px;"></i>Giới hạn tự nhiên:', unsafe_allow_html=True)
                st.markdown(
                    """
                    - **Bộ phân loại tuyến tính (Linear Classifier):** Ranh giới phân chia là các siêu phẳng tuyến tính trong không gian 784 chiều, không mô hình hóa được cấu trúc phi tuyến phức tạp.
                    - **Mất thông tin không gian 2D:** Làm phẳng ảnh 28×28 thành vector 784 làm mất tính liên kết giữa các pixel kề cận.
                    - **Trần độ chính xác:** ~89% là giới hạn thực tế của mô hình tuyến tính trên MNIST. Để đạt >98% cần sử dụng CNN (Convolutional Neural Networks).
                    """
                )
