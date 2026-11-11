import streamlit as st
from streamlit_option_menu import option_menu

from model.config import MODEL_INFO, MODEL_READY
from utils.styles import inject_css
from views import guide, home, metrics, recognize

st.set_page_config(
    page_title="Nhận diện chữ số viết tay - MNIST AI",
    page_icon=":material/draw:",
    layout="wide",
    initial_sidebar_state="expanded"
)

inject_css()

PAGES = {
    "Trang chủ": home.render,
    "Nhận diện chữ số": recognize.render,
    "Đánh giá & Hiệu năng": metrics.render,
    "Hướng dẫn": guide.render,
}

with st.sidebar:
    st.markdown("""
    <div style="padding:16px 8px 20px 8px;">
        <div style="font-size:11px;color:#93C5FD;font-weight:700;letter-spacing:1.8px;">MACHINE LEARNING PROJECT</div>
        <div style="font-size:22px;color:white;font-weight:800;margin-top:6px;">Digit Recognition</div>
        <div style="font-size:12px;color:#94A3B8;margin-top:4px;">Handwritten Digit Classification</div>
    </div>""", unsafe_allow_html=True)

    selected = option_menu(
        menu_title=None,
        options=list(PAGES.keys()),
        icons=["house", "pencil-square", "graph-up", "question-circle"],
        default_index=0,
        styles={
            "container": {"padding": "0", "background-color": "#0F172A", "border-radius": "0"},
            "icon": {"color": "#93C5FD", "font-size": "17px"},
            "nav-link": {"font-size": "14px", "text-align": "left", "margin": "4px 0",
                        "border-radius": "8px", "color": "#CBD5E1"},
            "nav-link-selected": {"background-color": "#2563EB", "color": "white", "font-weight": "600"},
        },
    )

    st.markdown("---")
    st.markdown('<div style="font-size:13px;font-weight:700;color:#CBD5E1;margin-bottom:6px;"><i class="bi bi-cpu" style="color:#93C5FD;margin-right:8px;"></i>Thông tin mô hình AI</div>', unsafe_allow_html=True)
    st.caption(f"Thuật toán: {MODEL_INFO.get('Thuật toán', 'Softmax Regression')}")
    st.caption(f"Độ chính xác: {MODEL_INFO.get('Độ chính xác (Test Accuracy)', '~89%')}")
    st.caption(":green[●] Sẵn sàng hoạt động" if MODEL_READY else ":orange[●] Chưa kết nối trọng số model")

    st.markdown('<div class="sidebar-footer">HANDWRITTEN DIGIT RECOGNITION<br>'
                '<span style="color:#64748B;">AI APPLICATION 2026</span></div>',
                unsafe_allow_html=True)

# Render trang tương ứng
PAGES[selected]()
