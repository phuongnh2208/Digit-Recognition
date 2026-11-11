import streamlit as st

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');
@import url('https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css');

/* Global Font & Smooth Layout */
html, body, [class*="css"], .stApp {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #0F172A;
}

.stApp {
    background: linear-gradient(180deg, #F8FAFC 0%, #EEF2F6 100%);
    min-height: 100vh;
}

/* Container Width & Spacing */
.main .block-container {
    padding: 2.2rem 3.5rem 3.5rem 3.5rem;
    max-width: 1380px;
}

/* Sidebar Customization */
[data-testid="stSidebar"] {
    background: #0F172A !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    box-shadow: 4px 0 24px rgba(0, 0, 0, 0.15);
}

[data-testid="stSidebar"] * {
    color: #E2E8F0;
}

[data-testid="stSidebar"] hr {
    border-color: rgba(255, 255, 255, 0.12) !important;
    margin: 18px 0;
}

/* Header & Banner Styles */
.header-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    background: rgba(37, 99, 235, 0.08);
    border: 1px solid rgba(37, 99, 235, 0.2);
    border-radius: 9999px;
    color: #2563EB;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    margin-bottom: 8px;
}

.main-title {
    color: #0F172A;
    font-size: 30px;
    font-weight: 800;
    letter-spacing: -0.5px;
    line-height: 1.25;
    margin: 4px 0 8px 0;
}

.subtitle {
    color: #64748B;
    font-size: 14.5px;
    line-height: 1.5;
    margin-bottom: 24px;
    font-weight: 450;
}

/* Cards & Containers */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.04), 0 2px 6px -1px rgba(15, 23, 42, 0.02);
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    padding: 2px;
}

div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: #CBD5E1;
    box-shadow: 0 8px 30px -4px rgba(15, 23, 42, 0.08);
}

/* Modern Segmented Radio Buttons */
div[data-testid="stRadio"] > div[role="radiogroup"] {
    background: #F1F5F9;
    padding: 4px;
    border-radius: 12px;
    border: 1px solid #E2E8F0;
    gap: 4px;
}

div[data-testid="stRadio"] > div[role="radiogroup"] > label {
    background: transparent;
    border-radius: 8px;
    padding: 8px 16px !important;
    font-weight: 600;
    font-size: 13.5px;
    color: #475569 !important;
    transition: all 0.2s ease;
    border: 1px solid transparent;
}

div[data-testid="stRadio"] > div[role="radiogroup"] > label:hover {
    color: #0F172A !important;
    background: rgba(255, 255, 255, 0.6);
}

div[data-testid="stRadio"] > div[role="radiogroup"] > label[data-checked="true"] {
    background: #FFFFFF !important;
    color: #2563EB !important;
    box-shadow: 0 2px 6px rgba(15, 23, 42, 0.08);
    border-color: #E2E8F0;
}

/* Streamlit Buttons */
.stButton > button {
    background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
    color: #FFFFFF !important;
    border: none;
    border-radius: 10px;
    height: 44px;
    font-weight: 600;
    font-size: 14px;
    letter-spacing: 0.2px;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 2px 8px rgba(37, 99, 235, 0.25);
}

.stButton > button:hover {
    background: linear-gradient(135deg, #1D4ED8 0%, #1E40AF 100%);
    box-shadow: 0 6px 16px rgba(37, 99, 235, 0.35);
    transform: translateY(-1px);
}

.stButton > button:active {
    transform: translateY(0);
}

.stButton > button:disabled {
    background: #E2E8F0 !important;
    color: #94A3B8 !important;
    box-shadow: none !important;
    cursor: not-allowed;
}

/* Secondary Button Styling */
button[kind="secondary"] {
    background: #FFFFFF !important;
    color: #334155 !important;
    border: 1px solid #CBD5E1 !important;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05) !important;
}

button[kind="secondary"]:hover {
    background: #F8FAFC !important;
    border-color: #94A3B8 !important;
    color: #0F172A !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 2px solid #E2E8F0;
    padding-bottom: 2px;
}

.stTabs [data-baseweb="tab"] {
    color: #64748B;
    font-weight: 600;
    font-size: 14px;
    padding: 10px 18px;
    border-radius: 8px 8px 0 0;
    transition: all 0.2s ease;
}

.stTabs [data-baseweb="tab"]:hover {
    color: #2563EB;
    background: rgba(37, 99, 235, 0.04);
}

.stTabs [aria-selected="true"] {
    color: #2563EB !important;
    border-bottom: 2px solid #2563EB !important;
    background: transparent !important;
}

/* File Uploader Section */
div[data-testid="stFileUploader"] section {
    background: #F8FAFC;
    border: 2px dashed #CBD5E1;
    border-radius: 14px;
    padding: 24px 16px;
    transition: all 0.2s ease;
}

div[data-testid="stFileUploader"] section:hover {
    border-color: #2563EB;
    background: #EFF6FF;
}

/* Feature & Stat Cards */
.feature-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    padding: 24px;
    height: 100%;
    transition: all 0.25s ease;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
}

.feature-card:hover {
    transform: translateY(-3px);
    border-color: #93C5FD;
    box-shadow: 0 12px 28px -6px rgba(37, 99, 235, 0.12);
}

.feature-card .icon-wrapper {
    width: 48px;
    height: 48px;
    border-radius: 12px;
    background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 14px;
}

.feature-card .icon {
    font-size: 22px;
    color: #2563EB;
}

.feature-card h4 {
    margin: 0 0 8px 0;
    color: #0F172A;
    font-size: 16.5px;
    font-weight: 700;
}

.feature-card p {
    color: #64748B;
    font-size: 13.5px;
    line-height: 1.55;
    margin: 0;
}

/* Metric KPI Mini Cards */
.kpi-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 18px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
}

.kpi-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 3px;
    background: linear-gradient(90deg, #2563EB, #60A5FA);
}

.kpi-label {
    font-size: 12px;
    font-weight: 600;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.kpi-val {
    font-size: 28px;
    font-weight: 800;
    color: #0F172A;
    margin: 6px 0 2px 0;
    letter-spacing: -0.5px;
}

.kpi-caption {
    font-size: 12px;
    color: #94A3B8;
}

/* Prediction Result Box */
.result-card-container {
    background: linear-gradient(135deg, #1E3A8A 0%, #1E40AF 50%, #2563EB 100%);
    border-radius: 16px;
    padding: 24px 20px;
    text-align: center;
    color: white;
    box-shadow: 0 10px 25px -5px rgba(37, 99, 235, 0.4);
    position: relative;
    overflow: hidden;
}

.result-card-container::after {
    content: '';
    position: absolute;
    top: -50%;
    right: -50%;
    width: 100%;
    height: 100%;
    background: radial-gradient(circle, rgba(255,255,255,0.15) 0%, transparent 70%);
    pointer-events: none;
}

.result-glow-digit {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 92px;
    font-weight: 800;
    line-height: 1;
    color: #FFFFFF;
    text-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    margin: 6px 0;
}

.result-glow-tag {
    font-size: 11.5px;
    font-weight: 700;
    letter-spacing: 1.5px;
    color: #BFDBFE;
    text-transform: uppercase;
}

.result-confidence-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(255, 255, 255, 0.2);
    backdrop-filter: blur(8px);
    padding: 6px 14px;
    border-radius: 9999px;
    font-size: 13.5px;
    font-weight: 700;
    color: #FFFFFF;
    margin-top: 8px;
    border: 1px solid rgba(255, 255, 255, 0.25);
}

/* Styled Steps */
.step-item {
    display: flex;
    gap: 14px;
    align-items: flex-start;
    margin-bottom: 16px;
}

.step-num {
    background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
    color: white;
    min-width: 32px;
    height: 32px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 13.5px;
    box-shadow: 0 2px 6px rgba(37, 99, 235, 0.3);
}

.step-txt {
    color: #334155;
    font-size: 14px;
    line-height: 1.5;
    padding-top: 5px;
}

/* Sidebar Footer */
.sidebar-footer {
    color: #64748B;
    font-size: 11px;
    text-align: center;
    padding-top: 24px;
    letter-spacing: 0.5px;
}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def header(label, title, subtitle):
    st.markdown(f'<div class="header-badge">{label}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="main-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="subtitle">{subtitle}</div>', unsafe_allow_html=True)


def icon_title(icon, text, color="#2563EB", size="h3"):
    st.markdown(
        f'<{size} style="color:#0F172A;font-weight:700;margin:0 0 14px 0;letter-spacing:-0.3px;display:flex;align-items:center;">'
        f'<span style="background:rgba(37,99,235,0.08);width:32px;height:32px;border-radius:8px;display:inline-flex;align-items:center;justify-content:center;margin-right:10px;">'
        f'<i class="bi bi-{icon}" style="color:{color};font-size:16px;"></i></span>{text}</{size}>',
        unsafe_allow_html=True
    )


def feature_card(icon, title, desc):
    st.markdown(
        f'<div class="feature-card">'
        f'<div class="icon-wrapper"><i class="bi bi-{icon} icon"></i></div>'
        f'<h4>{title}</h4><p>{desc}</p></div>',
        unsafe_allow_html=True
    )


def kpi_card(label, value, caption=""):
    st.markdown(
        f'<div class="kpi-card">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-val">{value}</div>'
        f'<div class="kpi-caption">{caption}</div>'
        f'</div>',
        unsafe_allow_html=True
    )


def step(n, text):
    st.markdown(
        f'<div class="step-item">'
        f'<div class="step-num">{n}</div>'
        f'<div class="step-txt">{text}</div>'
        f'</div>',
        unsafe_allow_html=True
    )
