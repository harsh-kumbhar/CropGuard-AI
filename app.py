import streamlit as st
import numpy as np
from PIL import Image
import cv2
import time

from utils import resize_image
from cv_engine import analyze_crop

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CropGuard AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
# CSS Injection — Dark Agri-Tech Dashboard
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #080d0a !important;
    color: #d4f0dc !important;
    font-family: 'Syne', sans-serif;
}

[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse at 20% 0%, #0a2116 0%, #080d0a 55%) !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header, [data-testid="stToolbar"] { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ── Main content padding ── */
.block-container {
    padding: 2rem 3rem 4rem !important;
    max-width: 1400px !important;
}

/* ── Hero Header ── */
.hero-header {
    display: flex;
    align-items: center;
    gap: 1.5rem;
    padding: 2.5rem 0 1.5rem;
    border-bottom: 1px solid #1a3328;
    margin-bottom: 2.5rem;
}

.hero-badge {
    background: linear-gradient(135deg, #00ff88, #00cc6a);
    color: #080d0a;
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    padding: 0.3rem 0.75rem;
    border-radius: 2px;
    text-transform: uppercase;
}

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    color: #f0fff5;
    letter-spacing: -0.02em;
    line-height: 1;
}

.hero-title span {
    color: #00ff88;
}

.hero-subtitle {
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    color: #4a7a5e;
    letter-spacing: 0.05em;
    margin-top: 0.4rem;
}

.hero-right {
    margin-left: auto;
    text-align: right;
}

.system-status {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #00ff88;
    letter-spacing: 0.08em;
}

.system-status::before {
    content: '●';
    margin-right: 0.4rem;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}

/* ── Upload Zone ── */
.upload-zone-wrapper {
    background: #0c1912;
    border: 1.5px dashed #1e4030;
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    transition: border-color 0.3s;
    margin-bottom: 1.5rem;
}

.upload-zone-wrapper:hover {
    border-color: #00ff88;
}

.upload-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #4a7a5e;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

/* ── Section Headers ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin: 2.5rem 0 1.2rem;
}

.section-number {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: #00ff88;
    background: #0c2018;
    border: 1px solid #1e4030;
    padding: 0.25rem 0.5rem;
    border-radius: 3px;
    letter-spacing: 0.1em;
}

.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.15rem;
    font-weight: 700;
    color: #c8ecd4;
    letter-spacing: 0.02em;
}

.section-line {
    flex: 1;
    height: 1px;
    background: linear-gradient(to right, #1a3328, transparent);
}

/* ── Metric Cards ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 1.5rem;
}

.metric-card {
    background: #0c1912;
    border: 1px solid #1a3328;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s, transform 0.2s;
}

.metric-card:hover {
    border-color: #2a5040;
    transform: translateY(-2px);
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #00ff88, transparent);
}

.metric-card.warning::before { background: linear-gradient(90deg, #ffcc00, transparent); }
.metric-card.danger::before  { background: linear-gradient(90deg, #ff4444, transparent); }
.metric-card.info::before    { background: linear-gradient(90deg, #00ccff, transparent); }

.metric-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem;
    color: #3a6050;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}

.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #00ff88;
    line-height: 1;
}

.metric-card.warning .metric-value { color: #ffcc00; }
.metric-card.danger  .metric-value { color: #ff4444; }
.metric-card.info    .metric-value { color: #00ccff; }

.metric-unit {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: #3a6050;
    margin-top: 0.3rem;
}

/* ── Result Banner ── */
.result-banner {
    background: #0c1912;
    border: 1px solid #1a3328;
    border-radius: 12px;
    padding: 1.8rem 2.2rem;
    display: flex;
    align-items: center;
    gap: 2rem;
    margin: 1.5rem 0;
    position: relative;
    overflow: hidden;
}

.result-banner::after {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(circle at 90% 50%, rgba(0,255,136,0.04), transparent 60%);
    pointer-events: none;
}

.result-banner.warning::after { background: radial-gradient(circle at 90% 50%, rgba(255,204,0,0.05), transparent 60%); }
.result-banner.danger::after  { background: radial-gradient(circle at 90% 50%, rgba(255,68,68,0.05), transparent 60%); }

.result-status-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: #4a7a5e;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}

.result-status-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    font-weight: 800;
    color: #00ff88;
}

.result-banner.warning .result-status-value { color: #ffcc00; }
.result-banner.danger  .result-status-value { color: #ff4444; }

.result-suggestion {
    font-family: 'Syne', sans-serif;
    font-size: 0.92rem;
    color: #8ab89a;
    line-height: 1.65;
    flex: 1;
    padding-left: 2rem;
    border-left: 1px solid #1a3328;
}

/* ── Progress Bar ── */
.progress-wrap {
    background: #0c1912;
    border: 1px solid #1a3328;
    border-radius: 8px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
}

.progress-label {
    display: flex;
    justify-content: space-between;
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: #4a7a5e;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}

.progress-bar-bg {
    height: 6px;
    background: #0f1f18;
    border-radius: 3px;
    overflow: hidden;
}

.progress-bar-fill {
    height: 100%;
    border-radius: 3px;
    background: linear-gradient(90deg, #00cc6a, #00ff88);
    transition: width 0.8s ease;
}

.progress-bar-fill.warning { background: linear-gradient(90deg, #cc9900, #ffcc00); }
.progress-bar-fill.danger  { background: linear-gradient(90deg, #cc2222, #ff4444); }
.progress-bar-fill.info    { background: linear-gradient(90deg, #0099cc, #00ccff); }

/* ── Image caption tags ── */
.img-tag {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: #3a6050;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.25rem 0;
    border-bottom: 1px solid #1a3328;
    margin-bottom: 0.5rem;
    display: flex;
    justify-content: space-between;
}

.img-tag span { color: #00ff88; }

/* ── Data Table ── */
.data-table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
}

.data-table th {
    text-align: left;
    color: #3a6050;
    font-size: 0.6rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0.5rem 0.75rem;
    border-bottom: 1px solid #1a3328;
}

.data-table td {
    padding: 0.55rem 0.75rem;
    color: #8ab89a;
    border-bottom: 1px solid #0f1f18;
}

.data-table td:last-child {
    color: #00ff88;
    text-align: right;
}

.data-table tr:hover td { background: #0c1912; }

/* ── Footer ── */
.footer {
    margin-top: 4rem;
    padding-top: 1.5rem;
    border-top: 1px solid #1a3328;
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: #2a4838;
    display: flex;
    justify-content: space-between;
    letter-spacing: 0.08em;
}

/* ── Streamlit overrides ── */
[data-testid="stFileUploader"] {
    background: #0c1912 !important;
    border: 1.5px dashed #1e4030 !important;
    border-radius: 10px !important;
    padding: 1rem !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: #00ff88 !important;
}

[data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
}

[data-testid="stFileUploaderDropzoneInstructions"] p,
[data-testid="stFileUploaderDropzoneInstructions"] span {
    color: #4a7a5e !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.75rem !important;
}

.stImage > img {
    border-radius: 8px;
    border: 1px solid #1a3328;
}

[data-testid="stSpinner"] {
    color: #00ff88 !important;
}

/* Divider override */
hr {
    border-color: #1a3328 !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Hero Header
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div>
        <div class="hero-badge">CV · Remote Sensing · Agri-AI</div>
        <div class="hero-title">Crop<span>Guard</span> AI</div>
        <div class="hero-subtitle">CROP LODGING DETECTION SYSTEM // COMPUTER VISION PIPELINE v2.0</div>
    </div>
    <div class="hero-right">
        <div class="system-status">SYSTEM ONLINE</div>
        <div style="font-family:'Space Mono',monospace;font-size:0.62rem;color:#2a4838;margin-top:0.3rem;">
            MODULES: VEGETATION · EDGE · STEM
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Upload
# ─────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-number">01</div>
    <div class="section-title">Image Input</div>
    <div class="section-line"></div>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload Crop Field Image",
    type=["jpg", "jpeg", "png"],
    help="Supports JPG, JPEG, PNG. Best results with overhead or close-up field shots.",
    label_visibility="collapsed"
)

# ─────────────────────────────────────────────
# Analysis Pipeline
# ─────────────────────────────────────────────
if uploaded_file is not None:

    image = Image.open(uploaded_file)
    image = np.array(image)

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    image = resize_image(image)

    # ── Section: Input Preview ──
    st.markdown("""
    <div class="section-header">
        <div class="section-number">02</div>
        <div class="section-title">Input Preview</div>
        <div class="section-line"></div>
    </div>
    """, unsafe_allow_html=True)

    col_img, col_info = st.columns([2, 1])

    with col_img:
        st.markdown('<div class="img-tag">ORIGINAL IMAGE <span>RAW INPUT</span></div>', unsafe_allow_html=True)
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)

    with col_info:
        h, w = image.shape[:2]
        st.markdown(f"""
        <div style="display:flex;flex-direction:column;gap:0.8rem;margin-top:1rem;">
            <div class="progress-wrap">
                <div class="metric-label">Image Dimensions</div>
                <div class="metric-value" style="font-size:1.1rem;color:#00ccff;">{w} × {h}</div>
                <div class="metric-unit">PIXELS</div>
            </div>
            <div class="progress-wrap">
                <div class="metric-label">Resolution</div>
                <div class="metric-value" style="font-size:1.1rem;color:#00ccff;">{round((w*h)/1_000_000, 2)}</div>
                <div class="metric-unit">MEGAPIXELS</div>
            </div>
            <div class="progress-wrap">
                <div class="metric-label">File Format</div>
                <div class="metric-value" style="font-size:1.1rem;color:#00ccff;">{uploaded_file.type.split('/')[1].upper()}</div>
                <div class="metric-unit">IMAGE FORMAT</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Run Analysis ──
    st.markdown("""
    <div class="section-header">
        <div class="section-number">03</div>
        <div class="section-title">Running CV Pipeline</div>
        <div class="section-line"></div>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Executing analysis pipeline..."):
        t_start = time.time()
        veg_mask, veg_density, veg_overlay, edges_colored, stem_img, result, suggestion, confidence, metrics = analyze_crop(image)
        t_end = time.time()
        process_time = round((t_end - t_start) * 1000, 1)

    st.success(f"Pipeline complete in {process_time}ms")

    # ── Section: Result Banner ──
    r = metrics["result"]
    res_label = r["label"]
    avg_angle = r.get("avg_angle")
    risk_score = r.get("risk_score", 0)

    banner_class = ""
    if "Moderate" in res_label:
        banner_class = "warning"
    elif "Severe" in res_label:
        banner_class = "danger"

    angle_display = f"{avg_angle}°" if avg_angle is not None else "N/A"
    risk_display = f"{risk_score}%" if risk_score is not None else "N/A"

    st.markdown(f"""
    <div class="section-header">
        <div class="section-number">04</div>
        <div class="section-title">Detection Result</div>
        <div class="section-line"></div>
    </div>
    <div class="result-banner {banner_class}">
        <div>
            <div class="result-status-label">Lodging Status</div>
            <div class="result-status-value">{res_label}</div>
            <div style="font-family:'Space Mono',monospace;font-size:0.65rem;color:#3a6050;margin-top:0.6rem;letter-spacing:0.08em;">
                CONFIDENCE: {confidence}% &nbsp;|&nbsp; AVG ANGLE: {angle_display} &nbsp;|&nbsp; RISK INDEX: {risk_display}
            </div>
        </div>
        <div class="result-suggestion">{suggestion}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Section: Key Metrics ──
    veg = metrics["vegetation"]
    edges = metrics["edges"]
    stems = metrics["stems"]

    conf_class = "danger" if confidence < 50 else ("warning" if confidence < 75 else "")
    risk_class = "danger" if (risk_score or 0) > 65 else ("warning" if (risk_score or 0) > 35 else "")
    angle_class = "danger" if (avg_angle or 90) < 40 else ("warning" if (avg_angle or 90) < 70 else "")
    cov_class = "info"

    st.markdown(f"""
    <div class="section-header">
        <div class="section-number">05</div>
        <div class="section-title">Key Metrics</div>
        <div class="section-line"></div>
    </div>
    <div class="metric-grid">
        <div class="metric-card {conf_class}">
            <div class="metric-label">Confidence Score</div>
            <div class="metric-value">{confidence}</div>
            <div class="metric-unit">PERCENT</div>
        </div>
        <div class="metric-card {angle_class}">
            <div class="metric-label">Avg Stem Angle</div>
            <div class="metric-value">{angle_display}</div>
            <div class="metric-unit">DEGREES (90° = VERTICAL)</div>
        </div>
        <div class="metric-card {risk_class}">
            <div class="metric-label">Risk Index</div>
            <div class="metric-value">{risk_display}</div>
            <div class="metric-unit">LODGING RISK LEVEL</div>
        </div>
        <div class="metric-card {cov_class}">
            <div class="metric-label">Vegetation Coverage</div>
            <div class="metric-value">{veg.get('coverage_pct', 0)}</div>
            <div class="metric-unit">PERCENT OF FRAME</div>
        </div>
    </div>
    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-label">Stem Lines Detected</div>
            <div class="metric-value">{stems.get('line_count', 0)}</div>
            <div class="metric-unit">HOUGH LINES</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Vertical Stems</div>
            <div class="metric-value">{stems.get('vertical_pct', 0)}</div>
            <div class="metric-unit">PERCENT UPRIGHT</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Edge Density</div>
            <div class="metric-value">{edges.get('edge_density_pct', 0)}</div>
            <div class="metric-unit">PERCENT OF FRAME</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Sharpness Score</div>
            <div class="metric-value">{edges.get('sharpness_score', 0)}</div>
            <div class="metric-unit">LAPLACIAN VARIANCE</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Progress Bars ──
    conf_bar_class = "danger" if confidence < 50 else ("warning" if confidence < 75 else "")
    risk_bar_class = "danger" if (risk_score or 0) > 65 else ("warning" if (risk_score or 0) > 35 else "")
    veg_bar = min(100, veg.get("coverage_pct", 0))
    vert_bar = stems.get("vertical_pct", 0)

    st.markdown(f"""
    <div class="section-header">
        <div class="section-number">06</div>
        <div class="section-title">Signal Strength Breakdown</div>
        <div class="section-line"></div>
    </div>
    <div class="progress-wrap">
        <div class="progress-label"><span>Detection Confidence</span><span>{confidence}%</span></div>
        <div class="progress-bar-bg">
            <div class="progress-bar-fill {conf_bar_class}" style="width:{confidence}%"></div>
        </div>
    </div>
    <div class="progress-wrap">
        <div class="progress-label"><span>Vegetation Coverage</span><span>{veg_bar}%</span></div>
        <div class="progress-bar-bg">
            <div class="progress-bar-fill info" style="width:{veg_bar}%"></div>
        </div>
    </div>
    <div class="progress-wrap">
        <div class="progress-label"><span>Upright Stem Ratio</span><span>{vert_bar}%</span></div>
        <div class="progress-bar-bg">
            <div class="progress-bar-fill" style="width:{vert_bar}%"></div>
        </div>
    </div>
    <div class="progress-wrap">
        <div class="progress-label"><span>Lodging Risk Index</span><span>{risk_score or 0}%</span></div>
        <div class="progress-bar-bg">
            <div class="progress-bar-fill {risk_bar_class}" style="width:{risk_score or 0}%"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Section: CV Outputs ──
    st.markdown("""
    <div class="section-header">
        <div class="section-number">07</div>
        <div class="section-title">Computer Vision Outputs</div>
        <div class="section-line"></div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="img-tag">VEGETATION MASK <span>HSV FILTER</span></div>', unsafe_allow_html=True)
        st.image(veg_mask, use_column_width=True)

    with c2:
        st.markdown('<div class="img-tag">VEGETATION DENSITY MAP <span>HEATMAP OVERLAY</span></div>', unsafe_allow_html=True)
        st.image(cv2.cvtColor(veg_density, cv2.COLOR_BGR2RGB), use_column_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<div class="img-tag">EDGE DETECTION <span>AUTO CANNY</span></div>', unsafe_allow_html=True)
        st.image(cv2.cvtColor(edges_colored, cv2.COLOR_BGR2RGB), use_column_width=True)

    with c4:
        st.markdown('<div class="img-tag">STEM ANALYSIS <span>HOUGH LINES</span></div>', unsafe_allow_html=True)
        st.image(cv2.cvtColor(stem_img, cv2.COLOR_BGR2RGB), use_column_width=True)

    st.markdown('<div class="img-tag" style="margin-top:1rem;">VEGETATION OVERLAY <span>MASK ON ORIGINAL</span></div>', unsafe_allow_html=True)
    st.image(cv2.cvtColor(veg_overlay, cv2.COLOR_BGR2RGB), use_column_width=True)

    # ── Section: Detailed Stats Table ──
    img_s = metrics["image"]

    tilt_var = stems.get("tilt_variance", "N/A")
    avg_len  = stems.get("avg_line_length", "N/A")
    v_count  = stems.get("vertical_count", "N/A")
    t_count  = stems.get("tilted_count", "N/A")
    canny_lo = edges.get("canny_low", "N/A")
    canny_hi = edges.get("canny_high", "N/A")
    veg_px   = veg.get("vegetation_pixels", "N/A")
    reg_cnt  = veg.get("region_count", "N/A")
    bright   = img_s.get("brightness", "N/A")
    contrast = img_s.get("contrast", "N/A")

    st.markdown(f"""
    <div class="section-header">
        <div class="section-number">08</div>
        <div class="section-title">Full Analysis Report</div>
        <div class="section-line"></div>
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:1rem;">
        <div class="progress-wrap">
            <div class="metric-label" style="margin-bottom:0.8rem;">🌿 Vegetation Module</div>
            <table class="data-table">
                <tr><td>Coverage</td><td>{veg.get('coverage_pct', 0)}%</td></tr>
                <tr><td>Veg Pixels</td><td>{veg_px}</td></tr>
                <tr><td>Regions</td><td>{reg_cnt}</td></tr>
            </table>
        </div>
        <div class="progress-wrap">
            <div class="metric-label" style="margin-bottom:0.8rem;">📐 Edge Detection Module</div>
            <table class="data-table">
                <tr><td>Edge Density</td><td>{edges.get('edge_density_pct', 0)}%</td></tr>
                <tr><td>Sharpness</td><td>{edges.get('sharpness_score', 0)}</td></tr>
                <tr><td>Canny Thresholds</td><td>{canny_lo} / {canny_hi}</td></tr>
            </table>
        </div>
        <div class="progress-wrap">
            <div class="metric-label" style="margin-bottom:0.8rem;">📏 Stem Detection Module</div>
            <table class="data-table">
                <tr><td>Lines Detected</td><td>{stems.get('line_count', 0)}</td></tr>
                <tr><td>Avg Length</td><td>{avg_len}px</td></tr>
                <tr><td>Tilt Variance</td><td>{tilt_var}</td></tr>
                <tr><td>Vertical / Tilted</td><td>{v_count} / {t_count}</td></tr>
            </table>
        </div>
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-top:1rem;">
        <div class="progress-wrap">
            <div class="metric-label" style="margin-bottom:0.8rem;">🖼️ Image Properties</div>
            <table class="data-table">
                <tr><td>Dimensions</td><td>{img_s.get('width', 'N/A')} × {img_s.get('height', 'N/A')} px</td></tr>
                <tr><td>Megapixels</td><td>{img_s.get('megapixels', 'N/A')} MP</td></tr>
                <tr><td>Brightness</td><td>{bright}</td></tr>
                <tr><td>Contrast</td><td>{contrast}</td></tr>
            </table>
        </div>
        <div class="progress-wrap">
            <div class="metric-label" style="margin-bottom:0.8rem;">🤖 Model Output</div>
            <table class="data-table">
                <tr><td>Status</td><td>{res_label}</td></tr>
                <tr><td>Confidence</td><td>{confidence}%</td></tr>
                <tr><td>Avg Stem Angle</td><td>{angle_display}</td></tr>
                <tr><td>Risk Index</td><td>{risk_display}</td></tr>
                <tr><td>Process Time</td><td>{process_time}ms</td></tr>
            </table>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Footer ──
    st.markdown(f"""
    <div class="footer">
        <span>CROPGUARD AI // CROP LODGING DETECTION</span>
        <span>PIPELINE: VEGETATION · EDGE · STEM · FUSION</span>
        <span>PROCESSED IN {process_time}ms</span>
    </div>
    """, unsafe_allow_html=True)

else:
    # ── Empty State ──
    st.markdown("""
    <div style="text-align:center;padding:5rem 2rem;border:1.5px dashed #1a3328;border-radius:12px;background:#0c1912;">
        <div style="font-size:3rem;margin-bottom:1rem;">🌾</div>
        <div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:700;color:#4a7a5e;margin-bottom:0.75rem;">
            Awaiting Field Image
        </div>
        <div style="font-family:'Space Mono',monospace;font-size:0.72rem;color:#2a4838;letter-spacing:0.08em;line-height:2;">
            SUPPORTED FORMATS: JPG · JPEG · PNG<br>
            PIPELINE: VEGETATION MASKING → EDGE DETECTION → STEM ANGLE ANALYSIS → LODGING CLASSIFICATION<br>
            BEST RESULTS WITH: OVERHEAD SHOTS · CLOSE-UP FIELD ROWS · GOOD LIGHTING
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:2rem;display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;">
        <div class="metric-card">
            <div class="metric-label">Technology</div>
            <div style="font-family:'Syne',sans-serif;font-size:0.9rem;color:#8ab89a;margin-top:0.4rem;">OpenCV · Hough Lines · HSV Masking · Canny Edge</div>
        </div>
        <div class="metric-card info">
            <div class="metric-label">Detection Modes</div>
            <div style="font-family:'Syne',sans-serif;font-size:0.9rem;color:#8ab89a;margin-top:0.4rem;">Healthy · Moderate Lodging · Severe Lodging</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Output Metrics</div>
            <div style="font-family:'Syne',sans-serif;font-size:0.9rem;color:#8ab89a;margin-top:0.4rem;">Stem Angle · Risk Index · Coverage · Confidence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)