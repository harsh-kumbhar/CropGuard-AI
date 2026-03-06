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
# CSS — DM Sans professional font
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600;9..40,700&family=DM+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #080d0a !important;
    color: #dff0e6 !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse at 20% 0%, #0a2116 0%, #080d0a 55%) !important;
}

#MainMenu, footer, header, [data-testid="stToolbar"] { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

.block-container {
    padding: 2rem 3rem 4rem !important;
    max-width: 1400px !important;
}

/* ── Hero ── */
.hero-header {
    display: flex; align-items: center; gap: 1.5rem;
    padding: 2.5rem 0 1.5rem;
    border-bottom: 1px solid #1a3328;
    margin-bottom: 2.5rem;
}
.hero-badge {
    background: linear-gradient(135deg, #00ff88, #00cc6a);
    color: #080d0a;
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem; font-weight: 500;
    letter-spacing: 0.1em; padding: 0.3rem 0.75rem;
    border-radius: 4px; text-transform: uppercase;
}
.hero-title {
    font-family: 'DM Sans', sans-serif;
    font-size: 2.6rem; font-weight: 700;
    color: #f0fff5; letter-spacing: -0.02em; line-height: 1.1;
}
.hero-title span { color: #00ff88; }
.hero-subtitle {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.88rem; font-weight: 400;
    color: #7ab893; margin-top: 0.35rem;
}
.hero-right { margin-left: auto; text-align: right; }
.system-status {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem; color: #00ff88;
    letter-spacing: 0.06em;
    display: flex; align-items: center; gap: 0.4rem; justify-content: flex-end;
}
.system-status::before { content: '●'; animation: pulse 2s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.25} }

/* ── Section headers ── */
.section-header {
    display: flex; align-items: center; gap: 0.75rem;
    margin: 2.5rem 0 1.2rem;
}
.section-number {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem; font-weight: 500; color: #00ff88;
    background: #0c2018; border: 1px solid #1e4030;
    padding: 0.2rem 0.55rem; border-radius: 4px; letter-spacing: 0.08em;
}
.section-title {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1.1rem; font-weight: 600; color: #dff0e6 !important;
    letter-spacing: -0.01em;
}
.section-line {
    flex: 1; height: 1px;
    background: linear-gradient(to right, #1a3328, transparent);
}

/* ── Metric cards ── */
.metric-grid {
    display: grid; grid-template-columns: repeat(4, 1fr);
    gap: 1rem; margin-bottom: 1.5rem;
}
.metric-card {
    background: #0c1912; border: 1px solid #1a3328;
    border-radius: 10px; padding: 1.2rem 1.4rem;
    position: relative; overflow: hidden;
    transition: border-color 0.2s, transform 0.2s;
}
.metric-card:hover { border-color: #2a5040; transform: translateY(-2px); }
.metric-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #00ff88, transparent);
}
.metric-card.warning::before { background: linear-gradient(90deg, #facc15, transparent); }
.metric-card.danger::before  { background: linear-gradient(90deg, #f87171, transparent); }
.metric-card.info::before    { background: linear-gradient(90deg, #38bdf8, transparent); }
.metric-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.72rem; font-weight: 500;
    color: #8bbf9a !important; text-transform: uppercase;
    letter-spacing: 0.03em; margin-bottom: 0.5rem;
}
.metric-value {
    font-family: 'DM Sans', sans-serif;
    font-size: 2rem; font-weight: 700; color: #00ff88;
    line-height: 1.1; letter-spacing: -0.02em;
}
.metric-card.warning .metric-value { color: #facc15; }
.metric-card.danger  .metric-value { color: #f87171; }
.metric-card.info    .metric-value { color: #38bdf8; }
.metric-unit {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.7rem; font-weight: 400; color: #6b9e7d !important; margin-top: 0.2rem;
}

/* ── Result banner ── */
.result-banner {
    background: #0c1912; border: 1px solid #1a3328;
    border-radius: 12px; padding: 1.6rem 2rem;
    display: flex; align-items: center; gap: 2rem;
    margin: 1.5rem 0; position: relative; overflow: hidden;
}
.result-banner::after {
    content: ''; position: absolute; inset: 0;
    background: radial-gradient(circle at 90% 50%, rgba(0,255,136,0.04), transparent 60%);
    pointer-events: none;
}
.result-banner.warning::after { background: radial-gradient(circle at 90% 50%, rgba(250,204,21,0.05), transparent 60%); }
.result-banner.danger::after  { background: radial-gradient(circle at 90% 50%, rgba(248,113,113,0.05), transparent 60%); }
.result-status-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.7rem; font-weight: 500; color: #8bbf9a !important;
    letter-spacing: 0.06em; text-transform: uppercase; margin-bottom: 0.35rem;
}
.result-status-value {
    font-family: 'DM Sans', sans-serif;
    font-size: 1.7rem; font-weight: 700; color: #00ff88; letter-spacing: -0.01em;
}
.result-banner.warning .result-status-value { color: #facc15; }
.result-banner.danger  .result-status-value { color: #f87171; }
.result-meta {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem; color: #7aaa8a !important;
    margin-top: 0.5rem; letter-spacing: 0.04em;
}
.result-suggestion {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.92rem; font-weight: 400;
    color: #c0dfc8 !important; line-height: 1.7;
    flex: 1; padding-left: 2rem; border-left: 1px solid #1a3328;
}

/* ── Progress bars ── */
.progress-wrap {
    background: #0c1912; border: 1px solid #1a3328;
    border-radius: 8px; padding: 1.1rem 1.3rem; margin-bottom: 0.85rem;
}
.progress-label {
    display: flex; justify-content: space-between;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.78rem; font-weight: 500;
    color: #9ecfae !important; margin-bottom: 0.55rem;
}
.progress-bar-bg {
    height: 6px; background: #0f1f18; border-radius: 3px; overflow: hidden;
}
.progress-bar-fill {
    height: 100%; border-radius: 3px;
    background: linear-gradient(90deg, #00cc6a, #00ff88); transition: width 0.8s ease;
}
.progress-bar-fill.warning { background: linear-gradient(90deg, #ca8a04, #facc15); }
.progress-bar-fill.danger  { background: linear-gradient(90deg, #dc2626, #f87171); }
.progress-bar-fill.info    { background: linear-gradient(90deg, #0284c7, #38bdf8); }

/* ── Image tags ── */
.img-tag {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.72rem; font-weight: 600;
    color: #9ecfae !important; text-transform: uppercase;
    letter-spacing: 0.04em; padding: 0.3rem 0 0.4rem;
    border-bottom: 1px solid #1a3328; margin-bottom: 0.5rem;
    display: flex; justify-content: space-between; align-items: center;
}
.img-tag span {
    font-family: 'DM Mono', monospace; font-size: 0.62rem; color: #00ff88;
    background: #0c2018; border: 1px solid #1e4030;
    padding: 0.15rem 0.45rem; border-radius: 3px;
}

/* ── Panel + table ── */
.panel {
    background: #0c1912; border: 1px solid #1a3328;
    border-radius: 10px; padding: 1.1rem 1.3rem; margin-bottom: 0.85rem;
}
.panel-title {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.78rem; font-weight: 600; color: #9ecfae !important;
    text-transform: uppercase; letter-spacing: 0.05em;
    margin-bottom: 0.75rem; padding-bottom: 0.55rem; border-bottom: 1px solid #1a3328;
}
.data-table { width: 100%; border-collapse: collapse; font-family: 'DM Sans', sans-serif; }
.data-table td {
    padding: 0.5rem 0.6rem; font-size: 0.82rem;
    border-bottom: 1px solid #0f1f18; color: #c0dfc8 !important;
}
.data-table td:first-child { color: #8bbf9a !important; font-weight: 500; }
.data-table td:last-child  { color: #e2f5e8 !important; font-weight: 600; text-align: right; }
.data-table tr:last-child td { border-bottom: none; }
.data-table tr:hover td { background: #0a1810; }

/* ── Footer ── */
.footer {
    margin-top: 4rem; padding-top: 1.2rem;
    border-top: 1px solid #1a3328;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.76rem; color: #6b9e7d !important;
    display: flex; justify-content: space-between;
}
.footer strong { color: #8bbf9a !important; font-weight: 600; }

/* ── Streamlit overrides ── */
[data-testid="stFileUploader"] {
    background: #0c1912 !important;
    border: 1.5px dashed #1e4030 !important;
    border-radius: 10px !important; padding: 1rem !important;
}
[data-testid="stFileUploaderDropzone"] { background: transparent !important; }
[data-testid="stFileUploaderDropzoneInstructions"] p,
[data-testid="stFileUploaderDropzoneInstructions"] span {
    color: #8bbf9a !important;
    font-family: 'DM Sans', sans-serif !important; font-size: 0.85rem !important;
}
[data-testid="stAlert"] p {
    font-family: 'DM Sans', sans-serif !important; color: #dff0e6 !important;
}
.stImage > img { border-radius: 8px; border: 1px solid #1a3328; }
hr { border-color: #1a3328 !important; }
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
        <div class="hero-subtitle">Crop Lodging Detection System &nbsp;·&nbsp; Computer Vision Pipeline v2.0</div>
    </div>
    <div class="hero-right">
        <div class="system-status">SYSTEM ONLINE</div>
        <div style="font-family:'DM Sans',sans-serif;font-size:0.75rem;color:#6b9e7d;margin-top:0.3rem;">
            Modules: Vegetation · Edge · Stem
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
    "Upload Crop Field Image", type=["jpg", "jpeg", "png"],
    help="Supports JPG, JPEG, PNG.", label_visibility="collapsed"
)

# ─────────────────────────────────────────────
# Analysis
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

    # ── Preview ──
    st.markdown("""
    <div class="section-header">
        <div class="section-number">02</div>
        <div class="section-title">Input Preview</div>
        <div class="section-line"></div>
    </div>
    """, unsafe_allow_html=True)

    col_img, col_info = st.columns([2, 1])
    with col_img:
        st.markdown('<div class="img-tag">Original Image <span>RAW INPUT</span></div>', unsafe_allow_html=True)
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
    with col_info:
        h, w = image.shape[:2]
        st.markdown(f"""
        <div style="display:flex;flex-direction:column;gap:0.75rem;margin-top:0.5rem;">
            <div class="progress-wrap">
                <div class="metric-label">Image Dimensions</div>
                <div style="font-family:'DM Sans',sans-serif;font-size:1.2rem;font-weight:700;color:#38bdf8;letter-spacing:-0.01em;">{w} × {h}</div>
                <div class="metric-unit">pixels</div>
            </div>
            <div class="progress-wrap">
                <div class="metric-label">Resolution</div>
                <div style="font-family:'DM Sans',sans-serif;font-size:1.2rem;font-weight:700;color:#38bdf8;letter-spacing:-0.01em;">{round((w*h)/1_000_000, 2)} MP</div>
                <div class="metric-unit">megapixels</div>
            </div>
            <div class="progress-wrap">
                <div class="metric-label">File Format</div>
                <div style="font-family:'DM Sans',sans-serif;font-size:1.2rem;font-weight:700;color:#38bdf8;letter-spacing:-0.01em;">{uploaded_file.type.split('/')[1].upper()}</div>
                <div class="metric-unit">image format</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Run ──
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
        process_time = round((time.time() - t_start) * 1000, 1)

    st.success(f"Pipeline complete in {process_time} ms")

    r          = metrics["result"]
    res_label  = r["label"]
    avg_angle  = r.get("avg_angle")
    risk_score = r.get("risk_score", 0)
    angle_display = f"{avg_angle}°" if avg_angle is not None else "N/A"
    risk_display  = f"{risk_score}%" if risk_score is not None else "N/A"

    banner_class = "warning" if "Moderate" in res_label else ("danger" if "Severe" in res_label else "")

    # ── Result ──
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
            <div class="result-meta">Confidence: {confidence}% &nbsp;·&nbsp; Avg Angle: {angle_display} &nbsp;·&nbsp; Risk Index: {risk_display}</div>
        </div>
        <div class="result-suggestion">{suggestion}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Key Metrics ──
    veg   = metrics["vegetation"]
    edges = metrics["edges"]
    stems = metrics["stems"]

    conf_class  = "danger" if confidence < 50 else ("warning" if confidence < 75 else "")
    risk_class  = "danger" if (risk_score or 0) > 65 else ("warning" if (risk_score or 0) > 35 else "")
    angle_class = "danger" if (avg_angle or 90) < 40 else ("warning" if (avg_angle or 90) < 70 else "")

    st.markdown(f"""
    <div class="section-header">
        <div class="section-number">05</div>
        <div class="section-title">Key Metrics</div>
        <div class="section-line"></div>
    </div>
    <div class="metric-grid">
        <div class="metric-card {conf_class}">
            <div class="metric-label">Confidence Score</div>
            <div class="metric-value">{confidence}%</div>
            <div class="metric-unit">Detection reliability</div>
        </div>
        <div class="metric-card {angle_class}">
            <div class="metric-label">Avg Stem Angle</div>
            <div class="metric-value">{angle_display}</div>
            <div class="metric-unit">90° = fully vertical</div>
        </div>
        <div class="metric-card {risk_class}">
            <div class="metric-label">Risk Index</div>
            <div class="metric-value">{risk_display}</div>
            <div class="metric-unit">Lodging severity</div>
        </div>
        <div class="metric-card info">
            <div class="metric-label">Vegetation Coverage</div>
            <div class="metric-value">{veg.get('coverage_pct', 0)}%</div>
            <div class="metric-unit">Green area in frame</div>
        </div>
    </div>
    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-label">Stem Lines</div>
            <div class="metric-value">{stems.get('line_count', 0)}</div>
            <div class="metric-unit">Hough lines detected</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Upright Stems</div>
            <div class="metric-value">{stems.get('vertical_pct', 0)}%</div>
            <div class="metric-unit">Percent vertical</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Edge Density</div>
            <div class="metric-value">{edges.get('edge_density_pct', 0)}%</div>
            <div class="metric-unit">Structural complexity</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Sharpness</div>
            <div class="metric-value">{edges.get('sharpness_score', 0)}</div>
            <div class="metric-unit">Laplacian variance</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Progress bars ──
    conf_bar_class = "danger" if confidence < 50 else ("warning" if confidence < 75 else "")
    risk_bar_class = "danger" if (risk_score or 0) > 65 else ("warning" if (risk_score or 0) > 35 else "")
    veg_bar  = min(100, veg.get("coverage_pct", 0))
    vert_bar = stems.get("vertical_pct", 0)

    st.markdown(f"""
    <div class="section-header">
        <div class="section-number">06</div>
        <div class="section-title">Signal Strength Breakdown</div>
        <div class="section-line"></div>
    </div>
    <div class="progress-wrap">
        <div class="progress-label"><span>Detection Confidence</span><span>{confidence}%</span></div>
        <div class="progress-bar-bg"><div class="progress-bar-fill {conf_bar_class}" style="width:{confidence}%"></div></div>
    </div>
    <div class="progress-wrap">
        <div class="progress-label"><span>Vegetation Coverage</span><span>{veg_bar}%</span></div>
        <div class="progress-bar-bg"><div class="progress-bar-fill info" style="width:{veg_bar}%"></div></div>
    </div>
    <div class="progress-wrap">
        <div class="progress-label"><span>Upright Stem Ratio</span><span>{vert_bar}%</span></div>
        <div class="progress-bar-bg"><div class="progress-bar-fill" style="width:{vert_bar}%"></div></div>
    </div>
    <div class="progress-wrap">
        <div class="progress-label"><span>Lodging Risk Index</span><span>{risk_score or 0}%</span></div>
        <div class="progress-bar-bg"><div class="progress-bar-fill {risk_bar_class}" style="width:{risk_score or 0}%"></div></div>
    </div>
    """, unsafe_allow_html=True)

    # ── CV Outputs ──
    st.markdown("""
    <div class="section-header">
        <div class="section-number">07</div>
        <div class="section-title">Computer Vision Outputs</div>
        <div class="section-line"></div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="img-tag">Vegetation Mask <span>HSV FILTER</span></div>', unsafe_allow_html=True)
        st.image(veg_mask, use_container_width=True)
    with c2:
        st.markdown('<div class="img-tag">Vegetation Density Map <span>HEATMAP</span></div>', unsafe_allow_html=True)
        st.image(cv2.cvtColor(veg_density, cv2.COLOR_BGR2RGB), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<div class="img-tag">Edge Detection <span>AUTO CANNY</span></div>', unsafe_allow_html=True)
        st.image(cv2.cvtColor(edges_colored, cv2.COLOR_BGR2RGB), use_container_width=True)
    with c4:
        st.markdown('<div class="img-tag">Stem Analysis <span>HOUGH LINES</span></div>', unsafe_allow_html=True)
        st.image(cv2.cvtColor(stem_img, cv2.COLOR_BGR2RGB), use_container_width=True)

    st.markdown('<div class="img-tag" style="margin-top:0.75rem;">Vegetation Overlay <span>MASK ON ORIGINAL</span></div>', unsafe_allow_html=True)
    st.image(cv2.cvtColor(veg_overlay, cv2.COLOR_BGR2RGB), use_container_width=True)

    # ── Full Report ──
    img_s    = metrics["image"]
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

    st.markdown("""
    <div class="section-header">
        <div class="section-number">08</div>
        <div class="section-title">Full Analysis Report</div>
        <div class="section-line"></div>
    </div>
    """, unsafe_allow_html=True)

    rt1, rt2, rt3 = st.columns(3)
    with rt1:
        st.markdown(f"""
        <div class="panel">
            <div class="panel-title">🌿 Vegetation Module</div>
            <table class="data-table">
                <tr><td>Coverage</td><td>{veg.get('coverage_pct', 0)}%</td></tr>
                <tr><td>Veg Pixels</td><td>{veg_px:,}</td></tr>
                <tr><td>Regions Found</td><td>{reg_cnt}</td></tr>
            </table>
        </div>""", unsafe_allow_html=True)
    with rt2:
        st.markdown(f"""
        <div class="panel">
            <div class="panel-title">📐 Edge Detection Module</div>
            <table class="data-table">
                <tr><td>Edge Density</td><td>{edges.get('edge_density_pct', 0)}%</td></tr>
                <tr><td>Sharpness Score</td><td>{edges.get('sharpness_score', 0)}</td></tr>
                <tr><td>Canny Low / High</td><td>{canny_lo} / {canny_hi}</td></tr>
            </table>
        </div>""", unsafe_allow_html=True)
    with rt3:
        st.markdown(f"""
        <div class="panel">
            <div class="panel-title">📏 Stem Detection Module</div>
            <table class="data-table">
                <tr><td>Lines Detected</td><td>{stems.get('line_count', 0)}</td></tr>
                <tr><td>Avg Line Length</td><td>{avg_len} px</td></tr>
                <tr><td>Tilt Variance</td><td>{tilt_var}</td></tr>
                <tr><td>Vertical / Tilted</td><td>{v_count} / {t_count}</td></tr>
            </table>
        </div>""", unsafe_allow_html=True)

    rb1, rb2 = st.columns(2)
    with rb1:
        st.markdown(f"""
        <div class="panel">
            <div class="panel-title">🖼️ Image Properties</div>
            <table class="data-table">
                <tr><td>Dimensions</td><td>{img_s.get('width', 'N/A')} × {img_s.get('height', 'N/A')} px</td></tr>
                <tr><td>Megapixels</td><td>{img_s.get('megapixels', 'N/A')} MP</td></tr>
                <tr><td>Brightness</td><td>{bright}</td></tr>
                <tr><td>Contrast</td><td>{contrast}</td></tr>
            </table>
        </div>""", unsafe_allow_html=True)
    with rb2:
        st.markdown(f"""
        <div class="panel">
            <div class="panel-title">🤖 Model Output</div>
            <table class="data-table">
                <tr><td>Lodging Status</td><td>{res_label}</td></tr>
                <tr><td>Confidence</td><td>{confidence}%</td></tr>
                <tr><td>Avg Stem Angle</td><td>{angle_display}</td></tr>
                <tr><td>Risk Index</td><td>{risk_display}</td></tr>
                <tr><td>Process Time</td><td>{process_time} ms</td></tr>
            </table>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="footer">
        <strong>CropGuard AI</strong> &nbsp;·&nbsp; Crop Lodging Detection System
        <span>Pipeline: Vegetation · Edge · Stem · Fusion</span>
        <span>Processed in {process_time} ms</span>
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align:center;padding:4.5rem 2rem;border:1.5px dashed #1a3328;border-radius:12px;background:#0c1912;">
        <div style="font-size:2.8rem;margin-bottom:0.75rem;">🌾</div>
        <div style="font-family:'DM Sans',sans-serif;font-size:1.4rem;font-weight:600;color:#c8e8d4;margin-bottom:0.6rem;letter-spacing:-0.01em;">
            Upload a field image to begin
        </div>
        <div style="font-family:'DM Sans',sans-serif;font-size:0.88rem;font-weight:400;color:#8bbf9a;line-height:2;">
            Supported formats: JPG · JPEG · PNG<br>
            Pipeline: Vegetation Masking → Edge Detection → Stem Angle Analysis → Lodging Classification<br>
            Best results with overhead shots or close-up field rows in good lighting
        </div>
    </div>
    <div style="margin-top:1rem;display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;">
        <div class="panel">
            <div class="panel-title">Technology Stack</div>
            <div style="font-family:'DM Sans',sans-serif;font-size:0.88rem;color:#c0dfc8;line-height:1.8;">OpenCV · Hough Lines · HSV Masking · Canny Edge Detection</div>
        </div>
        <div class="panel">
            <div class="panel-title">Detection Modes</div>
            <div style="font-family:'DM Sans',sans-serif;font-size:0.88rem;color:#c0dfc8;line-height:1.8;">Healthy Crop · Moderate Lodging · Severe Lodging</div>
        </div>
        <div class="panel">
            <div class="panel-title">Output Metrics</div>
            <div style="font-family:'DM Sans',sans-serif;font-size:0.88rem;color:#c0dfc8;line-height:1.8;">Stem Angle · Risk Index · Vegetation Coverage · Confidence Score</div>
        </div>
    </div>
    """, unsafe_allow_html=True)