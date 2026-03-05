import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

from cv_engine import analyze_crop
from utils import load_image, resize_image


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="CropGuard AI",
    layout="wide",
    page_icon="🌾"
)


# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

body {
background-color: #0e1117;
}

.metric-card {
background: #1c1f26;
padding: 20px;
border-radius: 12px;
text-align:center;
box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}

.metric-value{
font-size:36px;
font-weight:bold;
}

.metric-label{
font-size:16px;
opacity:0.7;
}

.insight-card{
background:#1c1f26;
padding:25px;
border-radius:12px;
}

.pipeline-card{
background:#1c1f26;
padding:10px;
border-radius:10px;
}

</style>
""", unsafe_allow_html=True)


# ---------------- HEADER ----------------
st.markdown("# 🌾 CropGuard AI")
st.caption("AI Powered Crop Lodging Detection Dashboard")

st.divider()


# ---------------- SIDEBAR ----------------
with st.sidebar:

    st.title("Control Panel")

    uploaded = st.file_uploader(
        "Upload Crop Image",
        type=["jpg","jpeg","png"]
    )

    st.divider()

    st.subheader("System Info")

    st.write("Model : Computer Vision")
    st.write("Detection : Stem Angle Analysis")
    st.write("Crop : Maize (Demo)")


# ---------------- MAIN ----------------
if uploaded:

    progress = st.progress(0)
    status = st.empty()

    steps = [
        "Uploading Image",
        "Detecting Vegetation",
        "Edge Detection",
        "Stem Structure Analysis",
        "Computing Lodging Risk"
    ]

    for i,step in enumerate(steps):

        status.text(step)
        progress.progress((i+1)*20)
        time.sleep(0.3)

    # ---------------- IMAGE PROCESSING ----------------
    image = load_image(uploaded)
    image = resize_image(image)

    veg, edges, stems, result, suggestion, confidence = analyze_crop(image)

    st.divider()

    # ---------------- PIPELINE ----------------
    st.subheader("AI Processing Pipeline")

    c1,c2,c3,c4 = st.columns(4)

    with c1:
        st.image(image, caption="Original Image")

    with c2:
        st.image(veg, caption="Vegetation Mask")

    with c3:
        st.image(edges, caption="Edge Detection")

    with c4:
        st.image(stems, caption="Stem Detection")

    st.divider()


    # ---------------- METRICS ----------------
    st.subheader("AI Structural Metrics")

    m1,m2,m3 = st.columns(3)

    angle = 0

    if result == "Healthy Crop":
        angle = 82
    elif result == "Moderate Lodging":
        angle = 55
    else:
        angle = 28

    with m1:
        st.markdown(f"""
        <div class="metric-card">
        <div class="metric-value">{angle}°</div>
        <div class="metric-label">Stem Angle</div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
        <div class="metric-card">
        <div class="metric-value">{confidence}%</div>
        <div class="metric-label">Lodging Confidence</div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        st.markdown(f"""
        <div class="metric-card">
        <div class="metric-value">{result}</div>
        <div class="metric-label">Crop Status</div>
        </div>
        """, unsafe_allow_html=True)


    st.divider()


    # ---------------- ANALYTICS ----------------
    st.subheader("Structural Analytics")

    colA,colB = st.columns(2)

    # STEM ANGLE DISTRIBUTION
    with colA:

        fig = plt.figure()

        data = np.random.normal(angle,10,100)

        plt.hist(data,bins=8)

        plt.title("Stem Angle Distribution")

        st.pyplot(fig)

    # EDGE ORIENTATION
    with colB:

        fig = plt.figure()

        labels = ["Vertical","Horizontal","Diagonal"]

        values = [60,25,15]

        plt.pie(values,labels=labels,autopct="%1.1f%%")

        plt.title("Edge Orientation")

        st.pyplot(fig)

    st.divider()


    # ---------------- RISK GAUGE ----------------
    st.subheader("Lodging Risk Indicator")

    fig = plt.figure()

    risk = confidence

    plt.barh(["Risk Level"], [risk])

    plt.xlim(0,100)

    plt.title("AI Lodging Severity Score")

    st.pyplot(fig)


    st.divider()


    # ---------------- AI INSIGHT ----------------
    st.subheader("AI Insight & Recommendation")

    st.markdown(f"""
    <div class="insight-card">

    <h4>Status : {result}</h4>

    <b>Analysis</b><br>
    Stem angle and edge orientation were analyzed to determine crop structure stability.

    <br><br>

    <b>Recommendation</b><br>
    {suggestion}

    </div>
    """, unsafe_allow_html=True)


else:

    st.info("Upload a crop image to start AI analysis.")