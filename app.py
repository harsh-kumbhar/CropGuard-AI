import streamlit as st
import numpy as np
from PIL import Image
import cv2

from utils import resize_image
from cv_engine import analyze_crop


# ---------------------------------------------------
# Page Config
# ---------------------------------------------------

st.set_page_config(
    page_title="CropGuard AI",
    page_icon="🌾",
    layout="wide"
)

st.title("🌾 CropGuard AI - Crop Lodging Detection System")

st.write(
"""
Upload a crop field image and the system will analyze
plant structure using computer vision techniques.
"""
)

st.divider()

# ---------------------------------------------------
# Upload Image
# ---------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Crop Image",
    type=["jpg", "jpeg", "png"]
)

# ---------------------------------------------------
# If Image Uploaded
# ---------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    image = np.array(image)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    image = resize_image(image)

    st.subheader("Uploaded Image")

    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)

    st.divider()

    # ---------------------------------------------------
    # Run Analysis
    # ---------------------------------------------------

    with st.spinner("Analyzing crop structure..."):

        veg, edges, stems, result, suggestion, confidence = analyze_crop(image)

    st.success("Analysis Complete")

    st.divider()

    # ---------------------------------------------------
    # Display Results
    # ---------------------------------------------------

    st.subheader("Processing Results")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Vegetation Mask")
        st.image(veg, use_column_width=True)

    with col2:
        st.write("Edge Detection")
        st.image(edges, use_column_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.write("Stem Detection")
        st.image(cv2.cvtColor(stems, cv2.COLOR_BGR2RGB), use_column_width=True)

    with col4:
        st.write("Original Image")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)

    st.divider()

    # ---------------------------------------------------
    # Metrics
    # ---------------------------------------------------

    st.subheader("Detection Metrics")

    m1, m2 = st.columns(2)

    with m1:
        st.metric(
            label="Lodging Status",
            value=result
        )

    with m2:
        st.metric(
            label="Confidence",
            value=str(confidence) + "%"
        )

    st.divider()

    # ---------------------------------------------------
    # AI Recommendation
    # ---------------------------------------------------

    st.subheader("AI Recommendation")

    if "Healthy" in result:

        st.success(suggestion)

    elif "Moderate" in result:

        st.warning(suggestion)

    elif "Severe" in result:

        st.error(suggestion)

    else:

        st.info(suggestion)


# ---------------------------------------------------
# If No Image Uploaded
# ---------------------------------------------------

else:

    st.info("Upload an image to start crop lodging analysis.")