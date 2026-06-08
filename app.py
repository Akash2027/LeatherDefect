import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import pandas as pd
import cv2

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Leather Defect Detection & Localization",
    page_icon="🔍",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

.title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    color: #2563EB;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #6B7280;
    margin-bottom: 25px;
}

.stButton>button {
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------

st.markdown(
    """
    <div class="title">
    🔍 Leather Defect Detection & Localization
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    AI-Powered Industrial Quality Inspection using YOLO
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model():
    return YOLO("modelv1.pt")

model = load_model()

# ---------------- FILE UPLOADER ----------------

uploaded_file = st.file_uploader(
    "📂 Upload a Leather Image",
    type=["jpg", "jpeg", "png"]
)

# ---------------- PREDICTION ----------------

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:

        image.save(temp_file.name)

        with st.spinner("🔍 Detecting defects..."):

            results = model.predict(
                source=temp_file.name,
                conf=0.25
            )

    result_image = results[0].plot()

    st.markdown("## Detection Results")

    spacer1, col1, col2, spacer2 = st.columns([1, 3, 3, 1])

    with col1:

        st.markdown(
            "<h4 style='text-align:center;'>Original Image</h4>",
            unsafe_allow_html=True
        )

        st.image(
            image,
            width=350
        )

    with col2:

        st.markdown(
            "<h4 style='text-align:center;'>Detected Defects</h4>",
            unsafe_allow_html=True
        )

        st.image(
            result_image,
            width=350
        )

    st.markdown("---")

    boxes = results[0].boxes

    if boxes is not None and len(boxes) > 0:

        st.success(f"Detected {len(boxes)} defect(s)")

        names = model.names

        detected_classes = []
        confidence_data = []

        for box in boxes:

            cls = int(box.cls[0])
            conf = float(box.conf[0])

            defect_name = names[cls]

            detected_classes.append(defect_name)

            confidence_data.append({
                "Defect": defect_name,
                "Confidence (%)": round(conf * 100, 2)
            })

        st.markdown("### Defect Summary")

        for defect in sorted(set(detected_classes)):

            count = detected_classes.count(defect)

            st.markdown(
                f"✅ **{defect}** : {count}"
            )

        st.markdown("---")

        st.markdown("### Confidence Scores")

        df = pd.DataFrame(confidence_data)

        left_space, center_col, right_space = st.columns([1, 2, 1])

        with center_col:

            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )

        st.markdown("---")

        # DOWNLOAD BUTTON

        result_bgr = cv2.cvtColor(
            result_image,
            cv2.COLOR_RGB2BGR
        )

        _, buffer = cv2.imencode(
            ".jpg",
            result_bgr
        )

        left_space, center_col, right_space = st.columns([1, 2, 1])

        with center_col:

            st.download_button(
                label="📥 Download Detection Result",
                data=buffer.tobytes(),
                file_name="detected_defects.jpg",
                mime="image/jpeg",
                use_container_width=True
            )

    else:

        st.success("No Defects Detected")

st.markdown("---")

st.caption(
    "Powered by YOLO and Streamlit"
)