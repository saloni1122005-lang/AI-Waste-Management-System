import streamlit as st
from PIL import Image
import waste_management_dl as dl

# ---- PAGE CONFIG ----
st.set_page_config(
    page_title="Al-Driven Circular Economy Optimization System",
    page_icon="♻️",
    layout="centered"
)

# ---- CUSTOM CSS ----
st.markdown("""
<style>
.main {background-color: #f6f9f7;}
h1 {color: #2e7d32;}
h2, h3 {color: #388e3c;}
.stButton>button {
    background-color: #2e7d32;
    color: white;
    border-radius: 8px;
    padding: 8px 20px;
}
</style>
""", unsafe_allow_html=True)

st.title("♻️ Al-Driven Circular Economy Optimization System")
st.write("Upload an image or manually enter waste details")

# --------------------------------------------------
# RULE-BASED CLASSIFIER (SMART LOGIC)
# --------------------------------------------------

def classify_logic(waste_type, toxicity, recyclable):

    # Hazardous condition
    if waste_type.lower() == "e-waste" or toxicity.lower() == "high":
        return "Hazardous", "Dispose at certified hazardous waste facility"

    # Recyclable condition
    if recyclable:
        return "Recyclable", "Send to authorized recycling center"

    # Default
    return "Residual", "Dispose in general waste bin"


# --------------------------------------------------
# IMAGE RECOGNITION
# --------------------------------------------------

st.header("📸 Image Recognition")

img = st.file_uploader(
    "Upload waste image",
    type=["jpg", "png", "jpeg", "webp"]
)

if img:
    image = Image.open(img)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    image.save("temp.jpg")

    detected = dl.analyze_image("temp.jpg")

    if detected:
        waste_type = detected.get("waste_type", "unknown")
        toxicity = detected.get("toxicity_level", "unknown")
        recyclable = detected.get("recyclability", 0)

        category, recommendation = classify_logic(
            waste_type,
            toxicity,
            recyclable == 1
        )

        st.subheader("🔍 Detection Result")
        st.success(f"Waste Type: {waste_type.upper()}")
        st.info(f"Toxicity Level: {toxicity.upper()}")
        st.info(f"Recyclable: {'Yes' if recyclable == 1 else 'No'}")

        st.subheader("♻️ Disposal Guidance")
        st.write("**Category:**", category)
        st.write("**Recommendation:**", recommendation)


# --------------------------------------------------
# MANUAL INPUT SECTION (UPGRADED)
# --------------------------------------------------

st.header("✍️ Manual Input")

col1, col2, col3 = st.columns(3)

with col1:
    waste_type = st.selectbox(
        "Waste Type",
        ["plastic", "paper", "metal", "glass", "organic", "e-waste"]
    )

with col2:
    toxicity = st.selectbox(
        "Toxicity Level",
        ["low", "medium", "high"]
    )

with col3:
    recyclable = st.selectbox(
        "Recyclable?",
        ["Yes", "No"]
    )

if st.button("Predict Disposal Method"):

    recyclable_bool = recyclable == "Yes"

    category, recommendation = classify_logic(
        waste_type,
        toxicity,
        recyclable_bool
    )

    st.subheader("📌 Result")
    st.success(f"Waste Type: {waste_type.upper()}")
    st.info(f"Toxicity Level: {toxicity.upper()}")
    st.info(f"Recyclable: {recyclable}")

    st.subheader("♻️ Disposal Guidance")
    st.write("**Category:**", category)
    st.write("**Recommendation:**", recommendation)