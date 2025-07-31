# app.py
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.set_page_config(
    page_title="Eye Disease Classifier",
    page_icon="🧿",
    layout="centered",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("ABC Eye-Dentification (1).keras")

model = load_model()

# ----- Sidebar -----
with st.sidebar:
    st.title(" About")
    st.markdown(
        """
        This tool uses a deep learning model to classify eye images into:
        -  Cataract  
        -  Diabetic Retinopathy  
        -  Glaucoma  
        -  Normal

        **Upload a retinal scan** to get started.
        """
    )
    st.markdown("---")

# ----- Main Header -----
st.markdown("<h1 style='text-align: center;'>🧿 Eye Disease Classifier</h1>", unsafe_allow_html=True)
st.write("Upload an eye image (JPEG or PNG). The model will analyze it and predict the condition.")

# ----- Category Mapping -----
categories = {
    0: "Cataract",
    1: "Diabetic Retinopathy",
    2: "Glaucoma",
    3: "Normal",
}

# ----- Upload and Display -----
uploaded_file = st.file_uploader("Choose an eye image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

    # Preprocess
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = int(np.argmax(prediction))
    confidence = prediction[0][predicted_class]

    # ----- Display Result -----
    st.markdown("---")
    st.subheader("🧠 Model Prediction")
    st.success(f"**{categories[predicted_class]}** ({confidence * 100:.2f}% confidence)")

    st.markdown("### 📊 Class Probabilities")
    for i, prob in enumerate(prediction[0]):
        st.write(f"{categories[i]} — {prob*100:.2f}%")
        st.progress(float(prob))

else:
    st.info("👆 Please upload an image to start diagnosis.")
