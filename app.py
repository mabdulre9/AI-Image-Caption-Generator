import streamlit as st
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="AI Image Captioning",
    page_icon="🖼️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Sidebar Branding

st.sidebar.title("🚀 Four Zero Productions")
st.sidebar.write("Bringing AI Innovations to Life.")
st.sidebar.markdown("---")
st.sidebar.write("🔹 **Developed by Me 😊**")
st.sidebar.write("📩 Contact: fourzero@ai.com")

# Load model and processor from local path
@st.cache_resource
def load_model():
    model_path = "C:/Users/FAYSAL COMPUTER/Desktop/Image Captioning with Generative AI/vit-gpt2-image-captioning"

    model = VisionEncoderDecoderModel.from_pretrained(model_path)
    processor = ViTImageProcessor.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, processor, tokenizer

model, processor, tokenizer = load_model()

# Main UI
st.markdown("<h1 style='text-align: center; color: #4A90E2;'>🖼️ AI Image Captioning</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #555;'>📷 Upload an image and let AI generate a caption for you!</h4>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    # Display the uploaded image
    st.image(image, caption="✅ Uploaded Image", width=150, output_format="JPEG")

    # Process image
    pixel_values = processor(image, return_tensors="pt").pixel_values

    # Generate caption
    outputs = model.generate(pixel_values)
    caption = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Display caption in a styled box
    st.markdown(
        f"""
        <div style="
            background-color: #FFF3CD;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            font-size: 1.2em;
            font-weight: bold;
            color: #856404;">
            📝 {caption}
        </div>
        """,
        unsafe_allow_html=True
    )

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>🔹 <b>Powered by Four Zero Productions</b> | AI for Creativity & Innovation 🚀</p>", unsafe_allow_html=True)
