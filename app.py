import streamlit as st
from diffusers import StableDiffusionPipeline
import torch

# Set up the page configuration
st.set_page_config(
    page_title="Text2Image AI",
    layout="wide"
)

# Load the fine-tuned model from Hugging Face
@st.cache_resource
def load_fine_tuned_model():
    model = StableDiffusionPipeline.from_pretrained(
        "kunaltibe/text2image-finetuned",   # <-- Hugging Face model path
        torch_dtype=torch.float16,
    ).to("cuda")
    return model

fine_tuned_model = load_fine_tuned_model()

# Page title
st.title("🖼️ Text2Image AI - Fine-Tuned Model")

# User input
prompt = st.text_input(
    "Enter your creative text prompt:",
    placeholder="e.g., An astronaut riding a horse on Mars"
)

# Button to generate image
if st.button("Generate Image"):
    if prompt.strip() == "":
        st.warning("⚠️ Please enter a prompt!")
    else:
        with st.spinner("Generating your masterpiece... 🎨"):
            result = fine_tuned_model(prompt, num_inference_steps=30)
            generated_image = result.images[0]

        # Display generated image
        st.image(generated_image, caption="Generated Image", use_column_width=True)

        # Optional: Download button
        st.download_button(
            label="Download Image",
            data=generated_image,
            file_name="generated_image.png",
            mime="image/png"
        )
