import os
from dotenv import load_dotenv
import streamlit as st
from diffusers import StableDiffusionPipeline
import torch

# 🔐 Load Hugging Face API token
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

# 📄 Page configuration
st.set_page_config(page_title="🎨 AI Image Generator", layout="centered")
st.title("🧠 Text-to-Image Generator with Stable Diffusion")
st.markdown("Enter a text prompt below to generate an image using Stable Diffusion.")

# 📝 User input prompt
prompt = st.text_input("🖊️ Enter your image prompt:", placeholder="e.g., A robot walking through Times Square")

# ⚙️ Load the Stable Diffusion model (cached)
@st.cache_resource(show_spinner="Loading the Stable Diffusion model...")
def load_sd_pipeline():
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        use_auth_token=hf_token,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    return pipe

# 🔄 Generate image on prompt
if prompt:
    st.info(f"Generating image for prompt: **{prompt}**")
    with st.spinner("Creating your image..."):
        pipe = load_sd_pipeline()
        image = pipe(prompt).images[0]
        st.image(image, caption=prompt, use_column_width=True)
