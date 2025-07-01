import os
from dotenv import load_dotenv
import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from langsmith import traceable, Client  # 🔍 LangSmith tracking

# 🔐 Load environment variables
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
langsmith_api_key = os.getenv("LANGCHAIN_API_KEY")
project_name = os.getenv("LANGCHAIN_PROJECT", "GenAI-Portfolio")

# 🌐 Initialize LangSmith client
client = Client(api_key=langsmith_api_key)

# 📄 Page config
st.set_page_config(page_title="🎨 AI Image Generator", layout="centered")
st.title("🧠 Text-to-Image Generator with Stable Diffusion")
st.markdown("Enter a text prompt below to generate an image using Stable Diffusion.")

# 📝 User prompt input
prompt = st.text_input("🖊️ Enter your image prompt:", placeholder="e.g., A robot walking through Times Square")

# ⚙️ Load pipeline
@st.cache_resource(show_spinner="Loading the Stable Diffusion model... Please wait ⏳")
def load_sd_pipeline():
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        token=hf_token,
        torch_dtype=torch.float32
    )
    pipe.to("cpu")
    return pipe

# 🔄 Generate & track image
if prompt:
    st.info(f"Generating image for prompt: **{prompt}**")
    with st.spinner("Creating your image..."):
        pipe = load_sd_pipeline()
        image = pipe(prompt).images[0]

        # 🔎 Track prompt and metadata in LangSmith
        client.create_run(
            name="stable-diffusion-image-generation",
            inputs={"prompt": prompt},
            outputs={"status": "image_generated"},
            project_name=project_name,
            tags=["stable-diffusion", "streamlit", "image"]
        )

        st.image(image, caption=prompt, use_column_width=True)
