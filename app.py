import streamlit as st
from huggingface_hub import InferenceClient
from io import BytesIO
from PIL import Image
import os

# Get API key from environment variable (Recommended for security)
API_KEY = os.getenv("HF_API_KEY", "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")

# Initialize Hugging Face Inference Client
client = InferenceClient(api_key=API_KEY)

def generate_logo(name, tagline, color, theme, genre):
    """Generate a logo based on user input including name, tagline, color, theme, and genre."""
    description = f"A {theme} logo for a {genre} brand with the name '{name}'"
    if tagline:
        description += f" and tagline '{tagline}'"
    description += f" using a {color} color scheme"

    try:
        # Request image from API
        image_bytes = client.text_to_image(
            description, model="CompVis/stable-diffusion-v1-4"
        )

        # Validate response
        if not image_bytes or not isinstance(image_bytes, bytes):
            st.error("Failed to generate the image. Please try again later.")
            return None

        # Convert bytes to PIL Image
        image = Image.open(BytesIO(image_bytes))
        return image
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

# Streamlit UI
st.title("AI-Powered Logo Generator")

name = st.text_input("Enter Brand Name")
tagline = st.text_input("Enter Tagline (Optional)")
color = st.selectbox("Select Color Scheme", ["Red", "Blue", "Green", "Black", "White"])
theme = st.selectbox("Select Theme", ["Modern", "Minimalist", "Vintage", "Futuristic"])
genre = st.selectbox("Select Genre", ["Sports", "Restaurant", "Tech", "Fashion", "Education"])

if st.button("Generate Logo"):
    if name and color and theme and genre:
        image = generate_logo(name, tagline, color, theme, genre)
        if image:
            st.image(image, caption="Generated Logo", use_column_width=True)
    else:
        st.error("Please fill in all required fields")
