import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

class ImageGenerator:
    def __init__(self, token: str, model_id: str):
        self.token = token
        self.model_id = model_id
        self.pipe = None

    def load_model(self):
        """Load the model with CPU configuration."""
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            use_auth_token=self.token,
            torch_dtype=torch.float32,  # Using float32 for CPU
        )
        return self.pipe

    def generate(self, prompt: str, num_steps: int = 20) -> Image.Image:
        """Generate image from prompt."""
        if self.pipe is None:
            self.load_model()
        
        with torch.no_grad():
            image = self.pipe(
                prompt,
                num_inference_steps=num_steps,
                guidance_scale=7.5,
            ).images[0]
        
        return image
