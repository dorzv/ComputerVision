from PIL import Image
import numpy as np
import torch
from diffusers import StableDiffusionPipeline
from tqdm import tqdm


class LDM:
    def __init__(self, device, seed=42):
        torch.manual_seed(seed)
        self.device = device
        self.pipe = self.create_pipe(device)

    def create_pipe(self, device):
        return StableDiffusionPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            variant="fp16",
        ).to(device)

    def __call__(
        self, prompt, height=512, width=512, num_inference_steps=50, guidance_scale=7.5
    ):
        inputs, unconditional_inputs = self.make_tokens(prompt)
        text_embedding = self.make_embedding(inputs, unconditional_inputs)

        self.pipe.scheduler.set_timesteps(num_inference_steps)

        latents = self.make_latents(height, width)

        prediction = self.predict(latents, text_embedding, guidance_scale)

        unnormalized_image = self.decode(prediction)

        image = self.normalize_image(unnormalized_image)

        return image

    def make_tokens(self, prompt: str):
        # Tokenize the input
        inputs = self.pipe.tokenizer(
            prompt, padding="max_length", truncation=True, return_tensors="pt"
        )

        # Tokenize an empty string as the unconditional input
        unconditional_inputs = self.pipe.tokenizer(
            "", padding="max_length", truncation=True, return_tensors="pt"
        )

        return inputs, unconditional_inputs

    def make_embedding(self, inputs, unconditional_inputs):
        with torch.inference_mode():
            conditional_output = self.pipe.text_encoder(
                input_ids=inputs.input_ids.to(self.device),
                return_dict=True,
            )

            conditional_embeddings = conditional_output.last_hidden_state

            unconditional_output = self.pipe.text_encoder(
                input_ids=unconditional_inputs.input_ids.to(self.device),
                return_dict=True,
            )

            unconditional_embeddings = unconditional_output.last_hidden_state

        text_embedding = torch.cat(
            [unconditional_embeddings, conditional_embeddings], dim=0
        )

        return text_embedding

    def make_latents(self, height, width):
        latents = torch.randn(
            (1, self.pipe.unet.config.in_channels, height // 8, width // 8),
            dtype=torch.float16,
        ).to(self.device)

        latents = latents * self.pipe.scheduler.init_noise_sigma

        return latents

    def predict(self, latents, text_embedding, guidance_scale):
        for t in tqdm(
            self.pipe.scheduler.timesteps, desc="Generating image", unit="steps"
        ):
            latent_input = torch.cat([latents] * 2, dim=0)
            latent_input = self.pipe.scheduler.scale_model_input(latent_input, t)

            with torch.inference_mode():
                noise_prediction = self.pipe.unet(
                    latent_input, t, encoder_hidden_states=text_embedding
                ).sample

            noise_prediction_unconditional, noise_prediction_text = (
                noise_prediction.chunk(2, dim=0)
            )

            # Perform guidance
            noise_prediction = noise_prediction_unconditional + guidance_scale * (
                noise_prediction_text - noise_prediction_unconditional
            )

            latents = self.pipe.scheduler.step(noise_prediction, t, latents).prev_sample

        return latents

    def decode(self, prediction):
        # Use VAE to decode the latents into an image
        prediction_scaled = 1 / self.pipe.vae.config.scaling_factor * prediction

        with torch.inference_mode():
            image = self.pipe.vae.decode(prediction_scaled).sample

        return image

    def normalize_image(self, image):
        return (image / 2 + 0.5).clamp(0, 1)

    @staticmethod
    def convert_numpy_to_pil(image: np.ndarray) -> Image:
        return Image.fromarray((image * 255).astype("uint8"))

    @staticmethod
    def convert_tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
        tensor = torch.squeeze(tensor)
        image = tensor.detach().cpu().float().numpy()
        image = image.transpose(1, 2, 0)
        return image
