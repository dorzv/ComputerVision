import argparse

import torch

from app import make_gradio_interface
from models import LDM

device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)

ldm = LDM(device)


def make_image(prompt, height, width, num_inference_steps, guidance_scale):
    tensor_image = ldm(prompt, height, width, num_inference_steps, guidance_scale)
    numpy_image = LDM.convert_tensor_to_numpy(tensor_image)
    pil_image = LDM.convert_numpy_to_pil(numpy_image)
    return pil_image


app = make_gradio_interface(make_image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch Gradio App")
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a publicly shareable link for the app",
    )
    args = parser.parse_args()

    print(f"Using device: {device}")
    app.launch(share=args.share)
