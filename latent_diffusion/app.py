import gradio as gr

prompt_input = gr.Textbox(label="Prompt", lines=2)

height_input = gr.Slider(
    minimum=256,
    maximum=1024,
    step=64,
    value=512,
    label="Height (pixels)",
    info="Image height in pixels"
)

width_input = gr.Slider(
    minimum=256,
    maximum=1024,
    step=64,
    value=512,
    label="Width (pixels)",
    info="Image width in pixels"
)

cfg_input = gr.Slider(
    minimum=1.0,
    maximum=20.0,
    step=0.5,
    value=7.5,
    label="CFG Scale",
    info="Classifier-Free Guidance scale"
)

steps_input = gr.Slider(
    minimum=1,
    maximum=150,
    step=1,
    value=50,
    label="Inference steps",
    info="Number of denoising steps"
)

output_image = gr.Image(label="Output Image", type="pil", format="png")


def make_gradio_interface(function: callable):
    return gr.Interface(
        fn=function,
        inputs=[prompt_input, height_input, width_input, steps_input, cfg_input],
        outputs=[output_image],
        title="Image Generation Interface",
        description="Generate images based on the given prompt and parameters."
    )
