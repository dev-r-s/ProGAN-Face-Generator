# app.py
import gradio as gr
import torch
import numpy as np
from inference import ProgressiveGANInference
from PIL import Image


def tensor_to_pil(img_tensor):
    img = (img_tensor + 1) / 2
    img = img.clamp(0, 1)
    img = img.permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((img * 255).astype(np.uint8))


class App:
    def __init__(self, checkpoint):
        self.infer = ProgressiveGANInference(checkpoint)

    def generate(self, num_images, seed, truncation):
        images = self.infer.generate(
            num_images=num_images,
            seed=None if seed < 0 else int(seed),
            truncation=truncation
        )
        return [tensor_to_pil(img) for img in images]

    def interpolate(self, seed1, seed2, frames, truncation):
        imgs = self.infer.interpolate(
            int(seed1), int(seed2), frames, truncation
        )
        return [tensor_to_pil(img) for img in imgs]


app = App("checkpoint_res512_step650000.pt")


with gr.Blocks(title="Progressive GAN") as demo:
    gr.Markdown("## 🎨 Progressive GAN Face Generator (512×512)")

    with gr.Tab("Random Generation"):
        n = gr.Slider(1, 9, value=4, step=1, label="Images")
        seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")
        trunc = gr.Slider(0.3, 1.2, value=0.8, step=0.05, label="Truncation")
        btn = gr.Button("Generate")
        gallery = gr.Gallery(columns=3, height="auto")

        btn.click(app.generate, [n, seed, trunc], gallery)

    with gr.Tab("Interpolation"):
        s1 = gr.Number(value=42, precision=0, label="Start Seed")
        s2 = gr.Number(value=123, precision=0, label="End Seed")
        f = gr.Slider(3, 20, value=8, step=1, label="Frames")
        t = gr.Slider(0.3, 1.2, value=0.7, step=0.05, label="Truncation")
        btn2 = gr.Button("Interpolate")
        gallery2 = gr.Gallery(columns=4, height="auto")

        btn2.click(app.interpolate, [s1, s2, f, t], gallery2)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
