from __future__ import annotations

from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from roadsentinel.detection import process_image, process_video

ROOT_DIR = Path(__file__).resolve().parent.parent
WEIGHTS_PATH = ROOT_DIR / "artifacts" / "helmet_tripling_detector.pt"
EXAMPLE_DIR = ROOT_DIR / "assets" / "examples"

APP_THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="zinc",
)


def _load_image(path: str) -> np.ndarray:
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Unable to load example image: {path}")
    return image


def run_image_demo(image: np.ndarray, conf_threshold: float) -> tuple[np.ndarray, str]:
    if image is None:
        raise gr.Error("Upload an image or pick one of the examples.")
    annotated_bgr, summary = process_image(image[:, :, ::-1], str(WEIGHTS_PATH), conf_threshold)
    return annotated_bgr[:, :, ::-1], summary


def run_video_demo(video_path: str, conf_threshold: float) -> tuple[str, str]:
    if not video_path:
        raise gr.Error("Upload a video clip to run roadway analysis.")
    output_path, summary = process_video(video_path, str(WEIGHTS_PATH), conf_threshold)
    return str(output_path), summary


def build_demo() -> gr.Blocks:
    examples = [[str(path), 0.10] for path in sorted(EXAMPLE_DIR.glob("*.jpg"))]

    with gr.Blocks(title="Triple Helmet Detection") as demo:
        gr.Markdown(
            """
            # Triple Helmet Detection
            Upload a traffic frame or short clip to spot bikes, estimate rider counts, flag missing helmets,
            and raise a tripling alert when more than two riders are associated with one bike.
            """
        )

        with gr.Tab("Image"):
            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(
                        type="numpy",
                        label="Traffic image",
                        sources=["upload", "clipboard"],
                    )
                    image_conf = gr.Slider(
                        minimum=0.05,
                        maximum=0.9,
                        value=0.10,
                        step=0.05,
                        label="Confidence threshold",
                    )
                    image_button = gr.Button("Analyze image", variant="primary")
                with gr.Column(scale=1):
                    image_output = gr.Image(type="numpy", label="Annotated result")
                    image_summary = gr.Markdown(label="Summary")

            gr.Examples(
                examples=examples,
                inputs=[image_input, image_conf],
                outputs=[image_output, image_summary],
                fn=run_image_demo,
                cache_examples=False,
            )
            image_button.click(
                fn=run_image_demo,
                inputs=[image_input, image_conf],
                outputs=[image_output, image_summary],
            )

        with gr.Tab("Video"):
            with gr.Row():
                with gr.Column(scale=1):
                    video_input = gr.Video(label="Traffic video")
                    video_conf = gr.Slider(
                        minimum=0.05,
                        maximum=0.9,
                        value=0.10,
                        step=0.05,
                        label="Confidence threshold",
                    )
                    video_button = gr.Button("Analyze video", variant="primary")
                with gr.Column(scale=1):
                    video_output = gr.Video(label="Processed clip")
                    video_summary = gr.Markdown(label="Clip summary")

            video_button.click(
                fn=run_video_demo,
                inputs=[video_input, video_conf],
                outputs=[video_output, video_summary],
            )

        gr.Markdown(
            """
            **Dataset note:** the detector was trained from the public Roboflow project shared by the user.
            This Space packages only the inference app and a few local example frames derived from the source demo.
            """
        )

    return demo


def get_launch_kwargs() -> dict:
    return {
        "theme": APP_THEME,
        "ssr_mode": False,
    }
