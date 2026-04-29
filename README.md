---
title: Triple Helmet Detection
emoji: 🚦
colorFrom: blue
colorTo: green
sdk: gradio
python_version: "3.10"
app_file: app.py
fullWidth: true
license: mit
---

# Triple Helmet Detection

Triple Helmet Detection is a Gradio app for checking motorcycle safety in street imagery. The app:

- detects bikes and rider head regions,
- estimates how many riders are associated with each bike,
- separates helmeted and unhelmeted riders,
- raises a tripling alert when a bike is carrying more than two riders.

## Owner

- **GitHub owner:** [Abs6187](https://github.com/Abs6187)
- **Hugging Face owner:** [Abs6187](https://huggingface.co/Abs6187)

## Links

- **Hugging Face Space:** [Abs6187/roadsentinel-helmet-monitor](https://huggingface.co/spaces/Abs6187/roadsentinel-helmet-monitor)
- **GitHub repository:** [Abs6187/Triple-Helmet-Detection](https://github.com/Abs6187/Triple-Helmet-Detection)

## Demo Images

### Example 1

![Example 1](assets/examples/sample_01.jpg)

### Example 2

![Example 2](assets/examples/sample_02.jpg)

### Example 3

![Example 3](assets/examples/sample_03.jpg)

## Included in this Space

- a clean Hugging Face Spaces deployment maintained under the `Abs6187` account,
- a compact YOLO weight file used for inference,
- bundled demo images for quick testing,
- a fresh app structure focused on deployment and demo use.

## Dataset reference

Training data reference: [Roboflow helmet-detection-8bftf](https://app.roboflow.com/hsherpa/helmet-detection-8bftf/1)

The Space itself ships inference assets only. It does not mirror the dataset.

## Local run

```bash
pip install -r requirements.txt
python app.py
```
