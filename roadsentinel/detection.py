from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
YOLO_CONFIG_DIR = ROOT_DIR / ".ultralytics"
YOLO_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_CONFIG_DIR))

from ultralytics import YOLO

BIKE_CLASS_ID = 0
HELMET_CLASS_ID = 1
NO_HELMET_CLASS_ID = 2
CLASS_LABELS = {
    BIKE_CLASS_ID: "Bike",
    HELMET_CLASS_ID: "Helmeted rider",
    NO_HELMET_CLASS_ID: "No helmet",
}
CLASS_COLORS = {
    BIKE_CLASS_ID: (36, 181, 122),
    HELMET_CLASS_ID: (244, 180, 0),
    NO_HELMET_CLASS_ID: (235, 87, 87),
}


@dataclass
class Detection:
    class_id: int
    score: float
    box: tuple[int, int, int, int]


@dataclass
class BikeSummary:
    bike: Detection
    rider_count: int
    helmeted_count: int
    unhelmeted_count: int
    is_tripling: bool


@dataclass
class FrameReport:
    detections: list[Detection]
    bike_summaries: list[BikeSummary]
    tripling_count: int

    @property
    def total_bikes(self) -> int:
        return len(self.bike_summaries)

    @property
    def total_riders(self) -> int:
        return sum(item.rider_count for item in self.bike_summaries)

    @property
    def total_helmeted(self) -> int:
        return sum(item.helmeted_count for item in self.bike_summaries)

    @property
    def total_unhelmeted(self) -> int:
        return sum(item.unhelmeted_count for item in self.bike_summaries)


def _center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _point_in_box(point: tuple[float, float], box: tuple[int, int, int, int]) -> bool:
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def _distance(point: tuple[float, float], box: tuple[int, int, int, int]) -> float:
    x, y = point
    x1, y1, x2, y2 = box
    cx, cy = _center(box)
    width = max(x2 - x1, 1)
    height = max(y2 - y1, 1)
    return ((x - cx) / width) ** 2 + ((y - cy) / height) ** 2


@lru_cache(maxsize=2)
def load_model(weights_path: str) -> YOLO:
    return YOLO(weights_path)


def _results_to_detections(result, conf_threshold: float) -> list[Detection]:
    if result.boxes is None:
        return []

    boxes = result.boxes.xyxy.cpu().numpy().astype(int)
    classes = result.boxes.cls.cpu().numpy().astype(int)
    scores = result.boxes.conf.cpu().numpy()
    detections: list[Detection] = []
    for box, class_id, score in zip(boxes, classes, scores):
        if score < conf_threshold:
            continue
        detections.append(
            Detection(
                class_id=class_id,
                score=float(score),
                box=tuple(int(value) for value in box),
            )
        )
    return detections


def analyze_frame(frame: np.ndarray, weights_path: str, conf_threshold: float) -> FrameReport:
    model = load_model(weights_path)
    result = model.predict(frame, conf=conf_threshold, verbose=False)[0]
    detections = _results_to_detections(result, conf_threshold)

    bikes = [item for item in detections if item.class_id == BIKE_CLASS_ID]
    heads = [item for item in detections if item.class_id in (HELMET_CLASS_ID, NO_HELMET_CLASS_ID)]

    assignments: dict[int, list[Detection]] = {index: [] for index in range(len(bikes))}
    for head in heads:
        head_center = _center(head.box)
        containing_bikes = [
            idx for idx, bike in enumerate(bikes) if _point_in_box(head_center, bike.box)
        ]
        candidates = containing_bikes or list(range(len(bikes)))
        if not candidates:
            continue
        best_idx = min(candidates, key=lambda idx: _distance(head_center, bikes[idx].box))
        assignments[best_idx].append(head)

    bike_summaries: list[BikeSummary] = []
    for idx, bike in enumerate(bikes):
        riders = assignments[idx]
        helmeted_count = sum(1 for item in riders if item.class_id == HELMET_CLASS_ID)
        unhelmeted_count = sum(1 for item in riders if item.class_id == NO_HELMET_CLASS_ID)
        bike_summaries.append(
            BikeSummary(
                bike=bike,
                rider_count=len(riders),
                helmeted_count=helmeted_count,
                unhelmeted_count=unhelmeted_count,
                is_tripling=len(riders) > 2,
            )
        )

    tripling_count = sum(1 for item in bike_summaries if item.is_tripling)
    return FrameReport(
        detections=detections,
        bike_summaries=bike_summaries,
        tripling_count=tripling_count,
    )


def render_report(frame: np.ndarray, report: FrameReport) -> np.ndarray:
    canvas = frame.copy()
    for detection in report.detections:
        x1, y1, x2, y2 = detection.box
        color = CLASS_COLORS.get(detection.class_id, (255, 255, 255))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"{CLASS_LABELS.get(detection.class_id, detection.class_id)} {detection.score:.2f}"
        cv2.putText(
            canvas,
            label,
            (x1, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    for item in report.bike_summaries:
        x1, y1, x2, y2 = item.bike.box
        banner_color = (61, 90, 254) if item.is_tripling else (40, 40, 40)
        banner_text = (
            f"Riders: {item.rider_count} | Helmet: {item.helmeted_count} | "
            f"No helmet: {item.unhelmeted_count}"
        )
        cv2.rectangle(canvas, (x1, y2 + 4), (min(x2 + 210, canvas.shape[1] - 1), y2 + 34), banner_color, -1)
        cv2.putText(
            canvas,
            banner_text,
            (x1 + 6, y2 + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        if item.is_tripling:
            cv2.putText(
                canvas,
                "TRIPLING ALERT",
                (x1, max(22, y1 - 12)),
                cv2.FONT_HERSHEY_DUPLEX,
                0.7,
                (61, 90, 254),
                2,
                cv2.LINE_AA,
            )

    return canvas


def summarize_report(report: FrameReport) -> str:
    lines = [
        f"- Bikes detected: {report.total_bikes}",
        f"- Riders associated with bikes: {report.total_riders}",
        f"- Helmeted riders: {report.total_helmeted}",
        f"- Riders without helmets: {report.total_unhelmeted}",
        f"- Tripling alerts: {report.tripling_count}",
    ]
    if report.bike_summaries:
        lines.append("")
        lines.append("Per-bike summary:")
        for index, item in enumerate(report.bike_summaries, start=1):
            status = "Tripling flagged" if item.is_tripling else "Within limit"
            lines.append(
                f"- Bike {index}: {item.rider_count} riders, "
                f"{item.helmeted_count} helmeted, {item.unhelmeted_count} without helmets, {status}"
            )
    return "\n".join(lines)


def process_image(image_bgr: np.ndarray, weights_path: str, conf_threshold: float) -> tuple[np.ndarray, str]:
    report = analyze_frame(image_bgr, weights_path, conf_threshold)
    return render_report(image_bgr, report), summarize_report(report)


def process_video(
    video_path: str | Path,
    weights_path: str,
    conf_threshold: float,
    sample_rate: int = 1,
) -> tuple[Path, str]:
    source = cv2.VideoCapture(str(video_path))
    if not source.isOpened():
        raise ValueError("Could not open the uploaded video.")

    frame_width = int(source.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    frame_height = int(source.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    fps = source.get(cv2.CAP_PROP_FPS) or 24.0

    output_dir = Path("tmp_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"annotated_{uuid4().hex}.mp4"
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_width, frame_height),
    )

    peak_tripling = 0
    peak_riders = 0
    frames_with_alerts = 0
    processed_frames = 0
    frame_index = 0

    while True:
        ok, frame = source.read()
        if not ok:
            break

        if frame_index % max(sample_rate, 1) == 0:
            report = analyze_frame(frame, weights_path, conf_threshold)
            processed_frames += 1
            peak_tripling = max(peak_tripling, report.tripling_count)
            peak_riders = max(peak_riders, report.total_riders)
            if report.tripling_count:
                frames_with_alerts += 1
            annotated = render_report(frame, report)
        else:
            annotated = frame

        writer.write(annotated)
        frame_index += 1

    source.release()
    writer.release()

    summary = "\n".join(
        [
            f"- Frames analyzed: {processed_frames}",
            f"- Frames with tripling alerts: {frames_with_alerts}",
            f"- Peak concurrent riders on detected bikes: {peak_riders}",
            f"- Highest tripling count in a frame: {peak_tripling}",
        ]
    )
    return output_path, summary
