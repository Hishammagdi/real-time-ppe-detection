"""
Thin wrapper around the fine-tuned YOLOv8 PPE-detection model.
Loaded once at startup and reused across requests.
"""

import base64
import time
from typing import List, Optional

import cv2
import numpy as np
from ultralytics import YOLO

from app.config import settings


class Detection:
    def __init__(self, class_name: str, confidence: float, box: List[float]):
        self.class_name = class_name
        self.confidence = confidence
        self.box = box  # [x1, y1, x2, y2]

    def to_dict(self):
        return {
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "box": [round(v, 2) for v in self.box],
        }


class PPEDetector:
    def __init__(self, weights_path: str):
        print(f"Loading model from: {weights_path}")
        self.model = YOLO(weights_path)
        self.class_names = self.model.names  # {0: 'Gloves', 1: 'Hard_hat', ...}
        print(f"Model loaded. Classes: {list(self.class_names.values())}")

    def get_class_names(self) -> List[str]:
        return list(self.class_names.values())

    def predict(
        self,
        image: np.ndarray,
        conf_threshold: Optional[float] = None,
        class_filter: Optional[List[str]] = None,
    ):
        """
        Runs inference on a single BGR image (numpy array).
        Returns (annotated_image, detections, inference_ms).
        """
        conf = conf_threshold if conf_threshold is not None else settings.DEFAULT_CONF_THRESHOLD

        start = time.time()
        results = self.model.predict(image, conf=conf, verbose=False)[0]
        inference_ms = (time.time() - start) * 1000

        detections: List[Detection] = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = self.class_names[cls_id]

            if class_filter and cls_name not in class_filter:
                continue

            confidence = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            detections.append(Detection(cls_name, confidence, xyxy))

        annotated = self._draw_boxes(image.copy(), detections)
        return annotated, detections, inference_ms

    @staticmethod
    def _draw_boxes(image: np.ndarray, detections: List[Detection]) -> np.ndarray:
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det.box]
            label = f"{det.class_name} {det.confidence:.2f}"

            color = (46, 204, 113) if "hat" in det.class_name.lower() or "vest" in det.class_name.lower() else (52, 152, 219)

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (x1, y1 - text_h - 8), (x1 + text_w + 4, y1), color, -1)
            cv2.putText(image, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return image


def image_to_base64(image: np.ndarray) -> str:
    _, buffer = cv2.imencode(".jpg", image)
    return base64.b64encode(buffer).decode("utf-8")


def base64_to_image(b64_string: str) -> np.ndarray:
    img_bytes = base64.b64decode(b64_string)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


# Singleton instance, loaded once when the app starts
detector = PPEDetector(settings.MODEL_WEIGHTS_PATH)
