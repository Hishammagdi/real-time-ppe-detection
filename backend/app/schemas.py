from typing import List, Optional
from pydantic import BaseModel


class DetectionSchema(BaseModel):
    class_name: str
    confidence: float
    box: List[float]


class ImagePredictionResponse(BaseModel):
    detections: List[DetectionSchema]
    annotated_image_base64: str
    inference_ms: float


class FramePredictionResponse(BaseModel):
    detections: List[DetectionSchema]
    annotated_frame_base64: str
    inference_ms: float


class FrameRequest(BaseModel):
    image_base64: str
    conf_threshold: Optional[float] = None
    class_filter: Optional[List[str]] = None


class ClassesResponse(BaseModel):
    classes: List[str]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    classes: List[str]
