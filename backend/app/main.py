import os
import tempfile
import time
from typing import List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.config import settings
from app.inference import detector, image_to_base64, base64_to_image
from app.schemas import (
    ImagePredictionResponse,
    FramePredictionResponse,
    FrameRequest,
    ClassesResponse,
    HealthResponse,
    DetectionSchema,
)

app = FastAPI(
    title="PPE Detection API",
    description="Real-time PPE (helmet/vest/gloves/mask/boots) object detection backend, powered by a fine-tuned YOLOv8 model.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="ok",
        model_loaded=detector.model is not None,
        classes=detector.get_class_names(),
    )


@app.get("/classes", response_model=ClassesResponse)
def get_classes():
    return ClassesResponse(classes=detector.get_class_names())


@app.post("/predict/image", response_model=ImagePredictionResponse)
async def predict_image(
    file: UploadFile = File(...),
    conf_threshold: Optional[float] = Query(None, ge=0.0, le=1.0),
    class_filter: Optional[str] = Query(
        None, description="Comma-separated class names, e.g. 'Hard_hat,Vest'"
    ),
):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode the uploaded image.")

    classes = class_filter.split(",") if class_filter else None
    annotated, detections, inference_ms = detector.predict(image, conf_threshold, classes)

    return ImagePredictionResponse(
        detections=[DetectionSchema(**d.to_dict()) for d in detections],
        annotated_image_base64=image_to_base64(annotated),
        inference_ms=round(inference_ms, 2),
    )


@app.post("/predict/frame", response_model=FramePredictionResponse)
def predict_frame(request: FrameRequest):
    """
    Used by the frontend's live-webcam loop: one base64-encoded frame in,
    one annotated base64-encoded frame out. Called repeatedly, ~5-15 fps
    depending on network/model speed.
    """
    image = base64_to_image(request.image_base64)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode the frame.")

    annotated, detections, inference_ms = detector.predict(
        image, request.conf_threshold, request.class_filter
    )

    return FramePredictionResponse(
        detections=[DetectionSchema(**d.to_dict()) for d in detections],
        annotated_frame_base64=image_to_base64(annotated),
        inference_ms=round(inference_ms, 2),
    )


@app.post("/predict/video")
async def predict_video(
    file: UploadFile = File(...),
    conf_threshold: Optional[float] = Query(None, ge=0.0, le=1.0),
    class_filter: Optional[str] = Query(None),
):
    """
    Processes an uploaded video frame-by-frame and returns the fully
    annotated video as a downloadable file.
    """
    classes = class_filter.split(",") if class_filter else None

    suffix = os.path.splitext(file.filename)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_in:
        tmp_in.write(await file.read())
        input_path = tmp_in.name

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Could not open the uploaded video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = input_path.replace(suffix, f"_annotated{suffix}")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        annotated, _, _ = detector.predict(frame, conf_threshold, classes)
        writer.write(annotated)
        frame_count += 1

    cap.release()
    writer.release()
    os.remove(input_path)

    total_time = time.time() - start_time
    print(f"Processed {frame_count} frames in {total_time:.1f}s")

    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename=f"annotated_{file.filename}",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=True)
