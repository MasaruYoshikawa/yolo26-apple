#!/usr/bin/env python3
"""
Apple Detection & Counter - FastAPI Backend Server
Runs YOLOv8 model inference on uploaded images, performs annotations, 
and serves the web application dashboard.
"""

import os
import sys
import base64
import time
from pathlib import Path
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

# Resolve project paths
APP_DIR = Path(__file__).parent.resolve()
WORKSPACE_DIR = APP_DIR.parent.resolve()

# Add parent dir to path
sys.path.append(str(WORKSPACE_DIR))

# Create FastAPI app
app = FastAPI(title="Apple Counter API")

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Static directory
app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")

# Setup Jinja2 Templates
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))

# Model global cache
YOLO_MODEL = None
MODEL_PATH = None

def load_yolo_model():
    """Lazy loads the YOLO model from best.pt or fallback."""
    global YOLO_MODEL, MODEL_PATH
    if YOLO_MODEL is not None:
        return YOLO_MODEL
    
    from ultralytics import YOLO

    # Search paths for model weights
    search_paths = [
        APP_DIR / "weights" / "best.pt",
        WORKSPACE_DIR / "weights" / "best.pt",
        APP_DIR / "best.pt",
        WORKSPACE_DIR / "best.pt",
        APP_DIR / "yolo11n.pt",
        WORKSPACE_DIR / "yolo11n.pt"
    ]
    
    selected_path = None
    for p in search_paths:
        if p.exists():
            selected_path = p
            break
            
    if selected_path is None:
        # If no model weights exist, download yolo11n.pt as a fallback
        selected_path = WORKSPACE_DIR / "yolo11n.pt"
        print(f"⚠️ No custom weights found. Downloading fallback YOLO model to: {selected_path}")
        
    print(f"📦 Loading YOLO model from: {selected_path.resolve()}")
    YOLO_MODEL = YOLO(str(selected_path))
    MODEL_PATH = selected_path
    return YOLO_MODEL

def draw_apple_count_banner(image, apple_count, conf_threshold):
    """Draws a clean top banner displaying the total number of detected apples."""
    h, w, _ = image.shape
    banner_height = 60
    
    # Create translucent dark overlay for banner header
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_height), (20, 30, 40), -1)
    cv2.addWeighted(overlay, 0.85, image, 0.15, 0, image)

    # Accent left border bar (vibrant emerald green)
    cv2.rectangle(image, (0, 0), (8, banner_height), (0, 204, 102), -1)

    # Banner text
    title_text = f"Apple Detection | Count: {apple_count}"
    sub_text = f"Conf Thresh: {conf_threshold:.2f}"

    cv2.putText(image, title_text, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, sub_text, (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 220, 200), 1, cv2.LINE_AA)

    return image

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serves the main application dashboard."""
    model_name = "best.pt (Custom Apple Model)"
    try:
        load_yolo_model()
        model_name = MODEL_PATH.name
    except Exception as e:
        model_name = f"Error loading model: {str(e)}"
        
    return templates.TemplateResponse("index.html", {"request": request, "model_name": model_name})

@app.post("/detect")
async def detect(
    image: UploadFile = File(...),
    conf: float = Form(0.25),
    iou: float = Form(0.45),
    imgsz: int = Form(640)
):
    """API endpoint to run object detection on an uploaded image file."""
    try:
        # Read image bytes into numpy array
        contents = await image.read()
        file_bytes = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            return JSONResponse({'success': False, 'error': 'Invalid image format'}, status_code=400)
            
        # Get model instance
        model = load_yolo_model()
        
        # Run inference
        t0 = time.time()
        results = model.predict(
            source=img,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            verbose=False
        )[0]
        inference_time_ms = (time.time() - t0) * 1000
        
        boxes = results.boxes
        apple_count = len(boxes) if boxes is not None else 0
        
        # Draw annotations
        annotated_img = img.copy()
        detections = []
        if boxes is not None:
            for idx, box in enumerate(boxes, 1):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                box_conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = model.names.get(cls_id, "apple")
                
                # Append detection info
                w_box = x2 - x1
                h_box = y2 - y1
                detections.append({
                    'no': idx,
                    'class': cls_name,
                    'confidence': round(box_conf, 2),
                    'x': x1,
                    'y': y1,
                    'w': w_box,
                    'h': h_box
                })
                
                # Bounding box color (emerald green: RGB(46, 204, 113) -> BGR(113, 204, 46))
                box_color = (113, 204, 46)
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), box_color, 2)
                
                # Label text
                label = f"{cls_name} {box_conf:.2f}"
                (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                
                # Label background rectangle
                cv2.rectangle(annotated_img, (x1, y1 - text_h - 6), (x1 + text_w + 6, y1), box_color, -1)
                cv2.putText(annotated_img, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                
        # Draw the top header banner
        annotated_img = draw_apple_count_banner(annotated_img, apple_count, conf)
        
        # Encode annotated image to JPEG base64
        _, buffer = cv2.imencode('.jpg', annotated_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            'success': True,
            'apple_count': apple_count,
            'inference_time_ms': round(inference_time_ms, 1),
            'image_data': f"data:image/jpeg;base64,{img_base64}",
            'detections': detections
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({'success': False, 'error': f"Detection failed: {str(e)}"}, status_code=500)

if __name__ == '__main__':
    # Initialize the model on startup so that user doesn't wait on first request
    try:
        load_yolo_model()
    except Exception as e:
        print(f"⚠️ Warning: Could not pre-load model weights: {e}")
        
    import uvicorn
    uvicorn.run("app:app", host='0.0.0.0', port=5000, reload=True)
