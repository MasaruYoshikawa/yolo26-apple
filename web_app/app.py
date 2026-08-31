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

# Model global cache (fruit_type -> (model_instance, model_path))
MODEL_CACHE = {}

def get_fruit_color(fruit_type):
    """Returns distinct BGR colors for different fruit types."""
    ft = str(fruit_type).lower()
    if "orange" in ft:
        return (30, 144, 255)    # Vibrant Amber / Orange (BGR)
    elif "blueberry" in ft or "berry" in ft:
        return (225, 105, 65)    # Royal Indigo / Violet (BGR)
    else:
        return (113, 204, 46)    # Emerald Green (BGR)


def load_fruit_model(fruit_type: str = "apple"):
    """Lazy loads and caches the YOLO model for specified fruit type."""
    fruit_key = fruit_type.lower()
    if fruit_key in MODEL_CACHE:
        return MODEL_CACHE[fruit_key]
    
    from ultralytics import YOLO

    # Search paths for model weights based on requested fruit
    if fruit_key == "orange":
        search_paths = [
            APP_DIR / "weights" / "orange_best.pt",
            WORKSPACE_DIR / "weights" / "orange_best.pt",
            WORKSPACE_DIR / "weights" / "best.pt",
            APP_DIR / "yolo11n.pt",
            WORKSPACE_DIR / "yolo11n.pt"
        ]
    elif fruit_key == "blueberry":
        search_paths = [
            APP_DIR / "weights" / "blueberry_best.pt",
            WORKSPACE_DIR / "weights" / "blueberry_best.pt",
            WORKSPACE_DIR / "weights" / "best.pt",
            APP_DIR / "yolo11n.pt",
            WORKSPACE_DIR / "yolo11n.pt"
        ]
    else: # apple or default
        search_paths = [
            APP_DIR / "weights" / "apple_best.pt",
            WORKSPACE_DIR / "weights" / "apple_best.pt",
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
        selected_path = WORKSPACE_DIR / "yolo11n.pt"
        print(f"⚠️ No custom weights found for '{fruit_key}'. Downloading fallback YOLO model to: {selected_path}")
        
    print(f"📦 Loading YOLO model for '{fruit_key}' from: {selected_path.resolve()}")
    model_instance = YOLO(str(selected_path))
    MODEL_CACHE[fruit_key] = (model_instance, selected_path)
    return MODEL_CACHE[fruit_key]


def draw_fruit_count_banner(image, count, conf_threshold, fruit_name="Fruit"):
    """Draws a clean top banner displaying the total number of detected fruits."""
    h, w, _ = image.shape
    banner_height = 60
    
    # Create translucent dark overlay for banner header
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_height), (20, 30, 40), -1)
    cv2.addWeighted(overlay, 0.85, image, 0.15, 0, image)

    # Accent left border bar
    accent_color = get_fruit_color(fruit_name)
    cv2.rectangle(image, (0, 0), (8, banner_height), accent_color, -1)

    # Banner text
    display_name = fruit_name.capitalize()
    title_text = f"{display_name} Detection | Count: {count}"
    sub_text = f"Conf Thresh: {conf_threshold:.2f}"

    cv2.putText(image, title_text, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, sub_text, (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 220, 200), 1, cv2.LINE_AA)

    return image


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serves the main application dashboard."""
    model_name = "Multi-Fruit AI Models"
    try:
        load_fruit_model("apple")
    except Exception as e:
        model_name = f"Error loading model: {str(e)}"
        
    return templates.TemplateResponse(request=request, name="index.html", context={"model_name": model_name})


@app.post("/detect")
async def detect(
    image: UploadFile = File(...),
    fruit_type: str = Form("apple"),
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
            
        # Get model instance for requested fruit
        model, model_path = load_fruit_model(fruit_type)
        
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
        fruit_count = len(boxes) if boxes is not None else 0
        box_color = get_fruit_color(fruit_type)
        
        # Draw annotations
        annotated_img = img.copy()
        detections = []
        if boxes is not None:
            for idx, box in enumerate(boxes, 1):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                box_conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = model.names.get(cls_id, fruit_type)
                
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
                
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), box_color, 2)
                
                # Label text
                label = f"{cls_name} {box_conf:.2f}"
                (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                
                # Label background rectangle
                cv2.rectangle(annotated_img, (x1, y1 - text_h - 6), (x1 + text_w + 6, y1), box_color, -1)
                cv2.putText(annotated_img, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                
        # Draw the top header banner
        annotated_img = draw_fruit_count_banner(annotated_img, fruit_count, conf, fruit_name=fruit_type)
        
        # Encode annotated image to JPEG base64
        _, buffer = cv2.imencode('.jpg', annotated_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            'success': True,
            'fruit_type': fruit_type,
            'fruit_count': fruit_count,
            'apple_count': fruit_count,  # Backwards compatibility
            'inference_time_ms': round(inference_time_ms, 1),
            'image_data': f"data:image/jpeg;base64,{img_base64}",
            'detections': detections
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({'success': False, 'error': f"Detection failed: {str(e)}"}, status_code=500)


if __name__ == '__main__':
    # Initialize the model on startup
    try:
        load_fruit_model("apple")
    except Exception as e:
        print(f"⚠️ Warning: Could not pre-load model weights: {e}")
        
    import uvicorn
    uvicorn.run("app:app", host='0.0.0.0', port=5000, reload=True)

