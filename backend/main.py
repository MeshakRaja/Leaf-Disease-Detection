from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import os
import base64
import cv2
import numpy as np
import time
from pathlib import Path

app = FastAPI()

# ✅ CORS for Vite (port 5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load model ONCE at startup
BASE_DIR = Path(__file__).resolve().parent
model = YOLO(str(BASE_DIR / "best.pt"))
UPLOAD_FOLDER = BASE_DIR / "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

CONF_THRESHOLD = 0.5
MIN_BOX_AREA = 5000

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    start_time = time.time()
    print(f"🔥 1. RECEIVED: {file.filename} ({file.size} bytes)")
    
    try:
        # ✅ 2. Read file ASYNC (non-blocking)
        contents = await file.read()
        print(f"✅ 2. FILE READ: {len(contents)} bytes ({time.time()-start_time:.1f}s)")
        
        # ✅ 3. Save to disk
        file_path = UPLOAD_FOLDER / file.filename
        with open(file_path, "wb") as buffer:
            buffer.write(contents)
        print(f"✅ 3. SAVED TO DISK: {file_path} ({time.time()-start_time:.1f}s)")
        
        # ✅ 4. YOLO prediction (non-blocking)
        print("🚀 4. YOLO PREDICTING...")
        results = model.predict(str(file_path), verbose=False, imgsz=640, conf=CONF_THRESHOLD)
        print(f"✅ 4. YOLO DONE: {time.time()-start_time:.1f}s")
        
        # ✅ 5. Process detections
        detections = []
        for r in results:
            if r.boxes is None:
                continue
                
            boxes = r.boxes.xywh.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            
            for box, cls, conf in zip(boxes, classes, scores):
                if float(conf) < CONF_THRESHOLD:
                    continue
                    
                x, y, w, h = box
                area = w * h
                if area < MIN_BOX_AREA:
                    continue
                    
                raw_name = model.names[int(cls)]
                formatted_name = raw_name.replace("_", " ").title()
                if formatted_name == "Healthy":
                    formatted_name = "Healthy Leaf"
                    
                detections.append({
                    "name": formatted_name,
                    "confidence": float(conf),
                    "bbox": [float(x), float(y), float(w), float(h)]
                })
        
        detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)
        is_leaf = len(detections) > 0
        
        # ✅ 6. Base64 image
        with open(file_path, "rb") as img_file:
            base64_str = base64.b64encode(img_file.read()).decode("utf-8")
        image_url = f"data:image/jpeg;base64,{base64_str}"
        
        print(f"🎉 TOTAL TIME: {time.time()-start_time:.2f}s | Detections: {len(detections)}")
        
        if not is_leaf:
            return {
                "detections": [],
                "imageUrl": image_url,
                "error": "No leaf detected. Please upload a clear tomato leaf image."
            }
            
        return {
            "detections": detections,
            "imageUrl": image_url,
            "isLeaf": True
        }
        
    except Exception as e:
        print(f"💥 ERROR at {time.time()-start_time:.1f}s: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # ✅ Clean up uploaded file
        if file_path.exists():
            file_path.unlink()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
