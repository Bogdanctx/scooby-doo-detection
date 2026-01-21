import io
import base64
import os
import time
import sys
import uuid

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "SDNet"))

import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from SDNet.scooby_doo_network import SDNet
from PIL import Image

from pydantic import BaseModel

class FeedbackData(BaseModel):
    image_base64: str
    correct_label: str

class RetrainMode(BaseModel):
    model: str  # 'all', 'detector', 'recognizer'

app = FastAPI()
sdnet = SDNet()
sdnet.load_models()

# Enable CORS so the frontend can communicate with the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def image_to_base64(image_obj):
    """Helper to convert PIL Image or Numpy array to Base64 string"""
    # If it is a numpy array (OpenCV format), convert to PIL first
    if isinstance(image_obj, np.ndarray):
        image_obj = Image.fromarray(image_obj)
        
    img_byte_arr = io.BytesIO()
    image_obj.save(img_byte_arr, format='PNG')

    # Convert to Base64
    raw_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

    # return the base64 string
    return raw_base64

@app.post("/api/detect")
async def detect_characters(image: UploadFile):
    # Read the uploaded image
    contents = await image.read()
    original_img = Image.open(io.BytesIO(contents)).convert("RGB")

    # Run detection
    # Expecting results = { "image_with_detections": np.array, "detected_faces": [ {"label": "Fred", "patch": np.array, "detection_score": float, "recognition_score": float}, ... ] }
    raw_results = sdnet.detect_faces(original_img)
    
    # Convert images to Base64 for JSON response
    response_data = {
        "detections": image_to_base64(raw_results["image_with_detections"]),
        "patches": []
    }

    for patch in raw_results["detected_faces"]:
        response_data["patches"].append({
            "name": patch["label"],
            "image": image_to_base64(patch["patch"]),
            "detection_score": patch["detection_score"],
            "recognition_score": patch["recognition_score"]
        })

    return response_data

@app.post("/api/retrain/full")
async def retrain_models():
    try:
        sdnet.train_detector()
        sdnet.train_recognizer()
        return {"status": "success", "message": "Full model retraining initiated."}
    except Exception as e:
        print(f"Error during full model retraining: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/api/retrain/detection")
async def retrain_detection_model():
    try:
        sdnet.train_detector()
        return {"status": "success", "message": "Detection model retraining initiated."}
    except Exception as e:
        print(f"Error during detection model retraining: {e}")
        return {"status": "error", "message": str(e)}
    
@app.post("/api/retrain/recognition")
async def retrain_recognition_model():
    try:
        sdnet.train_recognizer()
        return {"status": "success", "message": "Recognition model retraining initiated."}
    except Exception as e:
        print(f"Error during recognition model retraining: {e}")
        return {"status": "error", "message": str(e)}


@app.post("/api/feedback")
async def save_feedback(data: FeedbackData):
    try:
        if data.correct_label == "NOT_A_FACE":
            save_dir = "./SDNet/collected_negatives"
        else:
            # save mislabeled faces
            save_dir = "./SDNet/collected_positives"
            
        os.makedirs(save_dir, exist_ok=True)

        image_data = base64.b64decode(data.image_base64)
        image = Image.open(io.BytesIO(image_data))

        # use timestamp to ensure unique filenames
        filename = f"{data.correct_label.lower()}_{uuid.uuid4()}.png"
        file_path = os.path.join(save_dir, filename)

        image.save(file_path, "PNG")

        return {"status": "success", "message": f"Saved to {file_path}"}

    except Exception as e:
        # Print error to console so you can debug if it fails
        print(f"Error saving feedback: {e}")
        return {"status": "error", "message": str(e)}

app.mount("/", StaticFiles(directory=".", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)