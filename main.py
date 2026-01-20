import io
import base64
import sys
import os
import time
import numpy as np

# Add the SDNet folder to system path to allow imports of modules inside it
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "SDNet"))

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from SDNet import SDNet  # Ensure correct import based on your folder structure
from PIL import Image

from pydantic import BaseModel

class FeedbackData(BaseModel):
    image_base64: str
    correct_label: str

app = FastAPI()
sdnet = SDNet()

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
async def detect_characters(image: UploadFile = File(...)):
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

@app.post("/api/feedback")
async def save_feedback(data: FeedbackData):
    try:
        if data.correct_label == "NOT_A_FACE":
            save_dir = "./SDNet/collected_negatives"
        else:
            # save mislabeled faces
            save_dir = "./SDNet/collected_positives/"
            
        os.makedirs(save_dir, exist_ok=True)

        # decode and save
        if "," in data.image_base64:
            _, encoded = data.image_base64.split(",", 1)
        else:
            encoded = data.image_base64

        image_data = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(image_data))

        # use timestamp to ensure unique filenames
        filename = f"{data.correct_label.lower()}_{int(time.time())}.png"
        file_path = os.path.join(save_dir, filename)

        image.save(file_path, "PNG")

        return {"status": "success", "message": f"Saved to {file_path}"}

    except Exception as e:
        # Print error to console so you can debug if it fails
        print(f"Error saving feedback: {e}")
        return {"status": "error", "message": str(e)}

app.mount("/", StaticFiles(directory=".", html=True), name="static")
