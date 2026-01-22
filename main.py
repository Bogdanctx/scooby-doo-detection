import os
import sys

from torch import mode

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "SDNet"))

import asyncio
import numpy as np
import io
import base64
import uuid
from starlette.concurrency import run_in_threadpool
from fastapi import FastAPI, UploadFile, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from SDNet.scooby_doo_network import SDNet
from PIL import Image
from Logger import Logger
from pydantic import BaseModel

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

    return JSONResponse(status_code=status.HTTP_200_OK,
                        content=response_data)

@app.post("/api/retrain/{model}")
async def retrain_models(model: str):
    try:
        if model == "both":
            await run_in_threadpool(sdnet.train)
        elif model == "detector":
            await run_in_threadpool(sdnet.train_detector)
        elif model == "recognizer":
            await run_in_threadpool(sdnet.train_recognizer)

        return JSONResponse(status_code=status.HTTP_200_OK,
                            content={"status": "success", "message": "Full model retraining initiated."})
    except Exception as e:
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            content={"status": "error", "message": str(e)})


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

        return JSONResponse(status_code=status.HTTP_200_OK,
                            content={"status": "success", "message": f"Saved to {file_path}"})

    except Exception as e:
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            content={"status": "error", "message": str(e)})
    
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        # Open in read mode
        with open(Logger.LOG_FILE, "r") as f:
            # REMOVED: f.seek(0, os.SEEK_END) 
            # We want to read from the start so the user sees history
            
            while True:
                line = f.readline()
                
                if line:
                    await websocket.send_text(line.strip())
                else:
                    # Check if file was truncated (cleared)
                    if os.stat(Logger.LOG_FILE).st_size < f.tell():
                        f.seek(0)
                    else:
                        # Wait for new logs to be written
                        await asyncio.sleep(0.1)
                        
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")


app.mount("/", StaticFiles(directory=".", html=True), name="static")

if __name__ == "__main__":
    Logger.clear_log()

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)