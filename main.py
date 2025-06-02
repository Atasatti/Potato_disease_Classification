import os
os.environ["PYDANTIC_DISABLE_INTERNAL_VALIDATION"] = "1"
from fastapi import FastAPI, File, UploadFile, Request
# ... rest of your code
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from keras.models import load_model
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
import base64

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this for specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

# Load the Keras model
model = load_model("model_1/_model.keras")

# Mapping condition type to disease names
CONDITION_TYPE_MAPPING = {
    0: "Potato Early_blight",
    1: "Potato Late_blight",
    2: "Potato healthy",
 
}


def preprocess_image(image_bytes):
    image = np.array(Image.open(BytesIO(image_bytes)).convert('RGB'))
    resized_image = cv2.resize(image, (256, 256))  # Changed from 224x224 to 256x256
    # normalized_image = resized_image / 255.0
    return np.expand_dims(resized_image, axis=0)  # Add batch dimension

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = preprocess_image(image_bytes)

    # Perform prediction
    predictions = model.predict(image)
    print(predictions)
    print(np.argmax(predictions, axis=1)[0])
    predicted_class = np.argmax(predictions, axis=1)[0]
    print(predictions,"\n>>>")
    print(predicted_class)
    condition = CONDITION_TYPE_MAPPING.get(predicted_class, "Unknown")

    return {"condition": condition, "confidence": float(np.max(predictions))}

