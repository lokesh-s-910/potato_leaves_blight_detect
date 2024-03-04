from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image 
import tensorflow as tf 

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = tf.keras.models.load_model("../saved_models/1")
class_names = ["Early Blight","Late Blight", "Healthy"]


@app.get("/ping")
async def ping():
    return "hello"

def read_file_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

## upload the app
@app.post("/predict")
async def predict(
    file : UploadFile = File(...)
): 
    image = read_file_image(await file.read())   ## read files
    img_batch = np.expand_dims(image, axis=0)
    prediction = model.predict(img_batch)
    pred_clas = class_names[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])
    return {
        "class": pred_clas,
        "confidence": float(confidence)
    }

if __name__=="__main__":
    uvicorn.run(app, host="localhost", port=8000)