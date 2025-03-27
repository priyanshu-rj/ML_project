from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import load_model
import numpy as np
import base64
from io import BytesIO
from PIL import Image

app = FastAPI()

model = load_model("drawing_model.h5")


class ImageInput(BaseModel):
    image_data: str 


def preprocess_image(base64_string):
    image_bytes = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_bytes)).convert("L")  
    image = image.resize((28, 28))  
    image_array = np.array(image) / 255.0  
    image_array = image_array.reshape(1, 28, 28, 1)  
    return image_array


@app.post("/predict")
async def predict(data: ImageInput):
    try:
        
        input_image = preprocess_image(data.image_data)
        
      
        prediction = model.predict(input_image)
        predicted_label = np.argmax(prediction)  y

        return {"prediction": int(predicted_label)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
