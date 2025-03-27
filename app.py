from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
import io
import numpy as np
from PIL import Image
import tensorflow as tf

app = FastAPI()


model = tf.keras.models.load_model("drawing_model.h5")


class ImageData(BaseModel):
    image: str 


def preprocess_image(image_base64):
    try:
     
        image_data = base64.b64decode(image_base64.split(",")[1])
        image = Image.open(io.BytesIO(image_data)).convert("L")  
        image = image.resize((28, 28)) 
        image_array = np.array(image) / 255.0  
        image_array = image_array.reshape(1, 28, 28, 1) 
        return image_array
    except Exception as e:
        raise ValueError("Error in image processing: " + str(e))

@app.post("/predict")
async def predict(data: ImageData):
    try:
      
        image_array = preprocess_image(data.image)

   
        prediction = model.predict(image_array)
        predicted_class = np.argmax(prediction) 

        return {"prediction": int(predicted_class)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def home():
    return {"message": "ML Prediction API is running!"}
