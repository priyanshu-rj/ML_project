from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import base64
from io import BytesIO
from PIL import Image
import numpy as np

app = FastAPI()

# ✅ Fix CORS issue (allows frontend to communicate)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(data: dict):
    try:
       
        image_data = data.get("image")
        if not image_data:
            raise HTTPException(status_code=400, detail="No image provided")

       
        image_data = image_data.split(",")[1]  
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes)).convert("L") 

      
        img_array = np.array(image)
        print("Received Image Shape:", img_array.shape)  
      
        prediction = "cat"  

        return {"prediction": prediction}

    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail="Prediction failed")
