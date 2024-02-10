from fastapi import FastAPI, File, UploadFile
from PIL import Image
from io import BytesIO
import numpy as np
from tensorflow.keras.models import load_model
from fastapi.middleware.cors import CORSMiddleware 

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

model = load_model('cats_vs_dogs_model.h5')

def preprocess_image(img):
    img = img.resize((100, 100))
    img_array = np.array(img)
    img_array = img_array.astype('float32') / 255.0
    return np.expand_dims(img_array, axis=0)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(BytesIO(contents))
    preprocessed_img = preprocess_image(img)
    prediction = model.predict(preprocessed_img)
    return {"prediction": "dog" if prediction[0][0] > 0.5 else "cat"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
