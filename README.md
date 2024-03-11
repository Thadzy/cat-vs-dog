# Cat vs Dog Classification Project

This project aims to build a neural network model using the Sequential API from Keras to classify images of cats and dogs. The model is then deployed as a web application using React.js for the frontend and FastAPI for the backend.

## Project Structure

- `model/`: Contains the trained neural network model (`cats_vs_dogs_model.h5`).
- `frontend/`: Contains the React.js code for the web application.
- `backend/main.py`: Contains the FastAPI code for the backend server.

## Frontend (React.js)

The frontend is built using React.js and allows users to upload an image file. Upon submission, the file is sent to the backend server for prediction, and the result (either "cat" or "dog") is displayed on the web page.

Here's the relevant code snippet:

```jsx
import React, { useState } from 'react';
import axios from 'axios';

function App() {
 const [selectedFile, setSelectedFile] = useState(null);
 const [prediction, setPrediction] = useState('');

 const handleFileChange = (event) => {
   setSelectedFile(event.target.files[0]);
 };

 const handleSubmit = async (event) => {
   event.preventDefault();
   const formData = new FormData();
   formData.append('file', selectedFile);

   try {
     const response = await axios.post('http://localhost:8000/predict/', formData, {
       headers: {
         'Content-Type': 'multipart/form-data'
       }
     });
     setPrediction(response.data.prediction);
   } catch (error) {
     console.error(error);
   }
 };

 // Render component with file input and submission
}

export default App;
```
Backend (FastAPI)
The backend is built using FastAPI and handles the image prediction using the pre-trained neural network model. It receives the image file from the frontend, preprocesses it, and passes it through the model to obtain the prediction.

Here's the relevant code snippet:

```python


Copy code
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

```
Usage
Start the backend server by running uvicorn main:app in the backend/ directory.
Start the frontend development server by running npm start in the frontend/ directory.
Open the web application in your browser (usually http://localhost:3000).
Upload an image of a cat or dog, and the application will display the prediction.
Note: Make sure to place the pre-trained model file (cats_vs_dogs_model.h5) in the model/ directory before running the application.
