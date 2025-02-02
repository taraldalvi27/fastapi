from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
from io import BytesIO
from PIL import Image

app = FastAPI()

@app.post("/detect_edges/")
async def detect_edges(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        image = np.array(image)

        # Convert to grayscale if image is not already grayscale
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
        blurred_image = cv2.GaussianBlur(image, (5, 5), 0) # Apply Gaussian blur
        edges = cv2.Canny(blurred_image, 100, 200)

        # Convert back to PIL Image for proper response
        edges_image = Image.fromarray(edges)

        # Save image to bytes buffer
        buffered = BytesIO()
        edges_image.save(buffered, format="JPEG")  # Or any other suitable format

        # Return image as bytes
        return {"image": buffered.getvalue()}

    except Exception as e:
        return {"error": str(e)}
