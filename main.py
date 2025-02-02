from fastapi import FastAPI, File, UploadFile
import numpy as np
import cv2
from io import BytesIO
from fastapi.responses import FileResponse
import matplotlib.pyplot as plt
import tempfile

app = FastAPI()

@app.post("/edge-detection/")
async def edge_detection(file: UploadFile = File(...)):
    # Read the image file
    image_bytes = await file.read()
    np_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    # Convert to grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(image_gray, threshold1=100, threshold2=200)

    # Convert the result to an image for saving as a file
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(edges, cmap='gray')
    ax.axis('off')

    # Save the result to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    plt.savefig(temp_file, format='PNG')
    temp_file.close()  # Close the file after saving

    return FileResponse(temp_file.name, media_type='image/png', filename='edge_detection.png')
