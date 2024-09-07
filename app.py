import torch
from flask import Flask, request, jsonify
from PIL import Image
import io
from ultralytics import YOLO
import pandas as pd

app = Flask(__name__)

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')  # Update the path to your model

@app.route('/test', methods=['GET'])
def test():
    return jsonify({"message": "Server's Running"})

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image file from the POST request
    image_file = request.files['image']
 
    # Load the image using PIL
    image = Image.open(io.BytesIO(image_file.read()))

    # Perform prediction
    results = model(image)  # Pass the image directly to the YOLO model

    # Extract predictions with class names
    predictions = []
    for result in results[0].boxes:
        # Extract coordinates, class label, and confidence
        x_min, y_min, x_max, y_max = result.xyxy.tolist()[0]
        class_id = int(result.cls)  # Get the class ID
        confidence = float(result.conf)  # Get the confidence score
        class_name = model.names[class_id]  # Map class ID to class name
        
        # Append to predictions list
        predictions.append({
            'x_min': x_min,
            'y_min': y_min,
            'x_max': x_max,
            'y_max': y_max,
            'class_name': class_name,
            'confidence': confidence
        })

    return jsonify({'predictions': predictions})

if __name__ == '__main__':
    app.run(debug=True)