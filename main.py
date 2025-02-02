from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from torchvision import transforms
from PIL import Image
import pickle
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the model
def load_model(model_name='efficientnet_b0'):
    with open(f'saved_models/{model_name}_full_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

# Transform for image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Image preparation
def prepare_image(image):
    image = Image.open(image).convert('RGB')
    image = transform(image)
    return image.unsqueeze(0)  # Add batch dimension

@app.route('/predict', methods=['POST'])
def predict():
    print("Received request")
    print(f"Request headers: {request.headers}")
    print(f"Request files: {request.files}")
    print(f"Request form: {request.form}")
    
    # Check if any files were sent
    if not request.files:
        print("No files in request")
        return jsonify({'error': 'No files in request'}), 400
    
    # Check specifically for 'image' file
    if 'image' not in request.files:
        print("No image file in request")
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    print("Received file:", file.filename)
    if file:
        try:
            image = prepare_image(file)
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return jsonify({'error': f'Error processing image: {str(e)}'}), 400
    else:
        print("File is empty")
        return jsonify({'error': 'File is empty'}), 400

    # Move model and image to the appropriate device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    image = image.to(device)

    # Make prediction
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)
        prediction = torch.argmax(outputs, dim=1).item()
        probability = probabilities[0][prediction].item()

    result = {
        'prediction': 'Lesion' if prediction == 1 else 'Normal',
        'confidence': f"{probability:.2%}"
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
