from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from torchvision import transforms
from PIL import Image
import pickle
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager, create_access_token, jwt_required
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
from flask_mail import Mail, Message
from datetime import datetime, timedelta
import random
import string
from pymongo.server_api import ServerApi

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes



# JWT Configuration
app.config['JWT_SECRET_KEY'] = '1234'  # Change this to a secure key in production
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = 3600  # Token expires in 1 hour
# Configure Flask-Mail
app.config['MAIL_SERVER'] = 'smtp.gmail.com'  # Use your SMTP provider
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'smita.app7@gmail.com'  # Set your email
app.config['MAIL_PASSWORD'] = 'isjakmaxqdpqkqkl'  # Set your password

mail = Mail(app)

bcrypt = Bcrypt(app)
jwt = JWTManager(app)

# Connect to MongoDB
client = MongoClient('mongodb+srv://mupparapukoushik:Shadow_slave@cluster0.wl4w8.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0')
db = client['auth_db']
users = db['users']

# Ensure unique email index
users.create_index("email", unique=True)  # Prevent duplicate email 


# Generate OTP function
def generate_otp():
    return ''.join(random.choices(string.digits, k=6))  # 6-digit OTP



@app.route('/verify-otp', methods=['POST'])
def verify_otp():
    data = request.json
    if not data or "email" not in data or "otp" not in data:
        return jsonify({"message": "Missing fields"}), 400

    user = users.find_one({"email": data['email']})

    if not user:
        return jsonify({"message": "User not found"}), 404

    # Check OTP and expiry
    if user['otp'] != data['otp']:
        return jsonify({"message": "Invalid OTP"}), 400

    if datetime.utcnow() > user['otp_expiry']:
        return jsonify({"message": "OTP expired"}), 400

    # Update user as verified
    users.update_one({"email": data['email']}, {"$set": {"verified": True}, "$unset": {"otp": "", "otp_expiry": ""}})
    
    return jsonify({"message": "OTP verified successfully!"}), 200


# Signup Route with OTP sending
@app.route('/signup', methods=['POST'])
def signup():
    data = request.json
    if not data or "username" not in data or "email" not in data or "password" not in data:
        return jsonify({"message": "Missing fields"}), 400

    hashed_pw = bcrypt.generate_password_hash(data['password']).decode('utf-8')

    otp = generate_otp()
    otp_expiry = datetime.utcnow() + timedelta(minutes=10)  # OTP valid for 10 min

    try:
        users.insert_one({
            "username": data['username'],
            "email": data['email'],
            "password": hashed_pw,
            "otp": otp,
            "otp_expiry": otp_expiry,
            "verified": False  # User needs OTP verification
        })

        # Send OTP Email
        msg = Message('Your OTP Code', sender='smita.app7@gmail.com', recipients=[data['email']])
        msg.body = f'Your OTP is {otp}. It will expire in 10 minutes.'
        mail.send(msg)

        return jsonify({"message": "OTP sent to your email!"}), 201
    except DuplicateKeyError:
        return jsonify({"message": "Email already exists"}), 400


@app.route('/login', methods=['POST'])
def login():
    data = request.json
    if not data or "email" not in data or "password" not in data:
        return jsonify({"message": "Missing fields"}), 400

    user = users.find_one({"email": data['email']})
    if not user:
        return jsonify({"message": "Invalid credentials"}), 401

    if not user.get('verified', False):
        return jsonify({"message": "Please verify your email first"}), 403

    if bcrypt.check_password_hash(user['password'], data['password']):
        token = create_access_token(identity=data['email'])
        return jsonify({"token": token}), 200

    return jsonify({"message": "Invalid credentials"}), 401

def load_model(model_name='efficientnet_b0'):
    model_path = f'saved_models/{model_name}_full_model.pkl'
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)  # Load the model

    model.to(torch.device('cpu'))  # Move model to CPU
    model.eval()  # Set model to evaluation mode
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

# if __name__ == '__main__':
    # app.run(debug=False, host="0.0.0.0")
    
