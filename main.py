from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from torchvision import transforms
from PIL import Image
import pickle
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
from flask_mail import Mail, Message
from datetime import datetime, timedelta
import random
import string
from pymongo.server_api import ServerApi
import cloudinary
import cloudinary.uploader
import cloudinary.api
# from bson import ObjectId

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
# client = MongoClient('mongodb://localhost:27017/')
client = MongoClient('mongodb+srv://mupparapukoushik:Shadow_slave@cluster0.wl4w8.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0')
db = client['auth_db']
users = db['users']
feedbacks = db['feedbacks']

# Cloudinary Configuration
cloudinary.config(
    cloud_name="dea33xdja",
    api_key="965866172536318",
    api_secret="Zx2O5Q9xLexjty6FO6a358wOuEU"
)


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
# Predefined admin emails
admin_emails = ["admin1@gmail.com", "admin2@gmail.com", "admin3@gmail.com"]

@app.route('/admin/users', methods=['GET'])
@jwt_required()
def get_users():
    current_email = get_jwt_identity()
    # Check if current user is admin
    if current_email not in admin_emails:
        return jsonify({"message": "Unauthorized"}), 403
    
    # Retrieve all users (exclude password field)
    all_users = list(users.find({}, {"password": 0}))
    # Convert ObjectId to string for JSON serialization
    for user in all_users:
         user['_id'] = str(user['_id'])
    return jsonify(all_users), 200


@app.route('/signup', methods=['POST'])
def signup():
    data = request.json
    if not data or "username" not in data or "email" not in data or "password" not in data:
        return jsonify({"message": "Missing fields"}), 400

    hashed_pw = bcrypt.generate_password_hash(data['password']).decode('utf-8')
    otp = generate_otp()
    otp_expiry = datetime.utcnow() + timedelta(minutes=10)  # OTP valid for 10 min

    existing_user = users.find_one({"email": data['email']})

    if existing_user:
        if existing_user.get("verified"):  # User already verified
            return jsonify({"message": "Email already exists"}), 400
        else:
            # Resend OTP
            users.update_one({"email": data['email']}, {"$set": {"otp": otp, "otp_expiry": otp_expiry}})
            msg = Message('Your OTP Code', sender='smita.app7@gmail.com', recipients=[data['email']])
            msg.body = f'Your OTP is {otp}. It will expire in 10 minutes.'
            mail.send(msg)
            return jsonify({"message": "OTP resent to your email!"}), 200

    # Insert new user
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

# Login Route
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

# Add these imports at the top of your file
from bson import ObjectId

# Forgot Password Initiation Route
@app.route('/forgot-password', methods=['POST'])
def forgot_password():
    data = request.json
    if not data or "email" not in data:
        return jsonify({"message": "Email is required"}), 400

    user = users.find_one({"email": data["email"]})
    if not user:
        return jsonify({"message": "User with this email does not exist"}), 404

    # Generate a reset token (a random 32-character string)
    reset_token = ''.join(random.choices(string.ascii_letters + string.digits, k=32))
    reset_expiry = datetime.utcnow() + timedelta(minutes=30)  # Token valid for 30 minutes

    # Save the reset token and expiry to the user's document
    users.update_one(
        {"email": data["email"]}, 
        {"$set": {
            "reset_token": reset_token, 
            "reset_expiry": reset_expiry
        }}
    )

    return jsonify({
        "message": "Reset token generated successfully!", 
        "reset_token": reset_token
    }), 200

# Password Reset Route
@app.route('/reset-password', methods=['POST'])
def reset_password():
    data = request.json
    if not data or "email" not in data or "reset_token" not in data or "new_password" not in data or "confirm_password" not in data:
        return jsonify({"message": "Missing required fields"}), 400

    # Validate new password
    if data["new_password"] != data["confirm_password"]:
        return jsonify({"message": "Passwords do not match"}), 400

    # Find user by email and reset token
    user = users.find_one({
        "email": data["email"], 
        "reset_token": data["reset_token"]
    })

    if not user:
        return jsonify({"message": "Invalid reset token"}), 400

    # Check if reset token is expired
    if datetime.utcnow() > user.get('reset_expiry'):
        return jsonify({"message": "Reset token has expired"}), 400

    # Hash the new password
    hashed_new_password = bcrypt.generate_password_hash(data['new_password']).decode('utf-8')

    # Update user's password and remove reset token
    users.update_one(
        {"email": data["email"]}, 
        {"$set": {
            "password": hashed_new_password
        }, 
        "$unset": {
            "reset_token": "", 
            "reset_expiry": ""
        }}
    )

    return jsonify({"message": "Password reset successfully!"}), 200

def load_model(model_name='efficientnet_b0'):
    model_path = f'saved_models/{model_name}_full_model.pkl'
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    model.to(torch.device('cpu'))
    model.eval()
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
    return image.unsqueeze(0)

@app.route('/form_submit', methods=['POST'])
@jwt_required()
def form_submit():
    try:
        current_user_email = get_jwt_identity()

        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        image_file = request.files['image']

        # Upload image to Cloudinary
        upload_result = cloudinary.uploader.upload(image_file)
        image_url = upload_result.get("secure_url")  # Get Cloudinary image URL

        # Process image with model
        image = prepare_image(image_file)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        image = image.to(device)

        with torch.no_grad():
            outputs = model(image)
            probabilities = torch.softmax(outputs, dim=1)
            prediction = torch.argmax(outputs, dim=1).item()
            confidence = f"{probabilities[0][prediction].item():.2%}"

        # Store form data in MongoDB (including Cloudinary URL)
        form_data = {
            'date': request.form.get('date'),
            'smitaId': request.form.get('smitaId'),
            'firstName': request.form.get('firstName'),
            'lastName': request.form.get('lastName'),
            'prefix': request.form.get('prefix'),
            'age': request.form.get('age'),
            'sex': request.form.get('sex'),
            'religion': request.form.get('religion'),
            'maritalStatus': request.form.get('maritalStatus'),
            'education': request.form.get('education'),
            'occupation': request.form.get('occupation'),
            'income': request.form.get('income'),
            'phoneNumber': request.form.get('phoneNumber'),
            'address': request.form.get('address'),
            'type': request.form.get('type'),
            'prediction': 'Lesion' if prediction == 1 else 'Normal',
            'confidence': confidence,
            'image_url': image_url,  # Store Cloudinary URL
            'user_email': current_user_email,
            'timestamp': datetime.utcnow()
        }

        users.update_one({"email": current_user_email}, {"$push": {"form_submissions": form_data}})

        return jsonify({
            'prediction': form_data['prediction'],
            'confidence': confidence,
            'image_url': image_url,  # Return Cloudinary URL
            'message': 'Form submitted and stored successfully'
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    print("Received request")
    print(f"Request headers: {request.headers}")
    print(f"Request files: {request.files}")
    print(f"Request form: {request.form}")
    
    if not request.files:
        print("No files in request")
        return jsonify({'error': 'No files in request'}), 400
    
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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    image = image.to(device)

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

@app.route('/add-comment', methods=['POST'])
@jwt_required()
def add_comment():
    # Extract data from form (instead of request.json)
    print("Received Form Data:", request.form)

    comment_text = request.form.get("comment")
    smita_id = request.form.get("smitaId")  # Required for database update

    if not comment_text or not smita_id:
        return jsonify({'message': 'Missing required fields'}), 400

    current_email = get_jwt_identity()

    user = users.find_one({'email': current_email})
    print(f"Current User: {current_email}, User Found: {user}")  # Debugging print
    if not user:
        return jsonify({'message': 'User not found'}), 404

    username = user.get('username', 'Anonymous')
    comment = {
        'username': username,
        'comment': comment_text,
        'timestamp': datetime.utcnow()
    }
    print("Received comment:", comment)
    
    # Check if the submission exists before updating
    existing_submission = users.find_one(
    {
        'form_submissions.smitaId': smita_id
    },
        {'form_submissions.$': 1}  # Project only the matching submission
    )

    print("Matching Submission:", existing_submission)


    result = users.update_one(
        {
            'form_submissions.smitaId': smita_id,
        },
        {
            '$push': {
                'form_submissions.$.comments': comment
            }
        }
    )
    
    print("Modified Count:", result.modified_count)  # Debugging print


    if result.modified_count == 0:
        return jsonify({'message': 'Submission not found or comment not added'}), 400

    return jsonify({'message': 'Comment added successfully', 'comment': comment}), 200
    
    
@app.route('/get-comments/<smita_id>', methods=['GET'])
# @jwt_required()
def get_comments(smita_id):
    # current_email = get_jwt_identity()

    # user = users.find_one({'email': current_email})
    # if not user:
    #     return jsonify({'message': 'User not found'}), 404

    # Find the submission matching the given smita_id
    
    submission = users.find_one(
        {'form_submissions.smitaId': smita_id},
        {'form_submissions.$': 1}  # Project only the matching submission
    )

    if not submission or 'form_submissions' not in submission:
        return jsonify({'message': 'Submission not found'}), 404

    comments = submission['form_submissions'][0].get('comments', [])

    return jsonify({'comments': comments}), 200

@app.route('/edit-comment/<smita_id>', methods=['PUT'])
# @jwt_required()
def edit_comment(smita_id):
    data = request.get_json()
    comment_index = data.get("comment_index")  # Index of the comment to be edited
    new_comment_text = data.get("new_comment")

    if comment_index is None or new_comment_text is None:
        return jsonify({'message': 'Missing required fields'}), 400

    # Find the submission matching the given smita_id
    submission = users.find_one(
        {'form_submissions.smitaId': smita_id},
        {'form_submissions.$': 1}
    )

    if not submission or 'form_submissions' not in submission:
        return jsonify({'message': 'Submission not found'}), 404

    form_submission = submission['form_submissions'][0]
    comments = form_submission.get('comments', [])

    # Ensure the comment index is within range
    if comment_index < 0 or comment_index >= len(comments):
        return jsonify({'message': 'Invalid comment index'}), 400

    # Update the comment text
    comments[comment_index]['comment'] = new_comment_text
    comments[comment_index]['edited_timestamp'] = datetime.utcnow()

    # Update the comment in the database
    result = users.update_one(
        {'form_submissions.smitaId': smita_id},
        {'$set': {'form_submissions.$.comments': comments}}
    )

    if result.modified_count == 0:
        return jsonify({'message': 'Comment not updated'}), 400

    return jsonify({'message': 'Comment updated successfully', 'updated_comment': comments[comment_index]}), 200

@app.route('/check_feedback', methods=['GET'])
@jwt_required()
def check_feedback():
    user_id = get_jwt_identity()
    existing = feedbacks.find_one({"userId": user_id})
    return jsonify({"submitted": bool(existing)})


@app.route('/submit_feedback', methods=['POST'])
@jwt_required()
def submit_feedback():
    user_id = get_jwt_identity()
    data = request.get_json()
    answers = data.get('answers', [])

    if not answers or len(answers) != 7:
        return jsonify({"message": "All questions must be answered."}), 400

    feedbacks.insert_one({"userId": user_id, "answers": answers})
    return jsonify({"message": "Feedback submitted successfully"}), 200

@app.route('/admin/feedbacks', methods=['GET'])
@jwt_required()
def get_all_feedbacks():
    # Get user email from JWT
    current_email = get_jwt_identity()
    
    # Check if the user is an admin
    if current_email not in admin_emails:
        return jsonify({'message': 'Unauthorized access'}), 403
    
    # Check if a specific userId was requested
    specific_user_id = request.args.get('userId')
    
    # Prepare the query
    query = {}
    if specific_user_id:
        query = {"userId": specific_user_id}
    
    # Fetch feedback submissions
    all_feedbacks = list(feedbacks.find(query))
    
    # Add username information to each feedback
    for feedback in all_feedbacks:
        if '_id' in feedback:
            feedback['_id'] = str(feedback['_id'])
        
        # Look up the username for this feedback's userId
        if 'userId' in feedback:
            user = users.find_one({"email": feedback['userId']})
            if user and 'username' in user:
                feedback['username'] = user['username']
    
    return jsonify(all_feedbacks)

# if __name__ == '__main__':
#     app.run(debug=True, host="0.0.0.0", port="8080")
