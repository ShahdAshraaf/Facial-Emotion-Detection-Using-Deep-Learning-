import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io
from flask import Flask, render_template, redirect, url_for, request, jsonify

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Check for required files
MODEL_PATH = 'facial_recognition_model.h5'
CASCADE_PATH = 'haarcascade_frontalface_default.xml'

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Required model file not found: {MODEL_PATH}")
if not os.path.exists(CASCADE_PATH):
    raise FileNotFoundError(f"Required cascade file not found: {CASCADE_PATH}")

# Load the model and cascade classifier
# Modify the model loading line
model = load_model(MODEL_PATH, compile=False)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# Emotion labels
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def process_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
    
    if len(faces) == 0:
        return None
    
    # Process the first face found
    x, y, w, h = faces[0]
    face_roi = gray[y:y+h, x:x+w]
    
    # Resize to model input size
    face_roi = cv2.resize(face_roi, (48, 48))
    
    # Normalize
    face_roi = face_roi / 255.0
    
    # Reshape for model
    face_roi = np.reshape(face_roi, (1, 48, 48, 1))
    
    # Get predictions
    predictions = model.predict(face_roi)[0]
    
    # Create emotion probability dictionary
    emotions = {emotion: float(prob) for emotion, prob in zip(EMOTIONS, predictions)}
    
    return emotions

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('upload.html', error='No file uploaded')
        
        file = request.files['image']
        if file.filename == '':
            return render_template('upload.html', error='No file selected')
        
        # Read and process the image
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        emotions = process_image(image)
        if emotions is None:
            return render_template('upload.html', error='No face detected in the image')
        
        # Save the image
        filename = 'upload_' + str(np.random.randint(10000)) + '.jpg'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        cv2.imwrite(filepath, image)
        
        return render_template('upload.html', result={'image': filename, 'emotions': emotions})
    
    return render_template('upload.html')

@app.route('/camera')
def camera():
    return render_template('camera.html')

@app.route('/camera/capture', methods=['POST'])
def camera_capture():
    if 'image' not in request.files:
        return jsonify({'error': 'No image data'}), 400
    
    file = request.files['image']
    image_bytes = file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    emotions = process_image(image)
    if emotions is None:
        return jsonify({'error': 'No face detected'}), 400
    
    return jsonify({'emotions': emotions})

@app.route('/stream')
def stream():
    return render_template('stream.html')

@app.route('/stream/analyze', methods=['POST'])
def stream_analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'No image data'}), 400
    
    file = request.files['image']
    image_bytes = file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    emotions = process_image(image)
    if emotions is None:
        return jsonify({'error': 'No face detected'}), 400
    
    return jsonify({'emotions': emotions})

if __name__ == "__main__":
    app.run(debug=True)