from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
ALZHEIMER_MODEL_PATH = '/Users/sujaymukundtorvi/Documents/Alzheimers_Classification_using_MRI_Scans/Flask Code/alzheimer_cnn_model_best.h5'
BRAIN_TUMOR_MODEL_PATH = '/Users/sujaymukundtorvi/Documents/Alzheimers_Classification_using_MRI_Scans/Flask Code/brain_tumor_cnn_model.h5'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALZHEIMER_MODEL_PATH'] = ALZHEIMER_MODEL_PATH
app.config['BRAIN_TUMOR_MODEL_PATH'] = BRAIN_TUMOR_MODEL_PATH

alzheimer_model = load_model(app.config['ALZHEIMER_MODEL_PATH'])
brain_tumor_model = load_model(app.config['BRAIN_TUMOR_MODEL_PATH'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path, target_size=(176, 208)):
    img = load_img(image_path, target_size=target_size)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    analysis_type = request.form.get('analysis_type', 'alzheimer')  # Default to 'alzheimer' if not specified
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    
    img_array = preprocess_image(filepath)
    
    if analysis_type == 'alzheimer':
        predictions = alzheimer_model.predict(img_array)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class = ['Mild Dementia', 'Moderate Dementia', 'Non Dementia', 'Very Mild Dementia'][predicted_class_index]
    elif analysis_type == 'brain_tumor':
        predictions = brain_tumor_model.predict(img_array)
        predicted_class = 'No Tumor Detected' if predictions[0][0] > 0.5 else 'Tumor Detected'  # Assuming binary classification with a sigmoid output
    
    return jsonify({'prediction': str(predicted_class)})

if __name__ == '__main__':
    app.run(debug=True)
