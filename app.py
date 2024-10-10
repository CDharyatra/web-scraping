from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import pytesseract
from PIL import Image
import cv2
import numpy as np

# Set the path to tesseract executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

preprocessing_methods = {
    'none': lambda image: image,  # No preprocessing
    'basic': lambda image: Image.fromarray(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)),
    'threshold': lambda image: Image.fromarray(cv2.threshold(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
    'adaptive_threshold': lambda image: Image.fromarray(cv2.adaptiveThreshold(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)),
    'denoise': lambda image: Image.fromarray(cv2.fastNlMeansDenoising(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)))
}

def extract_text_from_image(image_path, preprocess_method='none'):
    try:
        # Check if the file exists
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"The file {image_path} does not exist.")

        # Open the image
        with Image.open(image_path) as image:
            # Preprocess the image based on the method provided
            preprocess_func = preprocessing_methods.get(preprocess_method, lambda image: image)
            processed_image = preprocess_func(image)
            
            # Perform OCR on the processed image
            custom_config = r'--oem 3 --psm 6'
            text = pytesseract.image_to_string(processed_image, config=custom_config)

        return text

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except pytesseract.TesseractNotFoundError:
        print("Error: Tesseract is not installed or the path is incorrect.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

@app.route('/')
def home():
    return render_template('index.html')  # Ensure your HTML page is named 'index.html'

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Run the OCR process
    text = extract_text_from_image(filepath)

    return jsonify({'text': text})  # Return the extracted text as JSON

if __name__ == "__main__":
    app.run(debug=True)
