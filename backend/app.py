from flask import Flask, request, jsonify
from main_ocr import *
from PIL import Image
import os
from flask_cors import CORS
import cv2
import pdf2image


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def is_pdf(file):
    """Helper function to check if the file is a PDF."""
    return file.filename.lower().endswith('.pdf')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file and file.filename != '':
        filepath = os.path.join('uploads', file.filename)
        file.save(filepath)
        
        # Perform OCR using pytesseract
        if is_pdf(file):
            # Convert PDF to images
            images = pdf2image.convert_from_path(filepath)
            
            # Initialize an empty string to hold the OCR results
            ocr_text = ""
            
            # Loop through the images, perform OCR, and concatenate results
            for i, image in enumerate(images):
                # Convert PIL image to OpenCV format
                open_cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                
                # Perform OCR
                ocr_text += f"\nPage {i+1}:\n"  # Add page number
                ocr_text += perform_ocr(open_cv_image)  # Perform OCR on the image
            
            # Remove the PDF file after processing
            os.remove(filepath)
            
            return jsonify({'ocr_text': ocr_text}), 200
        
        else:
            # Handle image files directly
            image = cv2.imread(filepath)
            ocr_text = perform_ocr(image)
            
            # Remove the image file after processing
            os.remove(filepath)
            
            return jsonify({'ocr_text': ocr_text}), 200
    
    return jsonify({'error': 'File not found or invalid'}), 400

if __name__ == '__main__':
    app.run(debug=True)
