"""
Flask Web Application for Meme Propaganda Detection
میم رو آپلود کن و تکنیک‌های پروپاگاندا رو ببین!
"""

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import sys
import torch
import json
from PIL import Image
import io
import base64

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from three_stream_architecture import ThreeStreamArchitecture

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Create upload folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize Knowledge Graph detector
print("🔧 Loading Knowledge Graph detector...")
kg_detector = ThreeStreamArchitecture(
    techniques_path="../propaganda_techniques.json",
    use_llm=False
)
print("✅ Knowledge Graph loaded!")

# TODO: Load your AdaBoost model here
# model = load_your_adaboost_model('path/to/model.pth')
model = None  # Placeholder


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text_from_image(image_path):
    """
    Extract text from image using OCR.
    TODO: Add proper OCR (pytesseract, EasyOCR, etc.)
    """
    # Placeholder - return empty for now
    return ""


def get_model_predictions(image_path, text):
    """
    Get predictions from your AdaBoost ensemble model.

    TODO: Replace this with actual model loading and prediction
    from your notebook: ali_write_adaboost_ensemble.ipynb
    """

    if model is None:
        # Simulated predictions for demonstration
        # Replace this with your actual model
        import random

        predictions = {}
        for tech in kg_detector.technique_names:
            # Random predictions for demo
            predictions[tech] = random.uniform(0.0, 1.0)

        return predictions

    # TODO: Your actual model prediction code here
    # Example:
    # image = Image.open(image_path)
    # features = extract_features(image, text)
    # predictions = model.predict(features)
    # return predictions


def analyze_meme(image_path):
    """
    Analyze meme and return detected propaganda techniques.

    This combines:
    1. Your AdaBoost ensemble model (neural predictions)
    2. Knowledge Graph reasoning (improved predictions)
    """

    # 1. Extract text from image (OCR)
    text = extract_text_from_image(image_path)

    # 2. Get predictions from your model
    neural_predictions = get_model_predictions(image_path, text)

    # 3. Apply Knowledge Graph reasoning
    result = kg_detector.predict(
        text=text,
        image_features={},  # Can add image features later
        neural_predictions=neural_predictions,
        return_explanations=True
    )

    # 4. Prepare results for display
    results = {
        'detected_techniques': result['detected_techniques'],
        'all_predictions': {},
        'explanations': result.get('explanations', {})
    }

    # Add scores for all techniques (sorted by confidence)
    for tech in kg_detector.technique_names:
        results['all_predictions'][tech] = {
            'neural_score': result['stream_predictions']['neural'].get(tech, 0.0),
            'kg_score': result['stream_predictions']['knowledge_graph'].get(tech, 0.0),
            'final_score': result['final_predictions'].get(tech, 0.0),
            'detected': tech in result['detected_techniques']
        }

    # Sort by final score
    results['all_predictions'] = dict(
        sorted(results['all_predictions'].items(),
               key=lambda x: x[1]['final_score'],
               reverse=True)
    )

    return results


@app.route('/')
def index():
    """Main page - upload interface."""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and analysis."""

    if 'file' not in request.files:
        return jsonify({'error': 'فایلی انتخاب نشده است'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'فایلی انتخاب نشده است'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'فرمت فایل باید jpg, jpeg, png یا gif باشد'}), 400

    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Analyze the meme
        results = analyze_meme(filepath)

        # Convert image to base64 for display
        with open(filepath, 'rb') as img_file:
            img_data = base64.b64encode(img_file.read()).decode('utf-8')

        results['image_data'] = f"data:image/jpeg;base64,{img_data}"

        # Clean up - delete uploaded file
        os.remove(filepath)

        return jsonify(results)

    except Exception as e:
        return jsonify({'error': f'خطا در تحلیل: {str(e)}'}), 500


@app.route('/techniques')
def get_techniques():
    """Get list of all propaganda techniques with descriptions."""
    with open('../propaganda_techniques.json', 'r', encoding='utf-8') as f:
        config = json.load(f)

    techniques_info = []
    for tech in config['propaganda_techniques']:
        techniques_info.append({
            'name': tech['name'],
            'category': tech.get('category', 'Other'),
            'description': tech.get('prerequisite', ''),
        })

    return jsonify(techniques_info)


if __name__ == '__main__':
    print("=" * 80)
    print("🚀 Starting Meme Propaganda Detection Web App")
    print("=" * 80)
    print("\n📝 TODO: Load your AdaBoost model")
    print("   Edit app.py and add your model loading code\n")
    print("🌐 Open your browser and go to: http://localhost:5000")
    print("=" * 80)

    app.run(debug=True, host='0.0.0.0', port=5000)
