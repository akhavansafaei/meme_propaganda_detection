"""
Flask Web Application for Meme Propaganda Detection
میم رو آپلود کن و تکنیک‌های پروپاگاندا رو ببین!
"""

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import sys
import json
from PIL import Image
import base64

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from three_stream_architecture import ThreeStreamArchitecture
from model_loader import SimplifiedPropagandaDetector, load_ground_truth_labels

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

# Load AdaBoost model
print("🔧 Loading AdaBoost model...")
# اگه مدل ذخیره‌شده دارید، مسیرش رو اینجا بذارید
model_path = '../saved_models/adaboost_model.pth' if os.path.exists('../saved_models/adaboost_model.pth') else None
propaganda_detector = SimplifiedPropagandaDetector(model_path=model_path)

if propaganda_detector.use_real_model:
    print("✅ مدل واقعی بارگذاری شد!")
else:
    print("⚠️  از شبیه‌سازی استفاده می‌شود (مدل واقعی رو وصل کنید)")

print("=" * 80)


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
    # اگه OCR می‌خواید، اینجا اضافه کنید
    return ""


def analyze_meme(image_path, ground_truth_labels=None):
    """
    Analyze meme and return detected propaganda techniques.

    This combines:
    1. Your model predictions (از model_loader)
    2. Knowledge Graph reasoning (بهبود پیش‌بینی‌ها)
    3. Ground truth labels (اگه داشته باشید)
    """

    # 1. Extract text from image (OCR)
    text = extract_text_from_image(image_path)

    # 2. Get predictions from your model
    model_result = propaganda_detector.predict(
        image_path=image_path,
        text=text,
        ground_truth_labels=ground_truth_labels
    )

    neural_predictions = model_result['predictions']

    # 3. Apply Knowledge Graph reasoning
    kg_result = kg_detector.predict(
        text=text,
        image_features={},  # Can add image features later
        neural_predictions=neural_predictions,
        return_explanations=False
    )

    # 4. Prepare results for display
    results = {
        'detected_techniques': kg_result['detected_techniques'],
        'all_predictions': {},
        'ground_truth': ground_truth_labels,
        'using_real_model': model_result['using_real_model']
    }

    # Add scores for all techniques (sorted by confidence)
    for tech in kg_detector.technique_names:
        neural_score = neural_predictions.get(tech, 0.0)
        kg_score = kg_result['stream_predictions']['knowledge_graph'].get(tech, 0.0)
        final_score = kg_result['final_predictions'].get(tech, 0.0)

        results['all_predictions'][tech] = {
            'neural_score': float(neural_score),
            'kg_score': float(kg_score),
            'final_score': float(final_score),
            'detected': tech in kg_result['detected_techniques'],
            'is_ground_truth': (tech in ground_truth_labels) if ground_truth_labels else None
        }

    # Sort by final score
    results['all_predictions'] = dict(
        sorted(results['all_predictions'].items(),
               key=lambda x: x[1]['final_score'],
               reverse=True)
    )

    # Calculate metrics if ground truth available
    if ground_truth_labels:
        results['metrics'] = calculate_metrics(
            predicted=kg_result['detected_techniques'],
            ground_truth=ground_truth_labels
        )

    return results


def calculate_metrics(predicted, ground_truth):
    """
    Calculate precision, recall, F1 for this sample
    """
    predicted_set = set(predicted)
    truth_set = set(ground_truth)

    true_positives = len(predicted_set & truth_set)
    false_positives = len(predicted_set - truth_set)
    false_negatives = len(truth_set - predicted_set)

    precision = true_positives / len(predicted_set) if len(predicted_set) > 0 else 0
    recall = true_positives / len(truth_set) if len(truth_set) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'precision': round(precision, 3),
        'recall': round(recall, 3),
        'f1': round(f1, 3)
    }


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

        # Check if ground truth labels provided
        ground_truth = None
        if 'ground_truth' in request.form:
            # Parse ground truth from form data
            try:
                ground_truth_str = request.form['ground_truth']
                ground_truth = json.loads(ground_truth_str) if ground_truth_str else None
            except:
                ground_truth = None

        # Analyze the meme
        results = analyze_meme(filepath, ground_truth_labels=ground_truth)

        # Convert image to base64 for display
        with open(filepath, 'rb') as img_file:
            img_data = base64.b64encode(img_file.read()).decode('utf-8')

        results['image_data'] = f"data:image/jpeg;base64,{img_data}"
        results['filename'] = filename

        # Clean up - delete uploaded file
        os.remove(filepath)

        return jsonify(results)

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"❌ Error: {error_details}")
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


@app.route('/add_ground_truth', methods=['POST'])
def add_ground_truth():
    """
    Add ground truth labels for a specific image to dataset

    این endpoint برای ذخیره لیبل‌های واقعی استفاده میشه
    """
    data = request.json
    image_name = data.get('image_name')
    labels = data.get('labels', [])

    if not image_name:
        return jsonify({'error': 'نام تصویر لازم است'}), 400

    # Load existing labels or create new
    labels_file = '../data/labels.json'
    os.makedirs('../data', exist_ok=True)

    if os.path.exists(labels_file):
        with open(labels_file, 'r', encoding='utf-8') as f:
            all_labels = json.load(f)
    else:
        all_labels = {}

    # Add/update labels
    all_labels[image_name] = labels

    # Save
    with open(labels_file, 'w', encoding='utf-8') as f:
        json.dump(all_labels, f, indent=2, ensure_ascii=False)

    return jsonify({'success': True, 'message': 'لیبل‌ها ذخیره شدند'})


if __name__ == '__main__':
    print("=" * 80)
    print("🚀 Starting Meme Propaganda Detection Web App")
    print("=" * 80)

    if not propaganda_detector.use_real_model:
        print("\n⚠️  توجه: از شبیه‌سازی استفاده می‌شود")
        print("   برای استفاده از مدل واقعی:")
        print("   1. مدل رو ذخیره کن در: ../saved_models/adaboost_model.pth")
        print("   2. model_loader.py رو ویرایش کن")
        print("   3. سرور رو دوباره راه‌اندازی کن")
        print()

    print("🌐 Open your browser and go to: http://localhost:5000")
    print("=" * 80)

    app.run(debug=True, host='0.0.0.0', port=5000)
