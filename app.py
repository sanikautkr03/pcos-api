import numpy as np
from PIL import Image
import io
import base64
import os
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# === Load TFLite model ===
MODEL_PATH = 'pcos_classifier.tflite'
print("🚀 Loading model from:", MODEL_PATH)

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
IMG_HEIGHT, IMG_WIDTH = int(input_shape[1]), int(input_shape[2])
print(f"✅ Model expects {IMG_HEIGHT}x{IMG_WIDTH}")

def preprocess_image(image_data):
    """Decode and preprocess base64 image for model"""
    try:
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        img_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img = img.resize((IMG_WIDTH, IMG_HEIGHT))
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print("❌ Preprocessing failed:", e)
        raise ValueError("Failed to analyze image")

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': '✅ PCOS Classification API is running',
        'endpoints': ['/health', '/predict'],
        'model': MODEL_PATH
    }), 200

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model': 'loaded'}), 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400

        image_data = data['image']
        input_data = preprocess_image(image_data)

        # Inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index']).squeeze()

        print("🔍 Raw model output:", output_data)

        # Normalize output
        if np.isscalar(output_data):
            pcos_prob = float(output_data)
            healthy_prob = 1.0 - pcos_prob
        elif len(output_data) == 2:
            healthy_prob = float(output_data[0])
            pcos_prob = float(output_data[1])
        else:
            raise ValueError(f"Unexpected output shape: {output_data.shape}")

        pcos_prob = np.clip(pcos_prob, 0, 1)
        healthy_prob = np.clip(healthy_prob, 0, 1)
        is_pcos = bool(pcos_prob > 0.5)
        confidence = round(float(max(healthy_prob, pcos_prob)) * 100, 2)

        # Severity level
        if pcos_prob >= 0.85:
            severity = 'severe'
        elif pcos_prob >= 0.7:
            severity = 'high'
        elif pcos_prob >= 0.5:
            severity = 'moderate'
        elif pcos_prob >= 0.3:
            severity = 'low'
        else:
            severity = 'none'

        # Recommendations
        recommendations = []
        if is_pcos:
            recommendations = [
                {"title": "Consult Gynecologist", "description": "Book a professional consultation.", "priority": "high"},
                {"title": "Lifestyle Modifications", "description": "Eat healthy, exercise regularly.", "priority": "medium"},
            ]
        else:
            recommendations = [
                {"title": "Regular Monitoring", "description": "Keep up a balanced routine.", "priority": "low"}
            ]

        result = {
            'success': True,
            'isPCOS': bool(is_pcos),
            'label': 'PCOS Detected' if is_pcos else 'Healthy',
            'confidence': confidence,
            'probabilities': {
                'pcos': round(float(pcos_prob) * 100, 2),
                'healthy': round(float(healthy_prob) * 100, 2)
            },
            'severity': severity,
            'recommendations': recommendations
        }

        return jsonify(result), 200

    except Exception as e:
        print("❌ Prediction error:", str(e))
        return jsonify({'error': 'Failed to analyze image', 'details': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting PCOS API Server...")
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
