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

# Get model input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Extract input size dynamically
input_shape = input_details[0]['shape']
IMG_HEIGHT, IMG_WIDTH = int(input_shape[1]), int(input_shape[2])
print(f"✅ Model input size: {IMG_HEIGHT}x{IMG_WIDTH}")

def preprocess_image(image_data):
    """Decode and preprocess base64 image for model input"""
    try:
        # If image string includes base64 prefix, remove it
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        # Decode base64
        img_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(img_bytes))
        img = img.convert('RGB')
        img = img.resize((IMG_WIDTH, IMG_HEIGHT))  # auto match model size

        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # shape (1, h, w, 3)
        return img_array
    except Exception as e:
        print("❌ Image preprocessing error:", str(e))
        raise ValueError("Failed to analyze image. Invalid format or corrupt data.")


@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': '✅ PCOS Classification API running successfully!',
        'endpoints': ['/health', '/predict'],
        'model': os.path.basename(MODEL_PATH)
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model': 'loaded'})


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400

        image_data = data['image']
        input_data = preprocess_image(image_data)

        # Model inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index']).squeeze()
        print("🔍 Raw output:", output_data.tolist() if hasattr(output_data, "tolist") else output_data)

        # Handle different output shapes
        if np.isscalar(output_data):
            pcos_prob = float(output_data)
            healthy_prob = 1 - pcos_prob
        elif output_data.shape[0] == 2:
            # Some models have [healthy, pcos]
            healthy_prob = float(output_data[0])
            pcos_prob = float(output_data[1])
        else:
            raise ValueError(f"Unexpected output shape: {output_data.shape}")

        # Confidence & label
        pcos_prob = np.clip(pcos_prob, 0, 1)
        healthy_prob = np.clip(healthy_prob, 0, 1)
        is_pcos = pcos_prob > 0.5
        confidence = round(max(healthy_prob, pcos_prob) * 100, 2)

        # Determine severity
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
                {'title': 'Consult Gynecologist', 'description': 'Book an appointment with a specialist.', 'priority': 'high'},
                {'title': 'Lifestyle Modifications', 'description': 'Adopt a healthy diet and regular exercise.', 'priority': 'medium'}
            ]
        else:
            recommendations = [
                {'title': 'Regular Checkup', 'description': 'Maintain healthy lifestyle and monitor symptoms.', 'priority': 'low'}
            ]

        result = {
            'success': True,
            'isPCOS': is_pcos,
            'label': 'PCOS Detected' if is_pcos else 'Healthy',
            'confidence': confidence,
            'probabilities': {
                'pcos': round(pcos_prob * 100, 2),
                'healthy': round(healthy_prob * 100, 2)
            },
            'severity': severity,
            'recommendations': recommendations
        }

        return jsonify(result)

    except Exception as e:
        print("❌ Prediction error:", str(e))
        return jsonify({'error': 'Failed to analyze image', 'details': str(e)}), 500


if __name__ == '__main__':
    print("🚀 Starting PCOS API Server...")
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
