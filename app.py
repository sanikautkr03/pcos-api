from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
import base64
import os
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# Load TFLite model once
MODEL_PATH = 'pcos_classifier.tflite'
print("🚀 Loading model from:", MODEL_PATH)

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Determine required input size from model dynamically
input_shape = input_details[0]['shape']
IMG_HEIGHT, IMG_WIDTH = input_shape[1], input_shape[2]
print(f"✅ Model input size: {IMG_HEIGHT}x{IMG_WIDTH}")

def preprocess_image(image_data):
    """Preprocess image for model inference"""
    img = Image.open(io.BytesIO(base64.b64decode(image_data)))
    img = img.convert('RGB')
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))  # ✅ Automatically match model input size
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    return img_array


@app.route('/', methods=['GET'])
def home():
    """Root endpoint to verify deployment"""
    return jsonify({
        'message': '✅ PCOS Classification API is running successfully!',
        'endpoints': ['/health', '/predict'],
        'model': os.path.basename(MODEL_PATH)
    }), 200


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model': 'loaded'}), 200


@app.route('/predict', methods=['POST'])
def predict():
    """PCOS prediction endpoint"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400

        # Extract base64 content
        image_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        input_data = preprocess_image(image_data)

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index']).squeeze()

        print("🔍 Raw model output:", output_data.tolist())

        # Handle both sigmoid and softmax cases
        if output_data.ndim == 0 or np.isscalar(output_data):
            # Sigmoid output → single probability for PCOS
            pcos_prob = float(output_data)
            healthy_prob = 1 - pcos_prob
        elif output_data.shape[0] == 2:
            # Softmax output → ensure correct order
            # Swap indices if predictions are flipped
            pcos_prob = float(output_data[1])
            healthy_prob = float(output_data[0])
        else:
            raise ValueError(f"Unexpected output shape: {output_data.shape}")

        # Ensure probabilities are within valid range
        pcos_prob = np.clip(pcos_prob, 0.0, 1.0)
        healthy_prob = np.clip(healthy_prob, 0.0, 1.0)

        # Determine result
        is_pcos = pcos_prob > 0.5
        confidence = round(max(healthy_prob, pcos_prob) * 100, 2)

        # Severity logic
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
            if pcos_prob > 0.8:
                recommendations.append({
                    'title': 'Urgent Medical Attention',
                    'description': 'High confidence PCOS detection. Consult a specialist immediately.',
                    'priority': 'urgent'
                })
            recommendations.extend([
                {
                    'title': 'Consult Gynecologist',
                    'description': 'Schedule appointment for professional diagnosis.',
                    'priority': 'high'
                },
                {
                    'title': 'Track Symptoms',
                    'description': 'Log symptoms daily for better monitoring.',
                    'priority': 'medium'
                },
                {
                    'title': 'Lifestyle Modifications',
                    'description': 'Focus on balanced diet and regular exercise.',
                    'priority': 'medium'
                }
            ])
        else:
            recommendations.extend([
                {
                    'title': 'Regular Monitoring',
                    'description': 'Continue regular checkups and maintain a healthy lifestyle.',
                    'priority': 'low'
                },
                {
                    'title': 'Preventive Care',
                    'description': 'Maintain healthy weight and balanced diet.',
                    'priority': 'medium'
                }
            ])

        result = {
            'success': True,
            'isPCOS': is_pcos,
            'confidence': confidence,
            'probabilities': {
                'healthy': round(healthy_prob * 100, 2),
                'pcos': round(pcos_prob * 100, 2)
            },
            'label': 'PCOS Detected' if is_pcos else 'Healthy',
            'severity': severity,
            'recommendations': recommendations
        }

        return jsonify(result), 200

    except Exception as e:
        print("❌ Prediction error:", str(e))
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("🚀 Starting PCOS API Server...")
    port = int(os.environ.get('PORT', 10000))  # Render’s dynamic port
    app.run(host='0.0.0.0', port=port)
