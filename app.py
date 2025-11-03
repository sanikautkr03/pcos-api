from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import os
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS

# Initialize Flask app
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

# --- Helper function to preprocess image ---
def preprocess_image(image_data):
    """Preprocess base64-encoded image for TFLite model."""
    try:
        img = Image.open(io.BytesIO(base64.b64decode(image_data)))
        img = img.convert('RGB')
        img = img.resize((256, 256))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        raise ValueError(f"Image preprocessing failed: {e}")

# --- Routes ---

@app.route('/', methods=['GET'])
def home():
    """Root endpoint to verify deployment"""
    return jsonify({
        'message': '✅ PCOS Classification API is running successfully!',
        'endpoints': ['/health', '/predict'],
        'model': MODEL_PATH
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

        # Extract base64 content (handle both raw and data URI formats)
        image_data = data['image'].split(',')[-1]
        input_data = preprocess_image(image_data)

        # Check input tensor shape and type
        expected_shape = tuple(input_details[0]['shape'])
        if input_data.shape != expected_shape:
            input_data = np.resize(input_data, expected_shape)

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # Assuming output shape = [1, 2] → [healthy_prob, pcos_prob]
        healthy_prob = float(output_data[0][0])
        pcos_prob = float(output_data[0][1])
        is_pcos = pcos_prob > 0.5
        confidence = round(max(healthy_prob, pcos_prob) * 100, 2)

        # Severity levels
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
                    'description': 'High confidence PCOS detection. Consult specialist immediately.',
                    'priority': 'urgent'
                })
            recommendations.extend([
                {
                    'title': 'Consult Gynecologist',
                    'description': 'Schedule an appointment for professional diagnosis.',
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
    port = int(os.environ.get('PORT', 10000))  # Use Render's dynamic port
    app.run(host='0.0.0.0', port=port)
