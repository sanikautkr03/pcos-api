import tensorflow as tf
import numpy as np
from PIL import Image
import io, base64, os
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

MODEL_PATH = 'pcos_classifier.tflite'
print("🚀 Loading model from:", MODEL_PATH)

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 🧠 Print shapes at startup for debugging
print(f"Input shape: {input_details[0]['shape']}")
print(f"Output shape: {output_details[0]['shape']}")

def preprocess_image(image_data):
    """Preprocess any-size base64 image for model input."""
    try:
        img = Image.open(io.BytesIO(base64.b64decode(image_data)))
        img = img.convert('RGB')

        # dynamically get required model input size (e.g., 224x224)
        target_h = input_details[0]['shape'][1]
        target_w = input_details[0]['shape'][2]
        img = img.resize((target_w, target_h))

        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        raise ValueError(f"Image preprocessing failed: {e}")

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model': MODEL_PATH}), 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400

        image_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        input_data = preprocess_image(image_data)

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index']).squeeze()

        # 🧩 Handle both model types (sigmoid or softmax)
        if output_data.ndim == 0:  # single value (sigmoid)
            pcos_prob = float(output_data)
            healthy_prob = 1 - pcos_prob
        elif output_data.shape[0] == 1:  # 1 output (sigmoid)
            pcos_prob = float(output_data[0])
            healthy_prob = 1 - pcos_prob
        else:  # 2 outputs (softmax)
            healthy_prob = float(output_data[0])
            pcos_prob = float(output_data[1])

        # Prediction results
        is_pcos = pcos_prob > 0.5
        confidence = round(max(healthy_prob, pcos_prob) * 100, 2)

        severity = (
            'severe' if pcos_prob >= 0.85 else
            'high' if pcos_prob >= 0.7 else
            'moderate' if pcos_prob >= 0.5 else
            'low' if pcos_prob >= 0.3 else 'none'
        )

        recommendations = []
        if is_pcos:
            if pcos_prob > 0.8:
                recommendations.append({
                    'title': 'Urgent Medical Attention',
                    'description': 'High confidence PCOS detection. Consult specialist immediately.',
                    'priority': 'urgent'
                })
            recommendations.extend([
                {'title': 'Consult Gynecologist', 'description': 'Schedule appointment for professional diagnosis.', 'priority': 'high'},
                {'title': 'Track Symptoms', 'description': 'Log symptoms daily for better monitoring.', 'priority': 'medium'},
                {'title': 'Lifestyle Modifications', 'description': 'Focus on balanced diet and regular exercise.', 'priority': 'medium'}
            ])
        else:
            recommendations.extend([
                {'title': 'Regular Monitoring', 'description': 'Continue regular checkups and maintain a healthy lifestyle.', 'priority': 'low'},
                {'title': 'Preventive Care', 'description': 'Maintain healthy weight and balanced diet.', 'priority': 'medium'}
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
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("🚀 Starting PCOS API Server...")
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
