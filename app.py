from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)
CORS(app)  # Enable CORS for Snack to access

# Load your TFLite model
interpreter = tf.lite.Interpreter(model_path='pcos_classifier.tflite')
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_image(image_data):
    """Preprocess image for model"""
    # Decode base64 image
    img = Image.open(io.BytesIO(base64.b64decode(image_data)))
    
    # Convert to RGB
    img = img.convert('RGB')
    
    # Resize to 256x256
    img = img.resize((256, 256))
    
    # Convert to numpy array and normalize
    img_array = np.array(img) / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    
    return img_array

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model': 'loaded'})

@app.route('/predict', methods=['POST'])
def predict():
    """PCOS prediction endpoint"""
    try:
        # Get image data from request
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Preprocess image
        image_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        input_data = preprocess_image(image_data)
        
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Process results
        healthy_prob = float(output_data[0][0])
        pcos_prob = float(output_data[0][1])
        
        is_pcos = pcos_prob > 0.5
        confidence = max(healthy_prob, pcos_prob) * 100
        
        # Calculate severity
        severity = 'none'
        if pcos_prob >= 0.85:
            severity = 'severe'
        elif pcos_prob >= 0.7:
            severity = 'high'
        elif pcos_prob >= 0.5:
            severity = 'moderate'
        elif pcos_prob >= 0.3:
            severity = 'low'
        
        # Generate recommendations
        recommendations = []
        if is_pcos:
            if pcos_prob > 0.8:
                recommendations.append({
                    'title': 'Urgent Medical Attention',
                    'description': 'High confidence PCOS detection. Consult specialist immediately',
                    'priority': 'urgent'
                })
            recommendations.extend([
                {
                    'title': 'Consult Gynecologist',
                    'description': 'Schedule appointment for professional diagnosis',
                    'priority': 'high'
                },
                {
                    'title': 'Track Symptoms',
                    'description': 'Log symptoms daily for better monitoring',
                    'priority': 'high'
                },
                {
                    'title': 'Lifestyle Modifications',
                    'description': 'Focus on balanced diet and regular exercise',
                    'priority': 'medium'
                }
            ])
        else:
            recommendations.extend([
                {
                    'title': 'Regular Monitoring',
                    'description': 'Continue regular checkups and healthy lifestyle',
                    'priority': 'low'
                },
                {
                    'title': 'Preventive Care',
                    'description': 'Maintain healthy weight and balanced diet',
                    'priority': 'medium'
                }
            ])
        
        # Return result
        result = {
            'success': True,
            'isPCOS': is_pcos,
            'confidence': round(confidence, 2),
            'probabilities': {
                'healthy': round(healthy_prob * 100, 2),
                'pcos': round(pcos_prob * 100, 2)
            },
            'label': 'PCOS Detected' if is_pcos else 'Healthy',
            'severity': severity,
            'recommendations': recommendations
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 PCOS API Server Starting...")
    print("📍 Model: pcos_classifier.tflite")
    print("✅ Server ready!")
    app.run(host='0.0.0.0', port=5000, debug=True)