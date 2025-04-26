from flask import Flask, request, jsonify
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import requests

webhook_url = 'https://discord.com/api/webhooks/1365730256811724830/0tE6KGPL2M0KRAOtOFsNrX7JUvW-Q7HLBt6YQxA907tK2JRtKKsDlLDD5EwO9p5ILAod'
data = {
    "content": "O servidor está online"
}
UPLOAD_FOLDER = './uploads'
MODEL_PATH = 'model.tflite'  # Changed model name to "model"

# Initialize TFLite interpreter
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Class labels (ensure this order matches your training dataset)
class_indices = {
    0: 'Museu Ferroviário',
    1: 'Ponte Medieval Do Rio Marnel',
    # Add more classes as needed
}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def home():
    return "Hello, World!"

@app.route('/data', methods=['POST'])
def get_data():
    if 'file' in request.files:
        file = request.files['file']
    elif 'image' in request.files:
        file = request.files['image']
    else:
        return jsonify({"error": "No file part in the request"}), 400

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if not file.filename.lower().endswith('.png'):
        return jsonify({"error": "Only PNG files are allowed"}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Load and preprocess image
    img = image.load_img(file_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], img_array)

    # Run inference
    interpreter.invoke()

    # Get output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_index = np.argmax(output_data)
    predicted_label = class_indices.get(predicted_index, "Unknown")

    return jsonify({
        "message": "File successfully uploaded",
        "prediction": predicted_label,
        "confidence": float(np.max(output_data)),
        "filename": file.filename
    }), 200

if __name__ == '__main__':
    if os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
        try:
            requests.post(webhook_url, json=data)
        except:
            pass
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
