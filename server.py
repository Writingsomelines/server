from flask import Flask, request, jsonify
import os
import numpy as np
from tflite_runtime.interpreter import Interpreter
from tensorflow.keras.preprocessing import image

UPLOAD_FOLDER = './uploads'
MODEL_PATH = 'model.tflite'  # Ensure your model is in the same folder

# Load the TFLite model with tflite-runtime
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Map class index to label
class_indices = {
    0: 'Museu Ferroviário',
    1: 'Ponte Medieval Do Rio Marnel',
    # Add more if needed
}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def home():
    return "TFLite Runtime Flask Server is running."

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

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    predicted_index = int(np.argmax(output_data))
    predicted_label = class_indices.get(predicted_index, "Unknown")

    return jsonify({
        "message": "File successfully uploaded",
        "prediction": predicted_label,
        "confidence": float(np.max(output_data)),
        "filename": file.filename
    }), 200

if __name__ == '__main__':
    app.run(host='192.168.1.238', port=5000, debug=True)
