from flask import Flask, request, jsonify
from PIL import Image
import base64
import io
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import json

app = Flask(__name__)

# Load a pre-trained model from TensorFlow Hub
model = hub.load("https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4")

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0  # Normalize the image to [0, 1] range
    image = np.expand_dims(image, axis=0)
    return image

def decode_predictions(preds, top=1):
    CLASS_INDEX_PATH = tf.keras.utils.get_file(
        'imagenet_class_index.json',
        'https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json')
    with open(CLASS_INDEX_PATH) as f:
        class_index = json.load(f)
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [(class_index[str(i)][1], pred[i]) for i in top_indices]
        results.append(result)
    return results

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image_data = data['image']
    image_data = image_data.split(",")[1]
    image = Image.open(io.BytesIO(base64.b64decode(image_data)))
    processed_image = preprocess_image(image)
    preds = model(processed_image)
    preds = preds.numpy()[0]  # Convert TensorFlow tensor to numpy array
    prediction = decode_predictions(preds, top=1)[0][0][0]
    return jsonify({'guess': prediction})

if __name__ == '__main__':
    app.run(debug=True)
