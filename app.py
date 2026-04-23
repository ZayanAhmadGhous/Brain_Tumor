from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image
import io

app = Flask(__name__)

# Load model safely once
model = tf.keras.models.load_model("brain_tumor_model.keras")

# Labels (safer than hardcoding logic)
labels = ["No Tumor Detected", "Tumor Detected"]

def preprocess_image(img):
    img = img.convert("L")  # grayscale
    img = img.resize((150, 150))
    img_array = np.array(img)
    img_array = img_array.reshape(1, 150, 150, 1) / 255.0
    return img_array


@app.route("/predict", methods=["POST"])
def predict():

    # 🔴 1. Check if image is provided
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        file = request.files["image"]

        # 🔴 2. Read image safely
        image = Image.open(io.BytesIO(file.read()))

        # 🔴 3. Preprocess
        processed = preprocess_image(image)

        # 🔴 4. Predict
        prediction = model.predict(processed)[0]

        class_index = np.argmax(prediction)
        result = labels[class_index]
        confidence = float(prediction[class_index])

        # 🔴 5. Response
        return jsonify({
            "result": result,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# 🔴 Run app
if __name__ == "__main__":
    app.run(debug=True)