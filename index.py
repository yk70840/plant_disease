from flask import Flask, jsonify, request
from PIL import Image
from python_scripts import predict

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return {"message":"Hello"}

@app.route("/predict", methods=["POST"])
def process_image():
  try:
    # Check if image file is uploaded
    if "image" not in request.files:
      return jsonify({"error": "No image file uploaded"}), 400

    # Open the image file
    img = Image.open(request.files["image"].stream)

    # Call the prediction function
    result = predict.predict(img=img)

    # Return the prediction result
    return jsonify(result), 200
  except Exception as e:
    # Handle any errors during processing
    return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
