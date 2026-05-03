from flask import Flask, request, jsonify
from PIL import Image
from inference import predict
from utils import extract_text
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict_api():
    try:
        image_file = request.files["image"]
        img = Image.open(image_file.stream).convert("RGB")

        text = request.form.get("text", "")
        ocr_text = extract_text(img)

        final_text = text + " " + ocr_text
        print("------ DEBUG ------")
        print("Tesseract path:", pytesseract.pytesseract.tesseract_cmd)
        print("User text:", text)
        print("OCR text:", ocr_text)
        print("Final text:", final_text)
        print("-------------------")
        result = predict(img, final_text)

        return jsonify(result)

    except Exception as e:
        import traceback
        print("❌ ERROR OCCURRED:")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)