import easyocr
import numpy as np

# Initialize once (IMPORTANT)
reader = easyocr.Reader(['en'], gpu=False)

def extract_text(image):
    try:
        # Convert PIL → numpy
        image_np = np.array(image)

        results = reader.readtext(image_np)

        text = " ".join([res[1] for res in results])

        return text if text else " "
    except Exception as e:
        print("OCR Error:", e)
        return " "