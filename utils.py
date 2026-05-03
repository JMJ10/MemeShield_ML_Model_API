import pytesseract
def extract_text(image):
    try:
        return pytesseract.image_to_string(image).strip()
    except:
        return ""