import pytesseract
def extract_text(image):
    try:
        return pytesseract.image_to_string(image, config='--psm 6').strip()
    except:
        return ""