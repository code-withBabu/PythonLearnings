import pytesseract
from PIL import Image

def extract_text_from_image(image_path, lang='eng'):
    """
    Extracts text from an image using OCR.

    Parameters:
    image_path (str): The path to the image file.
    lang (str): The language for OCR. Default is English ('eng').

    Returns:
    str: The extracted text from the image.
    """
    try:
        # Open the image file
        img = Image.open(image_path)
        
        # Use pytesseract to do OCR on the image
        text = pytesseract.image_to_string(img, lang=lang)
        
        return text
    except Exception as e:
        print(f"An error occurred: {e}")
        return ""
    
imagepath = input("Enter the image path: ")

extract_text_from_image(imagepath)