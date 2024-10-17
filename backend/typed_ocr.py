import cv2
import numpy as np
import pytesseract
from PIL import Image

def mask_bounding_box(image, bbox):
    """
    Function to mask the area inside the bounding box on the image.
    
    Parameters
    ----------
    image_path: str
        Path to the image file.
    
    bbox: tuple
        Bounding box coordinates in the format (x, y, w, h).
    
    Returns
    -------
    masked_image: np.array
        The image with the bounding box area masked.
    """
    masked_image = image.copy()

    print(bbox)
    # Unpack the bounding box coordinates
    x, y, w, h = [int(coord) for coord in bbox]

    # Create a mask (black rectangle) inside the bounding box
    cv2.rectangle(masked_image, (x, y), (x + w, y + h), (0, 0, 0), thickness=-1)  # -1 thickness fills the box
    
    return masked_image

def apply_typed_ocr(image,bbox):
    """
    Function to perform OCR on the image using pytesseract.
    
    Parameters
    ----------
    image: np.array
        The input image on which OCR will be performed.
    
    Returns
    -------
    extracted_text: str
        The text extracted from the image.
    """

    masked_image= mask_bounding_box(image,bbox)
    cv2.imwrite('output/masked_img.jpg',masked_image)
    # Convert image to PIL format (pytesseract works with PIL images)
    pil_image = Image.fromarray(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))

    # Perform OCR using pytesseract
    extracted_text = pytesseract.image_to_string(pil_image)

    return extracted_text
