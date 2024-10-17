import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_image(image_path):
    """
    Load the image from the given path.
    """
    image = cv2.imread(image_path)
    return image

def convert_to_grayscale(image):
    """
    Convert the image to grayscale.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def noise_reduction(image):
    """
    Apply noise reduction using fastNlMeansDenoising.
    """
    return cv2.fastNlMeansDenoising(image, None, h=10, templateWindowSize=7, searchWindowSize=21)

def adjust_contrast(image, alpha=1.1, beta=0):
    """
    Apply contrast adjustment to the image.
    """
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def thresholding(image):
    """
    Apply adaptive thresholding to the image.
    """
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY, 11, 2)

def otsu_thresholding(image):
    """
    Apply Otsu's thresholding to the image.
    """
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

def skew_correction(image):
    """
    Correct skewness in the image using Hough Line Transform to detect text orientation.
    """
    # Convert the image to grayscale
    gray = convert_to_grayscale(image)

    # Invert the grayscale image (optional, depending on the document type)
    gray = cv2.bitwise_not(gray)

    # Threshold the image to binary
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # Detect edges using Canny
    edges = cv2.Canny(thresh, 50, 150, apertureSize=3)

    # Use Hough Line Transform to detect lines in the image
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    # Calculate the angle of the detected lines
    angles = []
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            angle = np.degrees(theta)
            if angle > 45:  # Correct for diagonal lines
                angle -= 90
            angles.append(angle)

        # Find the median angle (to minimize noise effect)
        median_angle = np.median(angles)

        # Rotate the image to correct the skew
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        return rotated
    else:
        # If no lines are detected, return the original image
        return image


def resize_image(image, scale=1.5):
    """
    Resize the image by the given scale factor.
    """
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

def preprocess_image(image):
    """
    Full preprocessing pipeline for the input image.
    """
    # Load the image
    # image = load_image(image_path)

    # Correct skewness using Hough Line Transform
    corrected_image = skew_correction(image)

    # Convert to grayscale
    grayscale_image = convert_to_grayscale(corrected_image)

    # Reduce noise using fastNlMeansDenoising
    denoised_image = noise_reduction(grayscale_image)

    # Adjust contrast using histogram equalization
    contrast_image = adjust_contrast(denoised_image)

    # Apply adaptive thresholding (or alternatively, you can use Otsu's thresholding)
    thresholded_image = otsu_thresholding(contrast_image)

    # Optional: Resize image for better OCR performance
    resized_image = resize_image(thresholded_image)

    return resized_image
