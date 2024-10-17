import cv2
import mxnet as mx
import numpy as np
import matplotlib.pyplot as plt

# Import custom OCR utilities and models
from ocr.utils.expand_bounding_box import expand_bounding_box
from ocr.utils.word_to_line import sort_bbs_line_by_line, crop_line_images
from ocr.utils.iam_dataset import crop_handwriting_page
from ocr.paragraph_segmentation_dcnn import SegmentationNetwork, paragraph_segmentation_transform
from ocr.word_and_line_segmentation import SSD as WordSegmentationNet, predict_bounding_boxes
from ocr.handwriting_line_recognition import Network as HandwritingRecognitionNet, handwriting_recognition_transform
from ocr.handwriting_line_recognition import decode as decoder_handwriting, alphabet_encoding

# Set the context for MXNet (CPU or GPU)
ctx = mx.cpu()

# Image resizing helper function
def resize_image(image, desired_size):
    size = image.shape[:2]
    if size[0] > desired_size[0] or size[1] > desired_size[1]:
        ratio_w = float(desired_size[0]) / size[0]
        ratio_h = float(desired_size[1]) / size[1]
        ratio = min(ratio_w, ratio_h)
        new_size = tuple([int(x * ratio) for x in size])
        image = cv2.resize(image, (new_size[1], new_size[0]))
        size = image.shape

    delta_w = max(0, desired_size[1] - size[1])
    delta_h = max(0, desired_size[0] - size[0])
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = image[0][0]
    if color < 230:
        color = 230
    image = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=float(color)
    )
    crop_bb = (
        left / image.shape[1],
        top / image.shape[0],
        (image.shape[1] - right - left) / image.shape[1],
        (image.shape[0] - bottom - top) / image.shape[0],
    )
    image[image > 230] = 255
    return image, crop_bb

# Preprocess image function
def _pre_process_image(img_in, _parse_method):
    im = cv2.imread(img_in, cv2.IMREAD_GRAYSCALE)
    if im is None or np.size(im) == 1:
        return None
    MAX_IMAGE_SIZE_FORM = (1120, 800)
    MAX_IMAGE_SIZE_LINE = (60, 800)
    MAX_IMAGE_SIZE_WORD = (30, 140)
    if _parse_method in ["form", "form_bb"]:
        im, _ = resize_image(im, MAX_IMAGE_SIZE_FORM)
    elif _parse_method == "line":
        im, _ = resize_image(im, MAX_IMAGE_SIZE_LINE)
    elif _parse_method == "word":
        im, _ = resize_image(im, MAX_IMAGE_SIZE_WORD)
    img_arr = np.asarray(im)
    return img_arr

# Function 1: Image segmentation
def segment_image(image_path,paragraph_segmentation_net,word_segmentation_net):
    """
    Function 1: Takes an image as input and applies paragraph and word segmentation.
    Outputs:
    - The segmented handwritten part (cropped paragraph image with word bounding boxes).
    - The original image with paragraph bounding boxes overlaid.
    """
    # Preprocess the image
    img = _pre_process_image(image_path, 'form')
    if img is None:
        print("Image could not be loaded or is corrupt.")
        return None, None


    # Apply paragraph segmentation
    form_size = (1120, 800)
    resized_image = paragraph_segmentation_transform(img, form_size)
    bb_predicted = paragraph_segmentation_net(resized_image.as_in_context(ctx))
    bb_predicted = bb_predicted[0].asnumpy()
    bb_predicted = expand_bounding_box(
        bb_predicted, expand_bb_scale_x=0.03, expand_bb_scale_y=0.03
    )

    # Draw paragraph bounding box on the original image
    image_with_bb = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
    (x, y, w, h) = bb_predicted
    image_h, image_w = img.shape
    (x1, y1, w1, h1) = (
        int(x * image_w),
        int(y * image_h),
        int(w * image_w),
        int(h * image_h),
    )
    cv2.rectangle(
        image_with_bb, (x1, y1), (x1 + w1, y1 + h1), color=(0, 0, 255), thickness=2
    )
    bb_coord = (x1, y1, w1, h1)
    # Crop the paragraph image
    segmented_paragraph_size = (700, 700)
    paragraph_segmented_image = crop_handwriting_page(
        img, bb_predicted, image_size=segmented_paragraph_size
    )

    # Apply word segmentation on the paragraph_segmented_image
    min_c = 0.1
    overlap_thres = 0.1
    topk = 600
    predicted_bb = predict_bounding_boxes(
        word_segmentation_net,
        paragraph_segmented_image,
        min_c,
        overlap_thres,
        topk,
        ctx,
    )

    # Draw word bounding boxes on the paragraph_segmented_image
    image_with_word_bbs = cv2.cvtColor(paragraph_segmented_image.copy(), cv2.COLOR_GRAY2BGR)
    image_h, image_w = paragraph_segmented_image.shape
    for bb in predicted_bb:
        (x, y, w, h) = bb
        (x1, y1, w1, h1) = (
            int(x * image_w),
            int(y * image_h),
            int(w * image_w),
            int(h * image_h),
        )
        cv2.rectangle(
            image_with_word_bbs,
            (x1, y1),
            (x1 + w1, y1 + h1),
            color=(255, 0, 0),
            thickness=1,
        )

    return bb_coord, image_with_bb, predicted_bb, paragraph_segmented_image
