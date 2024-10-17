import cv2
from preprocessing import preprocess_image
from handwritten_ocr import *
from segment_handwritten_text import *
from typed_ocr import *


def perform_ocr(image):
    # image_path = 'test/test3.png'
    
    # Preprocess the image
    processed_image = preprocess_image(image)
    cv2.imwrite('output/processed_image.jpg', processed_image)
    

    # initialize the models

    # Initialize the paragraph segmentation network

    paragraph_segmentation_net = SegmentationNetwork(ctx=mx.cpu())
    paragraph_segmentation_net.cnn.load_parameters("models/paragraph_segmentation2.params", ctx=ctx)
    paragraph_segmentation_net.hybridize()


    # Initialize the word segmentation network
    word_segmentation_net = WordSegmentationNet(2, ctx=mx.cpu())
    word_segmentation_net.load_parameters("models/word_segmentation2.params", ctx=ctx)
    word_segmentation_net.hybridize()

    handwriting_line_recognition_net = HandwritingRecognitionNet(rnn_hidden_states=512,
                                                             rnn_layers=2, ctx=ctx, max_seq_len=160)
    handwriting_line_recognition_net.load_parameters("models/handwriting_line8.params", ctx=ctx)
    handwriting_line_recognition_net.hybridize()

    bb_predicted,image_with_bb, predicted_bb, paragraph_segmented_image = segment_image("output/processed_image.jpg",paragraph_segmentation_net,word_segmentation_net)
    
    cv2.imwrite('output/image_with_bounding_boxes.jpg', image_with_bb)
    if isinstance(paragraph_segmented_image, np.ndarray):
        recognized_texts_handwritten = apply_handwriting_ocr(paragraph_segmented_image, predicted_bb, handwriting_line_recognition_net)

    # Output the recognized texts
    handwritten_text = ""
    if len(recognized_texts_handwritten)!=0 :
        for i, text in enumerate(recognized_texts_handwritten):
            handwritten_text += text + '\n'
    else:
        recognized_texts_handwritten=""
    del paragraph_segmentation_net
    del word_segmentation_net
    del handwriting_line_recognition_net

    
    recognized_texts_typed = apply_typed_ocr(image_with_bb,bb_predicted)
    if len(recognized_texts_typed)!= 0:
        print(recognized_texts_typed)
    else:
        recognized_texts_typed=""

    return recognized_texts_typed+handwritten_text
