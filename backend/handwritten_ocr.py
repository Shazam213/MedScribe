import importlib
import cv2
import numpy as np
import mxnet as mx
import random
import matplotlib.pyplot as plt
import gluonnlp as nlp
import matplotlib.patches as patches
from skimage import transform as skimage_tf, exposure
from tqdm import tqdm

from ocr.utils.expand_bounding_box import expand_bounding_box
from ocr.utils.sclite_helper import ScliteHelper
from ocr.utils.word_to_line import sort_bbs_line_by_line, crop_line_images
from ocr.utils.iam_dataset import IAMDataset, resize_image, crop_image, crop_handwriting_page
from ocr.utils.encoder_decoder import Denoiser, ALPHABET, encode_char, decode_char, EOS, BOS
from ocr.utils.beam_search import ctcBeamSearch

import ocr.utils.denoiser_utils
import ocr.utils.beam_search

importlib.reload(ocr.utils.denoiser_utils)
from ocr.utils.denoiser_utils import SequenceGenerator

importlib.reload(ocr.utils.beam_search)
from ocr.utils.beam_search import ctcBeamSearch

from ocr.paragraph_segmentation_dcnn import SegmentationNetwork, paragraph_segmentation_transform
from ocr.word_and_line_segmentation import SSD as WordSegmentationNet, predict_bounding_boxes
from ocr.handwriting_line_recognition import Network as HandwritingRecognitionNet, handwriting_recognition_transform
from ocr.handwriting_line_recognition import decode as decoder_handwriting, alphabet_encoding

ctx = mx.cpu()

# Initialize denoiser model
FEATURE_LEN = 150
denoiser = Denoiser(alphabet_size=len(ALPHABET), max_src_length=FEATURE_LEN, max_tgt_length=FEATURE_LEN, num_heads=16, embed_size=256, num_layers=2)
denoiser.load_parameters('models/denoiser2.params', ctx=ctx)
denoiser.hybridize(static_alloc=True)

ctx_nlp = mx.cpu()
language_model, vocab = nlp.model.big_rnn_lm_2048_512(dataset_name='gbw', pretrained=True)
moses_tokenizer = nlp.data.SacreMosesTokenizer()
moses_detokenizer = nlp.data.SacreMosesDetokenizer()

beam_sampler = nlp.model.BeamSearchSampler(beam_size=20,
                                           decoder=denoiser.decode_logprob,
                                           eos_id=EOS,
                                           scorer=nlp.model.BeamSearchScorer(),
                                           max_length=150)

generator = SequenceGenerator(beam_sampler, language_model, vocab, ctx_nlp, moses_tokenizer, moses_detokenizer)

# Utility functions for decoding
def get_arg_max(prob):
    '''The greedy algorithm to convert the output of the handwriting recognition network into strings.'''
    arg_max = prob.topk(axis=2).asnumpy()
    return decoder_handwriting(arg_max)[0]

def get_beam_search(prob, width=5):
    possibilities = ctcBeamSearch(prob.softmax()[0].asnumpy(), alphabet_encoding, None, width)
    return possibilities[0]

def get_denoised(prob, ctc_bs=False):
    '''Denoise the text using the denoiser model.'''
    if ctc_bs:
        text = get_beam_search(prob)
    else:
        text = get_arg_max(prob)
    src_seq, src_valid_length = encode_char(text)
    src_seq = mx.nd.array([src_seq], ctx=ctx)
    src_valid_length = mx.nd.array(src_valid_length, ctx=ctx)
    encoder_outputs, _ = denoiser.encode(src_seq, valid_length=src_valid_length)
    states = denoiser.decoder.init_state_from_encoder(encoder_outputs, encoder_valid_length=src_valid_length)
    inputs = mx.nd.full(shape=(1,), ctx=src_seq.context, dtype=np.float32, val=BOS)
    output = generator.generate_sequences(inputs, states, text)
    return output.strip()

# Main OCR function
def apply_handwriting_ocr(paragraph_segmented_image, predicted_bb, handwriting_line_recognition_net):
    """
    Function to apply Handwriting OCR to the segmented lines of text.
    
    Parameters
    ----------
    paragraph_segmented_image: np.array
        The segmented paragraph image.

    predicted_bb: np.array
        Predicted bounding boxes for words/lines.

    handwriting_line_recognition_net: mx.gluon.Block
        The pre-trained handwriting recognition model.

    Returns
    -------
    recognized_texts: list of str
        The recognized text from the input segmented lines.
    """

    # Step 1: Extract line images based on bounding boxes
    line_bbs = sort_bbs_line_by_line(predicted_bb, y_overlap=0.4)
    line_images = crop_line_images(paragraph_segmented_image, line_bbs)

    # Step 2: Initialize the list for storing character probabilities
    character_probs = []
    line_image_size = (60, 800)

    # Step 3: Process each line image through the handwriting recognition network
    for line_image in line_images:
        # Transform the line image to the required size
        transformed_line_image = handwriting_recognition_transform(line_image, line_image_size)
        # Get character probabilities from the network
        line_character_prob = handwriting_line_recognition_net(transformed_line_image.as_in_context(ctx))
        character_probs.append(line_character_prob)

    # Step 4: Decode character probabilities and apply denoising

    recognized_texts = []
    for line_character_prob in character_probs:
        
        # Denoise the output text
        decoded_line_denoiser = get_denoised(line_character_prob, ctc_bs=False)
        recognized_texts.append(decoded_line_denoiser)

    return recognized_texts
