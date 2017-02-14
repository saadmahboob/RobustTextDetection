'''
main.py

Created by: Ive Xu, 2017

'''

import cv2
from detector import Detector
from scence_text import SceneText
from classifier import Classifier

classifier = Classifier()
file_path = 'data/lao.jpg'

# create instances of class and loads image from path
scene = SceneText(file_path, classifier)

scene.get_input_data()
#scene.preprocess_image()
scene.detect_text_candidates()
scene.remove_maybe_non_text()
#scene.classify_candidates(classifier)
#scene.reconstruct_text()
scene.plot_detected_image()
