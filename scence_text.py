'''
scene_text.py

Created by: Ive Xu, 2017

'''

from input_data import InputData
from detector import Detector

import cv2

class SceneText(object):

    def __init__(self, file_path, classifier):
        self.path = file_path
        self.classifier = classifier

        # uninitialized attributes
        self.detector = None
        self.image = None
        self.processed_image = None
        self.input_data = None
        self.detector = None

    def get_input_data(self):
        self.input_data = InputData(self.path)
        self.input_data.read_file(self.path)

        self.image = self.input_data.image

    def preprocess_image(self):
        self.image = self.input_data.preprocess_image()
        #self.plot_image(self.image, 'preprocess image')

    def detect_text_candidates(self):
        imc = self.input_data.image_wrap
        self.detector = Detector(imc)
        self.detector.detectRegions()
        self.plot_image(self.detector.image, 'text candidates')

    def remove_maybe_non_text(self):
        pass

    def classify_candidates(self):
        pass

    def reconstruct_text(self):
        pass

    def plot_detected_image(self):
        pass

    def plot_image(self, image, msg):
        cv2.imshow(msg, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
