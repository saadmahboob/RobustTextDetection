'''
main.py

Created by: Ive Xu, 2017

'''

import cv2
from detector import Detector
from scence_text import SceneText
from classifier import Classifier

classifier = Classifier()

n_files = 6
for i in range(n_files):
    file_path = 'data/' + str(i+1) +'.jpg'
    print(file_path)
    # create instances of class and loads image from path
    scene = SceneText(file_path, classifier)

    scene.get_input_data()
    scene.preprocess_image()
    scene.detect_text_candidates()
    scene.remove_maybe_non_text()
    #scene.classify_candidates(classifier) # using SVM, or Neural Network
    #scene.reconstruct_text()
    scene.plot_detected_image()
