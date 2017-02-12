'''
image_container.py

Created by: Ive Xu, 2017

'''

from skimage.io import imread
import pandas as pd
import numpy as np
import cv2


class ImageWrap(object):

    MIN_RATIO = 0.05 # ratio for MINIMUM width, height for text candidates
    MAX_RATIO = 0.1# ratio for MAXIMUM width, height fro text candidate

    def __init__(self, image_file):
        self.image_file = image_file

        # uninitialized attributes
        self.width = None
        self.height = None
        self.min_width = None
        self.min_height = None
        self.max_width = None
        self.max_height = None

    def load_image(self):
        #
        self.image = cv2.imread(self.image_file)
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.set_spec()

    def set_spec(self):
        shape = self.image.shape
        self.width = shape[0]
        self.height = shape[1]
        print(self.width, self.height)
        self.min_width = self.width * self.MIN_RATIO
        self.min_height = self.height * self.MIN_RATIO
        self.max_width = self.width * self.MAX_RATIO
        self.max_height = self.height * self.MAX_RATIO

    def get_image(self):
        return self.image
