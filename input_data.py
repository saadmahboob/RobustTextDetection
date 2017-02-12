'''
input_data.py

Created by: Ive Xu, 2017

This is a helper class for reading inputs from files.
'''

import cv2
from image_wrap import ImageWrap
from skimage.filters import threshold_otsu
from skimage import restoration
from skimage.morphology import closing, square

class InputData(object):

    def __init__(self, path):
        self.path = path

        # uninitialized attributes
        self.image_wrap = None
        self.image = None

    def read_file(self, file_path):
        self.image_wrap = ImageWrap(file_path)
        self.image_wrap.load_image()
        self.image = self.image_wrap.image
        self.gray = self.image_wrap.gray

    def raw_image(self):
        return self.image


    def preprocess_image(self):
        '''
        Denoise and increase contrast
        '''
        image = restoration.denoise_tv_chambolle(self.gray, weight=0.1)
        thresh = threshold_otsu(image)
        self.bw = closing(image > thresh, square(2))
        self.cleared = self.bw.copy()

        return self.cleared
