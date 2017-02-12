'''
mser.py

Created by: Ive Xu, 2017

use opencv 3.1 library to detect regions in image, MSER (Maximally Stable Extremal Regions) detector will be used.
'''

# Imports
import cv2
import numpy as np
import matplotlib.pyplot as plt

#from inputs import Inputs


# MSER wrapper class
class Detector(object):

    '''
    MSER class properties (@opencv Documentaion):

    _delta	         it compares (sizei−sizei−delta)/sizei−delta
    _min_area	     prune the area which smaller than minArea
    _max_area	     prune the area which bigger than maxArea
    _max_variation	 prune the area have simliar size to its children
    _min_diversity	 for color image, trace back to cut off mser with diversity less than min_diversity
    _max_evolution	 for color image, the evolution steps
    _area_threshold	 for color image, the area threshold to cause re-initialize
    _min_margin	     for color image, ignore too small margin
    _edge_blur_size	 for color image, the aperture size for edge blur
    '''

    def __init__(self, image_wrap):
        self.image_wrap = image_wrap
        self.image = self.image_wrap.image
        self.gray_image = self.image_wrap.gray

        self.min_area = int(self.image_wrap.min_width * self.image_wrap.min_height)
        self.max_area = int(self.image_wrap.max_width * self.image_wrap.max_height)

        print(self.min_area, self.max_area)
        self.descriptor = cv2.MSER_create(_min_area=0, _max_area=self.max_area, _delta=1)

        # uninitialized attributes
        self.regions = None
        self.hulls = None

    # Detect the regions using MSER descriptor
    def detectRegions(self):
        self.regions = self.descriptor.detectRegions(self.gray_image, None)
        #self.hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in self.regions]

        # iterate through each regions and draw rectangles over the points sets
        for points in self.regions:
            rect = cv2.minAreaRect(points)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(self.image, [box], 0, (0, 255, 0), 1)

    '''
    Several geometric properties for discriminating text versus non-text regions:
        - Aspect ratio:
        - Eccentricity:
        - Extent:
        - Solidarity:
        - EulenNumber:
    '''
