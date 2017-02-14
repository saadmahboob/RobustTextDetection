'''
mser.py

Created by: Ive Xu, 2017

use opencv 3.1 library to detect regions in image, MSER
(Maximally Stable Extremal Regions) detector will be used.
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from skimage import morphology
from scipy.ndimage.morphology import distance_transform_edt
from skimage import morphology

class Detector(object):

    DELTA = 1

    ASPECT_THRESH = 3
    ECCENT_THRESH = 0.995
    SOLIDITY_THRESH = 0.3
    EXTENT_THRESH_LOWER = 0.2
    EXTENT_THRESH_UPPER = 0.9
    EULER_THRESH = -4

################################################################################
    '''
    MSER class properties (@opencv Documentaion):

    _delta	         it compares (sizei−sizei−delta)/sizei−delta
    _min_area	       prune the area which smaller than minArea
    _max_area	       prune the area which bigger than maxArea
    _max_variation	 prune the area have simliar size to its children
    _min_diversity	 for color image, trace back to cut off mser with diversity less than min_diversity
    _max_evolution	 for color image, the evolution steps
    _area_threshold	 for color image, the area threshold to cause re-initialize
    _min_margin	     for color image, ignore too small margin
    _edge_blur_size	 for color image, the aperture size for edge blur
    '''
################################################################################

    def __init__(self, image_wrap):
        self.image_wrap = image_wrap
        self.image = self.image_wrap.image
        self.gray = self.image_wrap.gray

        self.min_area = int(self.image_wrap.min_width * self.image_wrap.min_height)
        self.max_area = int(self.image_wrap.max_width * self.image_wrap.max_height)

        self.mser = cv2.MSER_create(_min_area=self.min_area,_max_area=self.max_area, _delta=Detector.DELTA)
        self.filteredRegions = []

        # uninitialized attributes
        self.regions = None
        self.hulls = None
        print("min_area %d, max_area %d" % (self.min_area, self.max_area))

################################################################################

    # Detect the regions using MSER descriptor
    def detectRegions(self):
        self.regions = self.mser.detectRegions(self.gray, None)
        #self.hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in self.regions]

        print("Regions before filtered : %d" % len(self.regions))
        # iterate through each regions and draw rectangles over the points sets
        for cnt in self.regions:
            try:
                self.filter_regions(cnt)
            except:
                pass

        print("Regions after filtered : %d" % len(self.filteredRegions))


################################################################################

    def filter_regions(self, cnt):
        '''
        Several geometric properties for discriminating text versus non-text regions:
            - Aspect ratio:
            - Eccentricity:
            - Extent:
            - Solidarity:
        '''
        x,y,w,h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        rect_area = w*h
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)

        ######### Ascpect Ratio #########
        aspect_ratio = float(w)/h

        ######### Eccentricity #########
        ellipse = cv2.fitEllipse(cnt)

        # center, axis_length and orientation of ellipse
        (center, axes, orientation) = ellipse

        # length of MAJOR and minor axis
        majoraxis_length = max(axes)
        minoraxis_length = min(axes)

        # eccentricity = sqrt( 1 - (ma/MA)^2) --- ma= minor axis --- MA= major axis
        eccentricity = np.sqrt(1-(minoraxis_length/majoraxis_length)**2)

        ######### Extent #########
        extent = float(area)/rect_area

        ######### Solidity #########
        try:
            solidity = float(area)/hull_area
        except:
            solidity = 0


        ######### Descriminating the obvious text/non-text regions ###########
        if ((aspect_ratio < Detector.ASPECT_THRESH) and (eccentricity < Detector.ECCENT_THRESH)
            and ((extent > Detector.EXTENT_THRESH_LOWER) or (extent < Detector.EXTENT_THRESH_UPPER))
            and (solidity > Detector.SOLIDITY_THRESH)):
            self.filteredRegions.append(cnt)

        #print(aspect_ratio, eccentricity, extent, solidity)

################################################################################

    # Perform a distance transform and binary thinning operation on the detected regions
    '''
    def strokeWidthFilter(self, cont):

        im_bw = []
        for c in cont:
            p= self.gray[c]
            print(p.shape)
            if p < 127:
                im_bw.append(0)
            else:
                im_bw.append(1)

        (thresh, im_bw) = cv2.threshold(regionImage, 127, 255, cv2.THRESH_BINARY)
        print(im_bw)
        distanceImage = distance_transform_edt(1-im_bw)
        skeletonImage = morphology.skeletonize(im_bw)
        strokeWidthImage = distanceImage
        #strokeWidthImage = strokeWidthImage[-skeletonImage]
    '''

################################################################################

    def drawRegions(self, regions):
        img = self.image.copy()
        for cnt in regions:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            cv2.drawContours(img, [box], 0, (0, 255, 0), 1)
        return img
