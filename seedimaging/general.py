import numpy as np
from skimage.measure import find_contours
import pandas as pd
import Mask_RCNN.mrcnn.model as modellib
from Mask_RCNN.mrcnn import utils, visualize
import cv2

import random
import itertools
import colorsys
import copy

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def pad_mask_images(mask, padding_factor = 2):

    # Mask Polygon
    # Pad to ensure proper polygons for masks that touch image edges.
    padded_mask = np.zeros(
        (mask.shape[0] + padding_factor, mask.shape[1] + padding_factor), dtype=np.uint8)
    padded_mask[padding_factor//2:-padding_factor//2, padding_factor//2:-padding_factor//2] = mask
    #contours = find_contours(padded_mask, 0.5)
    return padded_mask

import math
def euclidean_distance(p1,p2):
    return math.sqrt(
        math.pow(p1[0] - p2[0],2) + math.pow(p1[1] - p2[1],2))


class RiceSeeds(modellib.MaskRCNN):

    def plot_single_detection(self, id):
        col = random_colors(len(self.bb))[id]

        imgclipped = copy.deepcopy(self._clip_image(self.image, id = id))
        maskimage = copy.deepcopy(self._get_mask(id = id, pad_factor=None))

        img = apply_mask(imgclipped, (maskimage*1).astype(np.uint8), col, alpha=0.2)
        linecolor = list((np.array(col)*255).astype(np.uint8))
        m = cv2.drawContours(img,[self._find_contours(id)],0,[int(i) for i in linecolor],1)
        return m

    def _clip_image(self, image, id = None):

        imgclipped = image[
            self.bb[id][0]:self.bb[id][2],self.bb[id][1]:self.bb[id][3]] 

        return imgclipped

    def _get_mask(self, id = None, pad_factor = 2):
        if id is None:
            id = 0
        maskimage = self._clip_image(self.masks[:,:,id], id)
        if pad_factor is not None:
            maskimage = pad_mask_images(maskimage, padding_factor=pad_factor)
            
        return maskimage

    def _find_contours(self, id):
        maskimage = self._get_mask(id)
        imgmas = (maskimage*255).astype(np.uint8)
        contours, hierarchy = cv2.findContours(imgmas, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        rect = cv2.minAreaRect(contours[0])
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        return box
    
    def calculate_oneseed_metrics(self, id):
        
        box = self._find_contours(id)
        maskimage = self._get_mask(id)
        distper = np.unique([euclidean_distance(box[i],box[i+1]) for i in range(len(box)-1) ])
        larger = distper[0] if distper[0]>distper[1] else distper[1]
        shorter = distper[0] if distper[0]<distper[1] else distper[1]
        area = np.sum(maskimage*1.)

        return pd.DataFrame({'id':[id],'height': [larger], 
                      'width': [shorter], 'area': [area]})

    def seeds_detect(self, image, verbose = 1):

        self.image = image
        if type(image) is not list:
            image =[image]
        
        results = self.detect(image, verbose = verbose)

        self.bb= results[0]['rois']
        self.masks = results[0]['masks']
        self.scores = results[0]['scores']
        self.class_ids = results[0]['class_ids']

        return results
    
    def display_all_detections(self):
        return visualize.display_instances(
            self.image, self.bb, self.masks, self.class_ids,
            ["",""], ["" for x in range(len(self.scores))],
            show_bbox=True, show_mask=True,
            title="")

    def __init__(self, config=None, model_dir=".", weigths = None) -> None:
        super().__init__(mode="inference", config=config, model_dir=model_dir)
        if weigths is not None:
            self.load_weights(weigths, by_name=True)