import numpy as np
from skimage.measure import find_contours
import pandas as pd

import sys
sys.path.append("Mask_RCNN") 

import Mask_RCNN.mrcnn.model as modellib
from Mask_RCNN.mrcnn.config import Config
from .image_functions import display_instances_cv
import cv2

from .utils import *
from .image_functions import add_label,random_colors, _apply_mask

import copy
import math
import matplotlib.pyplot as plt

def get_boundingboxfromseg(mask):
    
    pos = np.where(mask)
    xmin = np.min(pos[1])
    xmax = np.max(pos[1])
    ymin = np.min(pos[0])
    ymax = np.max(pos[0])
    
    return([xmin, ymin, xmax, ymax])
  
def getmidlewidthcoordinates(pinit,pfinal,alpha):

  xhalf=pfinal[0] - math.cos(alpha) * euclidean_distance(pinit,pfinal)/2
  yhalf=pinit[1] - math.sin(alpha) * euclidean_distance(pinit,pfinal)/2
  return int(xhalf),int(yhalf)

def getmidleheightcoordinates(pinit,pfinal,alpha):

  xhalf=math.sin(alpha) * euclidean_distance(pinit,pfinal)/2 + pinit[0]
  yhalf=math.cos(alpha) * euclidean_distance(pinit,pfinal)/2 + pinit[1]
  return int(xhalf),int(yhalf)

def pad_mask_images(mask, padding_factor = 2):

    # Mask Polygon
    # Pad to ensure proper polygons for masks that touch image edges.
    padded_mask = np.zeros(
        (mask.shape[0] + padding_factor, mask.shape[1] + padding_factor), dtype=np.uint8)
    padded_mask[padding_factor//2:-padding_factor//2, padding_factor//2:-padding_factor//2] = mask
    #contours = find_contours(padded_mask, 0.5)
    return padded_mask


def euclidean_distance(p1,p2):
    return math.sqrt(
        math.pow(p1[0] - p2[0],2) + math.pow(p1[1] - p2[1],2))

def check_weigth_path(path, suffix = 'h5'):
    if not path.endswith(suffix):
        downloadzip(path)
        path = filter_files_usingsuffix('models', suffix=suffix)
    
    return path

class InferenceConfig(Config):
    NAME = "riceseed"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1  # background + 1 seeds
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 8192
    DETECTION_MAX_INSTANCES = 1000
    IMAGE_RESIZE_MODE = "pad64"
    RPN_NMS_THRESHOLD = 0.4
    DETECTION_MIN_CONFIDENCE = 0


class RiceSeeds(modellib.MaskRCNN):
    
    
    def _get_heights_and_widths(self, id):

        p1,p2,p3,p4=self._find_contours(id)
        alpharad=math.acos((p2[0] - p1[0])/euclidean_distance(p1,p2))

        pheightu=getmidleheightcoordinates(p2,p3,alpharad)
        pheigthb=getmidleheightcoordinates(p1,p4,alpharad)
        pwidthu=getmidlewidthcoordinates(p4,p3,alpharad)
        pwidthb=getmidlewidthcoordinates(p1,p2,alpharad)

        return pheightu, pheigthb, pwidthu, pwidthb

    def _add_metriclines_to_single_detection(self, id, 
                    addlines = True, addlabel = True,
                    font = None,
                    fontcolor = None,
                    linetype = 2,
                    fontscale= 1,
                    thickness= 1):

        col = self.maskcolors[id]

        imgclipped = copy.deepcopy(self._clip_image(self.image, self.bbs[id]))
        maskimage = copy.deepcopy(self._get_mask(id = id, pad_factor=None))

        img = _apply_mask(imgclipped, (maskimage*1).astype(np.uint8), col, alpha=0.2)
        linecolor = list((np.array(col)*255).astype(np.uint8))
        m = cv2.drawContours(img,[self._find_contours(id)],0,[int(i) for i in linecolor],1)
        if addlines:
            pheightu, pheigthb, pwidthu, pwidthb = self._get_heights_and_widths(id)
            m = cv2.line(m, pheightu, pheigthb, (0,0,0), 1)
            m = cv2.line(m, pwidthu, pwidthb, (0,0,0), 1)
        
        if addlabel:
            
            xpos = m.shape[1]//2 - int(m.shape[1]//2*0.80)
            ypos = m.shape[0]//2 - int(m.shape[0]//2*0.80)
            m = add_label(m, str(id),
                        xpos, 
                        ypos,
                        font = font,
                        fontscale = fontscale,
                        fontcolor = [
                int(i*255) for i in self.maskcolors[id]],
                thickness = thickness,
                linetype = linetype)

        return m

    def _get_mask(self, id = None, pad_factor = 2):
        if id is None:
            id = 0
        if self._size is not None:
          maskc, bb = self._resize_mask(self.masks[:,:,id]*1, self._size)
          maskc = maskc.astype(bool)
        else:
            maskc = self.masks[:,:,id]
            bb = self.bbs[id]
            
        maskimage = self._clip_image(maskc, bb)
        if pad_factor is not None:
            maskimage = pad_mask_images(maskimage, padding_factor=pad_factor)
            
        return maskimage
    
    @staticmethod
    def _clip_image(image, bb):

        imgclipped = image[
            bb[0]:bb[2],bb[1]:bb[3]] 

        return imgclipped
     
    @staticmethod
    def _resize_mask(maskimg, new_size):
        imgc = (maskimg).astype(np.uint16)
        imgc = cv2.resize(imgc, [new_size[1],new_size[0]], 
                                  interpolation = cv2.INTER_AREA)
        bb = get_boundingboxfromseg(imgc)
        
        return imgc, bb
        
    
    def _original_size(self, original_size):
        """
        Resize masks and bounding boxes to original image size.
        """
        
        msksc = np.zeros((original_size[0],original_size[1], self.masks.shape[2]))
        bbsc = [0]* self.masks.shape[2]
        
        for i in range(msksc.shape[2]):
            imgc = (self.masks[:,:,i]*1).astype(np.uint8)
            msksc[:,:,i] = cv2.resize(imgc, 
                                  [original_size[1],original_size[0]], 
                                  interpolation = cv2.INTER_AREA)
            msksc[:,:,i] = msksc[:,:,i] == 1
            if len(bbsc)>0 and np.sum(msksc[:,:,i])>0:
                bbsc[i] = get_boundingboxfromseg(msksc[:,:,i])
            #else:
            #    bbsc[i] = []
        self.masks = msksc.astype(bool)
        self.bbs = np.array(bbsc)
    
    def _find_contours(self, id, resize = None):
        maskimage = self._get_mask(id)
          
        imgmas = (maskimage*255).astype(np.uint8)
        contours, _ = cv2.findContours(imgmas, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        rect = cv2.minAreaRect(contours[0])
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        return box
    
    def plot_individual_seed(self, id, figsize = (10,10),**kwargs):
        f, ax = plt.subplots(figsize = figsize)
        ax.imshow(self._add_metriclines_to_single_detection(id, **kwargs))

    def calculate_oneseed_metrics(self, id):
        
        box = self._find_contours(id)
        maskimage = self._get_mask(id)
        distper = np.unique([euclidean_distance(box[i],box[i+1]) for i in range(len(box)-1) ])
        larger = distper[0] if distper[0]>distper[1] else distper[1]
        shorter = distper[0] if distper[0]<distper[1] else distper[1]
        area = np.sum(maskimage*1.)

        return pd.DataFrame({'id':[id],'height': [larger], 
                      'width': [shorter], 'area': [area]})

    def seeds_summary(self):

        summarylist = []
        for i in self.ids:
            summarylist.append(
                self.calculate_oneseed_metrics(i))

        return pd.concat(summarylist)

    def seeds_detect(self, image, verbose = 1, score_threshold = 0.95, original_size = None):

        self.image = image
        if type(image) is not list:
            image =[image]
        
        results = self.detect(image, verbose = verbose)
        self.bb, self.masks, self.scores, self.class_ids = None, None, None, None
      
        if score_threshold:
          pos = [i for i, val in enumerate(results[0]['scores']) if val >= score_threshold]
          if len(pos)>0:
            self.bb= results[0]['rois'][pos]
            self.masks = results[0]['masks'][:,:,pos]
            self.scores = results[0]['scores'][pos]
            self.class_ids = results[0]['class_ids'][pos]
        else:
            self.bb= results[0]['rois']
            self.masks = results[0]['masks']
            self.scores = results[0]['scores']
            self.class_ids = results[0]['class_ids']
        if len(self.bb)>0:
          self.maskcolors = random_colors(len(self.bb))
          self.ids = list(range(len(self.bb)))

        if score_threshold is not None:
          self._size = original_size
        else:
          self._size = None
        return results
    
    def plot_all_detections(self, figsize = (15,15),show_bbox=True, show_mask=True,objfontscale = 0.8):
        img = display_instances_cv(
            self.image, self.bb, 
            self.masks, self.class_ids,
            ["",""], ["" for x in range(len(self.scores))],
            show_bbox=show_bbox, show_mask=show_mask,
            title="", colors = self.maskcolors, captions=self.ids, objfontscale = objfontscale)

        f, ax = plt.subplots(figsize = figsize)
        #ax.imshow(m)

        ax.imshow(img)
        return f

    def __init__(self, config=None, weigths = None, model_suffix = 'h5') -> None:

        if config is None:
            config = InferenceConfig()

        super().__init__(mode="inference", config=config, model_dir=".")
        if weigths is not None:
            self.load_weights(check_weigth_path(weigths, suffix=model_suffix), by_name=True)
