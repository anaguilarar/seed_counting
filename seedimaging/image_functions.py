

import cv2
import random
import colorsys
import numpy as np

from . import general 


def display_instances_cv(image, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None, objfontscale = 1):
    """
    Display all bounding boxes, modified from Mask_RCNN
    this version only uses cv2

    Parameters:
    ------

    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    

    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # Generate random colors
    colors = colors or random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]

    masked_image = image.astype(np.uint8).copy()

    for i in range(N):
        color = colors[i]
        #print(color)
        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        
        y1, x1, y2, x2 = boxes[i]

        if show_bbox:
            masked_image = cv2.rectangle(
                masked_image,(x1,y1),(x2,y2),
                 color = [int(i*255) for i in color],
                 thickness =2)

        widthrect = abs(x1 - x2)
        heigthrect = abs(y1 - y2)
        # Label
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            caption = "{} {:.3f}".format(label, score) if score else label

        else:
            caption = str(captions[i])
        
        mask = masks[:, :, i]
        if show_mask:
            masked_image = general._apply_mask(masked_image, mask, color)
        
        ymask,xmask = np.mean(np.where(mask),axis = 1).astype(int)
        masked_image = add_label(masked_image, caption,xmask - int(widthrect//2*0.8), 
                        ymask- int(heigthrect//2*0.2),
                        fontscale = objfontscale,
                        thickness= 2,
                        fontcolor = (255,255,255))
        
        imgmas = (mask*255).astype(np.uint8)
        contourscv2, _ = cv2.findContours(imgmas, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        masked_image = cv2.drawContours(masked_image, contourscv2, 0, 
        color = [int(i*255) for i in color])

    return masked_image


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



def add_label(img,
        label ,
        xpos,ypos,
        font = None,
        fontcolor = None,
        linetype = 2,
        fontscale= None,
        thickness= 1):
    
        fontscale = fontscale or 0.3
        fontcolor = fontcolor or (0,0,0)
        font = font or cv2.FONT_HERSHEY_SIMPLEX    

        img = cv2.putText(img,
                label, 
                (xpos, ypos), 
                font, 
                fontscale,
                fontcolor,
                thickness,
                linetype)

        return img