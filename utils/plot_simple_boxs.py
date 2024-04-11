import torch
import os
import cv2, numpy as np


def render(img_array, bboxes):
    """
    Function to add bounding boxes to the image.
    Returns image with boxes applied.

    Args:
        img_array (np.ndarray | torch.Tensor): input image
        bboxes (torch.Tensor): yolo prediction -> Format in [x1, y1, x2, y2] not-normalized

    Returns:
        image (np.ndarray: image including boxes
    """
    
    img = img_array.astype(np.uint8)
    
    for box in bboxes:
        # for regular boxes
        x1 = box[0].cpu().numpy().astype(int)
        y1 = box[1].cpu().numpy().astype(int)
        x2 = box[2].cpu().numpy().astype(int)
        y2 = box[3].cpu().numpy().astype(int)            
            
        # Convention is NOT RGB but BGR .. what shall I say.. 
        # color = (7,102,236)
        color = (0,255,0)

        print(x1, y1, x2, y2)
        
        # cv2.line(img, (int(x1), int(y1)), (int(x2), int(y1)), color, 10)
        # cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, -1) # just a smaller rectangle, not using y2
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 3) # just a smaller rectangle, not using y2
        
        
    return img
    