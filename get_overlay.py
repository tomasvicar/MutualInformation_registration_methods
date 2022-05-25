from mat2gray import mat2gray
import numpy as np

def get_overlay(restult, VO_img):

    overlay = (mat2gray(np.dstack((VO_img,restult,VO_img)),[0,1]) * 255).astype(np.uint8)
    
    overlay[int(0.1*overlay.shape[0]):int(0.2*overlay.shape[0]),:,[0,2]] = 0
    overlay[int(0.2*overlay.shape[0]):int(0.3*overlay.shape[0]),:,1] = 0
    
    overlay[int(0.7*overlay.shape[0]):int(0.8*overlay.shape[0]),:,[0,2]] = 0
    overlay[int(0.8*overlay.shape[0]):int(0.9*overlay.shape[0]),:,1] = 0
    
    return overlay
