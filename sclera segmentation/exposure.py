import json
import cv2
import numpy as np
from skimage import exposure
from retinex_filter import automatedMSRCR2

def gamma_correction(image, gamma):
    lookUpTable = np.empty((1, 256), np.uint8)
    for i in range(256):
        lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    return cv2.LUT(image, lookUpTable)

def automatedMSRCR(image):
    height, width, rgb = image.shape
    scale = height * 0.0072
 
    sigma_list = np.array([15.0, 80.0, 250.0]) * np.full(3, scale)
    sigma_list = np.array(sigma_list, dtype=int)
    return automatedMSRCR2(image, sigma_list)

def preprocess(image, blur_scale=1.5, gamma=0.8, cutoff=0.65, gain=8):
    modified = image
    height, width, rgb = image.shape
    modified = gamma_correction(modified, gamma)
    k = int(np.ceil(height * 0.11)) // 2 * 2 + 1
    sigma = int(height * 0.04 * blur_scale)
    modified = cv2.GaussianBlur(modified, (k, k), sigmaX=sigma, sigmaY=sigma)
    modified = exposure.adjust_sigmoid(modified, cutoff=cutoff, 
    gain=gain)
    modified = gamma_correction(modified, gamma)
    return modified
