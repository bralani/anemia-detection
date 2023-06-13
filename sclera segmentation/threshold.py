import cv2
import numpy as np
from skimage.morphology import closing, opening, square

MIN_RED = 220
MIN_GREEN = 180
MIN_BLUE = 230

def threshold(img: np.array, tolerance: float = 0.975, 
    min_tolerance: float = 0.7, tol_decrease_rate: float = 0.95) -> np.array:
    height, width, rgb = img.shape
    min_region_size = height * width / 200
    img_res = np.zeros((height, width))
    while np.count_nonzero(img_res == 1) < min_region_size and tolerance > min_tolerance:
        blue, green, red = cv2.split(img)
        cv2.threshold(red, int(MIN_RED * tolerance), dst=red, 
        maxval=255, type=cv2.THRESH_BINARY)
        cv2.threshold(green, int(MIN_GREEN * tolerance), 
        dst=green, maxval=255, type=cv2.THRESH_BINARY)
        cv2.threshold(blue, int(MIN_BLUE * tolerance), dst=blue, 
        maxval=255, type=cv2.THRESH_BINARY)
        mask = red * green * blue
        img_res = refine(mask)
        tolerance *= tol_decrease_rate
        img_res
    return img_res

def refine(img):
    height, width = img.shape
    block_size = height * 0.025
    if block_size < 1: block_size = 1
    img = opening(img, square(int(block_size * 1.25)))
    img = closing(img, square(int(block_size)))
    img = cv2.GaussianBlur(img, (7, 7), 10)
    img = np.where(img > 0, 1, img)
    return img
