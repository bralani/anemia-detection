import cv2
import numpy as np

from PIL import ImageFilter, Image
from skimage import segmentation, color
from skimage import graph
from skimage.morphology import closing, square, dilation

black = (0, 0, 0)
def resize(img, percentage):
    """ Cv2 Resizing image utility
    Parameters
    ----------
    img: matrix shaped image
    percentage: % resolution to preserve from img
    Returns
    ----------
    img' : resized image according to %
    """
    width = int(img.shape[1] * percentage / 100)
    height = int(img.shape[0] * percentage / 100)
    dim = (width, height)
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

def nCut(img, thresh=0.01, cuts=10, compactness=10, n_cuts=0):
    """ Resizing image utility
    Parameters
    ----------
    img: matrix shaped image
    thresh: threshold for cut partitioning (r in original paper)
    cuts: approximate number of cuts from the original graph
    compactness: weight -> Color -> Spatial prop
 
    Returns
    ----------
    (img, ncut, kmeans) : original image, ncut segmentation, 
    kmeans segmentation
    """
    labels1 = segmentation.slic(img, compactness=compactness, 
    n_segments=cuts, start_label=1)
    kmeans = color.label2rgb(labels1, img, kind='avg', 
    bg_label=0)
    if n_cuts <= 0:
        n_cuts = int(cuts / 2)
        
    g = graph.rag_mean_color(kmeans, labels1, mode='similarity')
    labels2 = graph.cut_normalized(labels1, g, thresh=thresh, num_cuts=n_cuts)
    ncut = color.label2rgb(labels2, kmeans, kind='avg', bg_label=0)
    return img, kmeans, ncut

def check_ajacency(region1, region2):
    """
    Parameters
    ----------
    region1: List of coordinates (x,y)
    region2: List of coordinates (x,y)
    Returns True if there are at least 2 adjacent points, False 
    otherwise
    -------
    """
    for p1 in region1:
        for p2 in region2:
            if (abs(p1[0] - p2[0]) <= 1) and (abs(p1[1] - p2[1]) <= 1):
                return True
    return False

def jointRegions(img, seg_img, mask, fix_range, refine_intensity=1):
    """ Get single class based on region based on mask 
    coefficients
    Parameters
    ----------
    img: matrix shaped image
    seg_img: segmented image (k-means, n-cut, other approaches)
    mask: convolutional mask with coefficients (0,1)
    fix_range: lab color distance to accept smaller regions also 
    included in the mask.
    refine_intensity: intensity of the output contour smoothing. 
    A 0 value means no refinement, 1 is the default.
    Returns
    ----------
    res_img : binary segmented image
    """
    regions = {}
    regions_lab = {}
    regions_map = {}
    height, width, rgb = mask.shape
    lab = seg_img.astype("float32") / 255
    lab = cv2.cvtColor(lab, cv2.COLOR_BGR2Lab)
    for i in range(height):
        for j in range(width):
            rgb_t = tuple(mask[i][j])
            if rgb_t != black:
                lab_t = tuple(lab[i][j])
                if rgb_t in regions.keys():
                    regions[rgb_t] += 1
                    regions_lab[lab_t] += 1
                    regions_map[lab_t].append((i, j))
                else:
                    regions_lab.update({lab_t: 1})
                    regions.update({rgb_t: 1})
                    regions_map[lab_t] = [(i, j)]
    colors = select_colors(fix_range, regions_lab, regions_map)
    # create the binary mask with pixels with colors included in the modes list
    mask_bw = np.apply_along_axis(lambda p: np.full(3, 255 * (tuple(p) in colors), dtype=np.uint8), -1, lab)
    mask_bw = refine_mask(mask_bw, refine_intensity)
    # apply the mask to the original picture
    res_img = img * np.where(mask_bw == 255, 1, mask_bw)
    return res_img, mask_bw

def select_colors(fix_range, regions_lab, regions_map):
    modes = []
    try:
        mode = max(regions_lab, key=(lambda key: 
        regions_lab[key]))
        modes.append(mode)
        if fix_range > 0:
            for region in regions_lab:
                if region != mode:
                    # compute the euclidean distance in lab space
                    deltas = [abs(mode[i] - region[i]) for i in range(3)]
                    distance = np.linalg.norm(deltas)
                    # print(np.array(region), np.array(mode), deltas, distance)
                    if distance < fix_range and check_ajacency(regions_map[mode], regions_map[region]):
                        modes.append(region)
    except ValueError:
        print('No region selected!')
        pass
    return modes

def refine_mask(mask, intensity=1):
    """
    Smooths mask contours.
    Parameters
    ----------
    mask: The mask to refine. Should be a numpy array with 0 and 
    1 values.
    intensity: This parameter is mutiplied by the default 
    intensity
    Returns the refined mask.
    -------
    """
    r, c, rgb = mask.shape
    if intensity > 0:
        block_size = r * 0.032 * intensity
        if block_size < 1: block_size = 1
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = closing(mask, square(int(block_size * 1.5)))
        mask = Image.fromarray(mask).filter(ImageFilter.ModeFilter(int(block_size)))
        mask = np.array(mask)
        mask = dilation(mask, square(int(block_size * 0.45)))
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return mask

def gaussian_mask(seg_img, bin_img):
    """ Retrieve binary mask from convolution
    Parameters
    ----------
    seg_img: segmented image (k-means, n-cut, other approaches) 
    bin_img: binary representation of the convolved image
    
    Returns
    ----------
    res_img : binary segmented image
    """
    bin_img = cv2.cvtColor(bin_img.astype('uint8'), 
    cv2.COLOR_GRAY2BGR)
    return seg_img * bin_img