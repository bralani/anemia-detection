import cv2
from matplotlib import pyplot as plt
import numpy as np
from operator import itemgetter
import exposure
import norm_cuts as nc
from scipy import ndimage
from threshold import threshold, refine

SHOW_NCUT = True
WRITE_STEPS = False
VERBOSE = False

L_min = 160
L_max = 211
L_25 = 193
L_75 = 180
A_min = 130
A_max = 145
A_25 = 139
A_75 = 135
B_min = 109
B_max = 123
B_25 = 118
B_75 = 114

def segment(img, fix_range=0, cuts=10, compactness=10, 
    blur_scale=1, n_cuts=10, n_thresh=0.1,
    imp_cuts=10, imp_thresh=0.1, imp_comp=6, 
    imp_fix=30.0, gamma=1):
 
    retinex_img = exposure.automatedMSRCR(img)
    preprocessed = exposure.preprocess(retinex_img)


    original, kmeans, ncut = nc.nCut(preprocessed, cuts=cuts, 
    compactness=compactness, n_cuts=n_cuts, thresh=n_thresh)

    img_threshold = threshold(retinex_img)

    test = calcola_area_e_filtra(ncut)

    mask_ncut = nc.gaussian_mask(test, img_threshold)

    sclera_ncut, mask_ncut = nc.jointRegions(img, ncut, mask_ncut, fix_range, 0)
    
    sclera_ncut, ncut, img_threshold = improve_precision_ncut(original=img,
    img_threshold=img_threshold,
    seg_img=ncut,
    mask=mask_ncut,
    preprocessed=preprocessed,
    ret_img=retinex_img,
    res_img=sclera_ncut,
    blur_scale=blur_scale,
    cuts=imp_cuts, thresh=imp_thresh, comp=imp_comp,
    fix=imp_fix, gamma=gamma)


    # converti in scala di grigi la foto
    sclera_ncut = cv2.cvtColor(sclera_ncut, cv2.COLOR_BGR2GRAY)

    #converti in bianco i valori più grandi di 0
    sclera_ncut = np.where(sclera_ncut > 0, 255, sclera_ncut)
    
    
    return sclera_ncut, img_threshold, kmeans, ncut

def calcola_area_e_filtra(image, soglia_y=380):

    image_backup = image.copy()

    #converti in scala di grigi
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # converti in nero tutti i valori più piccoli di 200
    image = np.where(image < 200, 0, image)
    
    # converti in bianco tutti i valori più grandi di 200
    image = np.where(image >= 200, 255, image)

    # Etichettatura delle regioni connesse
    labeled_image, num_labels = ndimage.label(image)
    
    # Applica il filtro di soglia y per scartare i pixel al di sotto della soglia
    filtered_image = np.copy(image)

    filtered_image[filtered_image > soglia_y] = 0

    # Ricalcola le dimensioni delle regioni connesse dopo l'applicazione del filtro di soglia
    filtered_sizes = ndimage.sum(filtered_image, labeled_image, range(1, num_labels + 1))
    
    if len(filtered_sizes) > 0:
        largest_area_index = np.argmax(filtered_sizes)
        
        # Crea una maschera per mantenere solo l'area più grande
        largest_area_mask = np.zeros_like(image)
        largest_area_mask[labeled_image == largest_area_index + 1] = 255


        image_output = cv2.bitwise_and(image_backup, image_backup, mask=largest_area_mask)

        return image_output
    else:
        return filtered_image



def suspect(img):
    l, alpha, beta = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))
    alpha_avg = np.average(alpha[l != 0])
    beta_avg = np.average(beta[l != 0])
    lum_avg = np.average(l[l != 0])
 # if VERBOSE:
    print(lum_avg, alpha_avg, beta_avg)
    l_distance = max(lum_avg - L_75, lum_avg - L_25) / (L_max-L_min)
    a_distance = max(alpha_avg - A_75, alpha_avg - A_25) / (A_max-A_min)
    b_distance = max(beta_avg - B_75, beta_avg - B_25) / (B_max-B_min)
    score = np.average([l_distance, a_distance, b_distance])
    print([l_distance, a_distance, b_distance], score)
    return score

def improve_precision_ncut(original, img_threshold, seg_img, 
    mask, preprocessed,
    ret_img, res_img, max_iter=3, 
    blur_scale=1, cuts=20, comp=6, thresh=0.1, fix=30.0, gamma=0.8):
    if blur_scale != 1 or gamma != 0.8:
        preprocessed = exposure.preprocess(ret_img, blur_scale=blur_scale, gamma=gamma)
    if SHOW_NCUT:
        i = 0
        history = []
        while i < max_iter:
            original = original * np.where(mask == 255, 1, mask)
            suspect_score = suspect(original)
            history.append({
            'result': res_img,
            'segment': seg_img,
            'threshold': img_threshold,
            'mask': mask,
            'score': abs(suspect_score)})
            if abs(suspect_score) > 0.05:
                preprocessed = preprocessed * np.where(mask == 255, 1, mask)
                indexes = preprocessed == 0
                preprocessed = np.where(indexes, 240, preprocessed)
                preprocessed, kmeans, seg_img = nc.nCut( 
                preprocessed, cuts=cuts * ((i + 1)) / 2,
                compactness=comp, thresh=thresh * (5 ** i), n_cuts= 6 + i)
                seg_img = np.where(indexes, 0, seg_img)
                mask_ncut = nc.gaussian_mask(seg_img, img_threshold)
                res_img = np.where(indexes, 0, res_img)
                res_img, mask = nc.jointRegions(res_img, seg_img, mask_ncut, fix, 0)
            else: 
                break
            i += 1
    best = min(history, key=lambda x: x['score'])
    return best['result'], best['segment'], best['threshold']

