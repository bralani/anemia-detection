import cv2
import numpy as np
import matplotlib.pyplot as plt

#BGR
vessels_density = []

value_r_minous_g_img_sclera = []
value_r_minous_g_img_vessels = []

sclera_quantiles_bgr = []
vessels_quantiles_bgr = []

dev_std_sclera = []
dev_std_vessels = []

#Lab
vessels_density_lab = []

value_a_img_sclera = []
value_a_img_vessels = []

sclera_quantiles_lab = []
vessels_quantiles_lab = []

dev_std_sclera_cielab = []
dev_std_vessels_cielab = []

vessels_colors_white_deviations_cielab = []
vessels_colors_white_quantiles_cielab = []

def calculate_bgr(img_segmented, sclera_vessels_masked):
    # calculate the metrics
    number_sclera_pixel = 0
    number_vessels_pixel = 0
    number_sclera_pixel_coulored = 0
    number_vessels_pixel_coulored = 0

    b = 0
    g = 0
    r = 0

    value_r_minous_g_pixel_sclera = 0
    value_r_minous_g_pixel_vessels = 0


    number_sclera_pixel = 0
    number_vessels_pixel = 0
    value_r_minous_g_pixel_sclera = 0
    value_r_minous_g_pixel_vessels = 0

    # counts coloured pixels
    img_gray = cv2.cvtColor(img_segmented, cv2.COLOR_BGR2GRAY)
    _, img_binaria = cv2.threshold(img_gray, 1, 255, cv2.THRESH_BINARY)
    number_sclera_pixel_coulored = cv2.countNonZero(img_binaria)

    # counts total pixels
    number_sclera_pixel = img_binaria.size

    # sclera quantile initialization
    b, g, r = cv2.split(img_segmented)
    value_r_minous_g_pixel_sclera = np.sum(r/10) - np.sum(g/10)

    std_sclera = cv2.meanStdDev(r-g)[1]

    quantile = np.empty((3, 3))
    filtered_bgr = img_segmented[:, :, 0]
    filtered_bgr = filtered_bgr[filtered_bgr != 0]

    # calculate all three quantiles
    quantile[0] = np.quantile(filtered_bgr, [0.25, 0.5, 0.75])

    filtered_bgr = img_segmented[:, :, 1]
    filtered_bgr = filtered_bgr[filtered_bgr != 0]

    quantile[1] = np.quantile(filtered_bgr, [0.25, 0.5, 0.75])

    filtered_bgr = img_segmented[:, :, 2]
    filtered_bgr = filtered_bgr[filtered_bgr != 0]

    quantile[2] = np.quantile(filtered_bgr, [0.25, 0.5, 0.75])
    sclera_quantiles_bgr.append(quantile)

    # count coloured pixels
    img_gray = cv2.cvtColor(sclera_vessels_masked, cv2.COLOR_BGR2GRAY)
    _, img_binaria = cv2.threshold(img_gray, 1, 255, cv2.THRESH_BINARY)
    number_vessels_pixel_coulored = cv2.countNonZero(img_binaria)

    # count total pixels
    number_vessels_pixel = img_binaria.size

    # calculate the difference between r and g channel
    b, g, r = cv2.split(sclera_vessels_masked)
    value_r_minous_g_pixel_vessels = np.sum(r/10) - np.sum(g/10)

    std_vessels = cv2.meanStdDev(r-g)[1]

    # sclera quantile initialization
    quantile = np.empty((3, 3))
    filtered_bgr = sclera_vessels_masked[:, :, 0]
    filtered_bgr = filtered_bgr[filtered_bgr != 0]

    # calculate all three quantiles
    quantile[0] = np.quantile(filtered_bgr, [0.25, 0.5, 0.75])

    filtered_bgr = sclera_vessels_masked[:, :, 1]
    filtered_bgr = filtered_bgr[filtered_bgr != 0]

    quantile[1] = np.quantile(filtered_bgr, [0.25, 0.5, 0.75])

    filtered_bgr = sclera_vessels_masked[:, :, 2]
    filtered_bgr = filtered_bgr[filtered_bgr != 0]

    quantile[2] = np.quantile(filtered_bgr, [0.25, 0.5, 0.75])
    vessels_quantiles_bgr.append(quantile)

    vessels_density.append(number_vessels_pixel_coulored/number_sclera_pixel_coulored)
    value_r_minous_g_img_sclera.append(value_r_minous_g_pixel_sclera/number_sclera_pixel)
    value_r_minous_g_img_vessels.append(value_r_minous_g_pixel_vessels/number_vessels_pixel)

    dev_std_sclera.append(std_sclera)
    dev_std_vessels.append(std_vessels)

    return sclera_quantiles_bgr, vessels_quantiles_bgr, vessels_density, value_r_minous_g_img_sclera, value_r_minous_g_img_vessels, dev_std_sclera, dev_std_vessels

def adjust_contrast(img_segmented, sclera_vessels_masked):
    sclera_lab = []
    sclera_vessels_lab = []

    sclera_es = cv2.cvtColor(img_segmented, cv2.COLOR_BGR2LAB)       # saving an image to compare with a converted one

    # format conversion --> contrast enhancement --> insertion in list
    img_lab = cv2.cvtColor(img_segmented, cv2.COLOR_BGR2LAB)
    img_lab[:, :, 1] = cv2.multiply(img_lab[:, :, 1], 1.2)
    sclera_lab.append(img_lab)

    # Comparison between image with and without contrast enhancement
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(sclera_es)
    plt.subplot(1, 2, 2)
    plt.imshow(sclera_lab[0])
    plt.show()

    # Convert vessels to Lab format
    vessel_es = cv2.cvtColor(sclera_vessels_masked, cv2.COLOR_BGR2LAB)       # saving an image to compare with a converted one

    # format conversion --> contrast enhancement --> insertion in list
    img_lab = cv2.cvtColor(sclera_vessels_masked, cv2.COLOR_BGR2LAB)
    img_lab[:, :, 1] = cv2.multiply(img_lab[:, :, 1], 1.2)
    sclera_vessels_lab.append(img_lab)

    # Comparison between image with and without contrast enhancement
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(vessel_es)
    plt.subplot(1, 2, 2)
    plt.imshow(sclera_vessels_lab[0])
    plt.show()

    return sclera_lab, sclera_vessels_lab

def calculate_lab(img_segmented, sclera_lab, sclera_vessels_lab):
    # Calculate the metrics
    number_sclera_pixel = 0
    number_vessels_pixel = 0
    number_sclera_pixel_coulored = 0
    number_vessels_pixel_coulored = 0

    L = 0
    a = 0
    b = 0

    value_a_pixel_sclera = 0
    value_a_pixel_vessels = 0


    number_sclera_pixel = 0
    number_vessels_pixel = 0
    value_a_pixel_sclera = 0
    value_a_pixel_vessels = 0

    # counts coloured pixels
    img_gray = cv2.cvtColor(img_segmented, cv2.COLOR_BGR2GRAY)
    _, img_binaria = cv2.threshold(img_gray, 1, 255, cv2.THRESH_BINARY)

    # counts total pixels
    number_sclera_pixel = img_binaria.size

    # Calculate the sum of each a* value
    L, a, b = cv2.split(sclera_lab[0])
    value_a_pixel_sclera = np.sum(a/10)

    std_sclera = cv2.meanStdDev(a)[1]

    # sclera quantile inizialization
    quantile = np.empty((3, 3))
    filtered_lab = sclera_lab[0][:, :, 0]
    filtered_lab = filtered_lab[filtered_lab != 0]

    # calculate all three quantiles
    quantile[0] = np.quantile(filtered_lab, [0.25, 0.5, 0.75])

    filtered_lab = sclera_lab[0][:, :, 1]
    filtered_lab = filtered_lab[filtered_lab != 154]

    quantile[1] = np.quantile(filtered_lab, [0.25, 0.5, 0.75])

    filtered_lab = sclera_lab[0][:, :, 2]
    filtered_lab = filtered_lab[filtered_lab != 128]

    quantile[2] = np.quantile(filtered_lab, [0.25, 0.5, 0.75])
    sclera_quantiles_lab.append(quantile)


    # count coloured pixels
    lab_to_bgr = cv2.cvtColor(sclera_vessels_lab[0], cv2.COLOR_LAB2BGR)
    img_gray = cv2.cvtColor(lab_to_bgr, cv2.COLOR_BGR2GRAY)
    _, img_binaria = cv2.threshold(img_gray, 1, 255, cv2.THRESH_BINARY)
    number_vessels_pixel_coulored = cv2.countNonZero(img_binaria)

    # count total pixels
    number_vessels_pixel = img_binaria.size

    # Calculate the sum of each a* value
    L, a, b = cv2.split(sclera_vessels_lab[0])
    value_a_pixel_vessels = np.sum(a/10)

    std_vessels_L = cv2.meanStdDev(L)[1]
    std_vessels_b = cv2.meanStdDev(b)[1]

    # sclera quantile inizialization
    quantile = np.empty((3, 3))
    filtered_lab = sclera_vessels_lab[0][:, :, 0]
    filtered_lab = filtered_lab[filtered_lab != 0]

    # calculate all three quantiles
    quantile[0] = np.quantile(filtered_lab, [0.25, 0.5, 0.75])

    filtered_lab = sclera_vessels_lab[0][:, :, 1]
    filtered_lab = filtered_lab[filtered_lab != 154]

    quantile[1] = np.quantile(filtered_lab, [0.25, 0.5, 0.75])

    filtered_lab = sclera_vessels_lab[0][:, :, 2]
    filtered_lab = filtered_lab[filtered_lab != 128]

    quantile[2] = np.quantile(filtered_lab, [0.25, 0.5, 0.75])
    vessels_quantiles_lab.append(quantile)


    vessels_density_lab = vessels_density.copy()
    value_a_img_sclera.append(value_a_pixel_sclera/number_sclera_pixel)
    value_a_img_vessels.append(value_a_pixel_vessels/number_vessels_pixel)

    dev_std_sclera_cielab.append(std_sclera)
    dev_std_vessels_cielab.append([std_vessels_L, std_vessels_b])

    return value_a_img_sclera, value_a_img_vessels, dev_std_sclera_cielab, dev_std_vessels_cielab, sclera_quantiles_lab, vessels_quantiles_lab

def extract_white(pred_image, img_segmented):
    white_only = cv2.bitwise_not(pred_image)

    # convert img_segmented to grey scale
    sclera_only = cv2.cvtColor(img_segmented, cv2.COLOR_BGR2GRAY)
    # apply thresholding
    _, sclera_only = cv2.threshold(sclera_only, 1, 255, cv2.THRESH_BINARY)

    maschera = cv2.bitwise_and(sclera_only, white_only)
    maschera =  cv2.resize(maschera, img_segmented.shape[:2][::-1])

    # Get the image of the sclera only
    vessels_colors_white = cv2.bitwise_and(img_segmented, img_segmented, mask=maschera)
    vessels_colors_white = cv2.cvtColor(vessels_colors_white, cv2.COLOR_BGR2LAB)

    # Calculate standard deviation of both L* and b* channel
    L, a, b = cv2.split(vessels_colors_white)
    std_vessels_color_white_L = cv2.meanStdDev(L)[1]
    std_vessels_color_white_b = cv2.meanStdDev(b)[1]

    vessels_colors_white_deviations_cielab.append([std_vessels_color_white_L, std_vessels_color_white_b])

    # Initialization of vessel quantiles
    quantile = np.empty((3, 3))
    filtered_lab = vessels_colors_white[:, :, 0]
    filtered_lab = filtered_lab[filtered_lab != 0]

    # calculate all three quantiles
    quantile[0] = np.quantile(filtered_lab, [0.25, 0.5, 0.75])

    filtered_lab = vessels_colors_white[:, :, 1]
    filtered_lab = filtered_lab[filtered_lab != 154]

    quantile[1] = np.quantile(filtered_lab, [0.25, 0.5, 0.75])

    filtered_lab = vessels_colors_white[:, :, 2]
    filtered_lab = filtered_lab[filtered_lab != 128]

    quantile[2] = np.quantile(filtered_lab, [0.25, 0.5, 0.75])

    vessels_colors_white_quantiles_cielab.append(quantile)


    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(vessels_colors_white)
    plt.show()

    return vessels_colors_white_deviations_cielab, vessels_colors_white_quantiles_cielab



