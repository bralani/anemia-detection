# import libraries
import sklearn
import pandas as pd
import numpy as np
import os
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf

patch_size=48        # patch image size

def normal_normalized_single(imgs,mask):
    imgs_normalized = np.empty(imgs.shape)
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_normalized = (imgs-imgs_mean)/imgs_std
    for i in range(imgs.shape[2]):
        imgs_normalized[:,:,i] = ((imgs_normalized[:,:,i] - np.min(imgs_normalized[:,:,i])) / (np.max(imgs_normalized[:,:,i])-np.min(imgs_normalized[:,:,i])))*255
    return imgs_normalized


# CLAHE (Contrast Limited Adaptive Histogram Equalization)
#adaptive histogram equalization is used. In this, image is divided into small blocks called "tiles"
# (tileSize is 8x8 by default in OpenCV). Then each of these blocks are histogram equalized as usual.
#  So in a small area, histogram would confine to a small region (unless there is noise). 
# If noise is there, it will be amplified. To avoid this, contrast limiting is applied. 
# If any histogram bin is above the specified contrast limit (by default 40 in OpenCV),
#  those pixels are clipped and distributed uniformly to other bins before applying histogram equalization.
#  After equalization, to remove artifacts in tile borders, bilinear interpolation is applied
def clahe_equalized_single(imgs):
  clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
  imgs_equalized = np.empty(imgs.shape)
  for i in range(imgs.shape[2]):
    imgs_equalized[:,:,i] = clahe.apply(np.array(imgs[:,:,i], dtype = np.uint8))
  return imgs_equalized


def adjust_gamma_single(imgs, gamma=1.0):
  invGamma = 1.0 / gamma
  table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
  # apply gamma correction using the lookup table
  new_imgs = np.empty(imgs.shape)
  for i in range(imgs.shape[2]):
    new_imgs[:,:,i] = cv2.LUT(np.array(imgs[:,:,i], dtype = np.uint8), table)
  return new_imgs

def preprocess_single(image,mask):
  
  assert np.max(mask)==1
  image=np.array(image)
  image[:,:,0]=image[:,:,0]*mask
  image[:,:,1]=image[:,:,1]*mask
  image[:,:,2]=image[:,:,2]*mask

  image=normal_normalized_single(image,mask)
  image=clahe_equalized_single(image)
  image=adjust_gamma_single(image,0.8)
  image=image/255.0
  return image


def load_test_data(image):
  #image=tf.image.decode_jpeg(image,channels=1)
  #print(image.shape)
  image=tf.image.resize(image,[patch_size,patch_size])
  #image/=255.0
  return image

# pad images
def padding_images(image,mask,stride):
    h,w=image.shape[:2]
    new_h,new_w=h,w
    while (new_h-patch_size)%stride!=0:
        new_h+=1
    while (new_w-patch_size)%stride!=0:
        new_w+=1
    pad_image=np.zeros((new_h,new_w,3))
    pad_image[:h,:w,:]=image
    
    pad_mask=np.zeros((new_h,new_w))
    pad_mask[:h,:w]=mask
    
    return pad_image,pad_mask

# images to patches
def img2patch_list(image,stride=patch_size):
    patch_list=[]
    #image_binary=0.8*image[:,:,1:2]+0.2*image[:,:,2:3]  
    for j in range(0,image.shape[1]-patch_size+1,stride):
        for i in range(0,image.shape[0]-patch_size+1,stride):
            patch=image[i:i+patch_size,j:j+patch_size,:]
            patch_list.append(patch)
    return patch_list

# patches to image
def patchlist2image(patch_list,stride,image_shape):
    result=np.zeros(image_shape[:2])
    sum_matrix=np.zeros(image_shape[:2])
    index_x,index_y=0,0
    for i in range(patch_list.shape[0]):
        patch=patch_list[i,:,:,0]
        #patch=np.where(patch>0.5,1,0)
        #print(patch)
        result[index_x:index_x+patch_size,index_y:index_y+patch_size]+=patch
        sum_matrix[index_x:index_x+patch_size,index_y:index_y+patch_size]+=1
        index_x+=stride
        if index_x+patch_size>image_shape[0]:
            index_x=0
            index_y+=stride
    return result/sum_matrix

def get_vessels_from_image(image):
    model = "pretrained_model"
    absolute_path = os.path.join(os.getcwd() + "/unet/", model)
    model_unet= tf.keras.models.load_model(absolute_path)
    stride=5
        
    # load and process test images
    original_shape=image.shape

    # generate mask
    mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask=np.where(mask>0,1,0)

    image = image[:, :, :3]  # Rimuovi il canale alfa
    # image to patches
    image,pad_mask=padding_images(image,mask,stride)

    image=preprocess_single(image,pad_mask)
    test_patch_list=img2patch_list(image,stride)

    # test dataloader
    test_dataset=tf.data.Dataset.from_tensor_slices(test_patch_list)
    test_dataset=test_dataset.map(load_test_data)
    test_dataset=test_dataset.batch(64)
    pred_result=[]

    # test process
    for batch, patch in enumerate(test_dataset):
        _,pred=model_unet(patch,training=False)
        
        pred=pred.numpy()
        pred_result.append(pred)
    pred_result=np.concatenate(pred_result,axis=0)

    pred_image=patchlist2image(pred_result,stride,image.shape)

    pred_image=pred_image[:original_shape[0],:original_shape[1]]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))

    mask = cv2.erode(mask.astype(np.uint8), kernel)

    pred_image=pred_image*mask
    pred_image=np.where(pred_image>0.5,1,0)
        
    return pred_image