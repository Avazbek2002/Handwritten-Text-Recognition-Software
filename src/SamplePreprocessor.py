# function converts the img data into normalized form
import cv2
import numpy as np
import tensorflow as tf

# function reads an image. Normalizes the values between [-1, 1] and returns an array (128, 32, 1)
def preprocess(img, imgSize):
    "put img into target img of size imgSize, transpose for TF and normalize gray-values"

    # there are damaged files in IAM dataset - just use black image instead
    if img is None:
        img = np.zeros(imgSize[::-1])

    # data augmentation
    img = img.astype(np.float)

    # center image
    wt, ht = imgSize
    h, w = img.shape
    f = min(wt / w, ht / h)
    tx = (wt - w * f) / 2
    ty = (ht - h * f) / 2

    # map image into target image
    M = np.float32([[f, 0, tx], [0, f, ty]])
    target = np.ones(imgSize[::-1]) * 255 / 2
    img = cv2.warpAffine(img , M, dsize=imgSize, dst=target, borderMode=cv2.BORDER_TRANSPARENT)

    # transpose for TF
    img = cv2.transpose(img)

    # convert to range [-1, 1]
    img = img / 255 - 0.5
    img = tf.expand_dims(img, 2)
    return img