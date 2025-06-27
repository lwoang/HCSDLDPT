import cv2
import numpy as np
from skimage.feature import hog
from skimage.feature import graycomatrix, graycoprops

def extract_color_hist(img):
    hist = []
    for i in range(3):  # RGB
        h = cv2.calcHist([img], [i], None, [8], [0, 256])
        hist.extend(h.flatten())
    return np.array(hist)

def extract_hog(img_gray):
    return hog(img_gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)

def extract_glcm_features(img_gray):
    glcm = graycomatrix(img_gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], symmetric=True, normed=True)
    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    features = []
    for prop in props:
        features.extend(graycoprops(glcm, prop).flatten())
    return np.array(features)
