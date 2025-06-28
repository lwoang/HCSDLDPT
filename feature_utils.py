import cv2
import numpy as np
from skimage.feature import hog
from skimage.feature import graycomatrix, graycoprops
from ultralytics import YOLO
from resize3 import apply_mask_and_white_bg, crop_to_mask
import streamlit as st 

target_size = (512, 288)
TARGET_SIZE = (512, 288)  # width x height

@st.cache_resource
def load_model():
    from ultralytics import YOLO
    return YOLO('yolov8n-seg.pt')

def extract_color_hist(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8],
                        [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist  # shape (512,)

def extract_hog(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        transform_sqrt=True,
        feature_vector=True
    )
    return features  # shape khoảng 79000

def extract_glcm_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 64))  # chuẩn hóa kích thước
    distances = [1, 2]
    angles = [0, np.pi/4]
    glcm = graycomatrix(gray, distances=distances, angles=angles,
                        levels=256, symmetric=True, normed=True)

    features = []
    props = ['contrast', 'correlation', 'energy', 'homogeneity']
    for prop in props:
        vals = graycoprops(glcm, prop)
        features.extend(vals.flatten())  # mỗi prop: 2 distances × 2 angles = 4
    return np.array(features)  # shape (16,)


#model = YOLO('yolov8n-seg.pt')  # Đảm bảo model YOLO đã được tải

# def preprocess_image_for_query(image_path):
#     img = cv2.imread(image_path)
#     if img is None:
#         raise ValueError("❌ Không thể đọc ảnh.")

#     results = model(img)[0]

#     if results.masks is None or len(results.masks.data) == 0:
#         print("⚠️ Không tìm thấy mask – dùng ảnh gốc resize thường.")
#         return resize_with_padding(img, TARGET_SIZE)

#     # Dùng mask đầu tiên
#     mask = results.masks.data[0].cpu().numpy()
#     mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
#     mask = (mask > 0.5).astype(np.uint8) * 255

#     # Tách vật thể – nền trắng
#     white_bg = np.full_like(img, 255)
#     for c in range(3):
#         white_bg[:, :, c] = np.where(mask == 255, img[:, :, c], 255)

#     # Cắt sát vật thể
#     ys, xs = np.where(mask == 255)
#     if len(xs) == 0 or len(ys) == 0:
#         return resize_with_padding(white_bg, TARGET_SIZE)
#     x1, x2 = xs.min(), xs.max()
#     y1, y2 = ys.min(), ys.max()
#     cropped = white_bg[y1:y2, x1:x2]

#     return resize_with_padding(cropped, TARGET_SIZE)

def resize_with_padding(image, target_size=(512, 288)):
    h, w = image.shape[:2]
    scale = min(target_size[0]/w, target_size[1]/h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))

    pad_top = (target_size[1] - new_h) // 2
    pad_bottom = target_size[1] - new_h - pad_top
    pad_left = (target_size[0] - new_w) // 2
    pad_right = target_size[0] - new_w - pad_left

    return cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right,
                              cv2.BORDER_CONSTANT, value=(255, 255, 255))
    
    
    
def preprocess_image_for_query(image_path):
    model = load_model()
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("❌ Không thể đọc ảnh.")

    results = model(img)[0]

    if results.masks is None or len(results.masks.data) == 0:
        print("⚠️ Không tìm thấy mask – dùng ảnh gốc resize thường.")
        return resize_with_padding(img, TARGET_SIZE)

    mask = results.masks.data[0].cpu().numpy()
    img_white_bg, bin_mask = apply_mask_and_white_bg(img, mask)
    cropped = crop_to_mask(img_white_bg, bin_mask)
    final = resize_with_padding(cropped, TARGET_SIZE)

    return final
