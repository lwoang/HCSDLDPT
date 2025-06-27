import cv2
import numpy as np
from skimage.feature import hog
from skimage.feature import graycomatrix, graycoprops
from ultralytics import YOLO


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



import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # Đảm bảo model YOLO đã được tải

def preprocess_image_for_query(image_path):
    TARGET_SIZE = (512, 288)  # width, height
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("❌ Không thể đọc ảnh.")

    results = model(img)[0]
    if len(results.boxes) == 0:
        print("⚠️ Không phát hiện vật thể. Dùng lại ảnh gốc resize bình thường.")
        return cv2.resize(img, TARGET_SIZE)

    # Dùng bounding box đầu tiên (ảnh chỉ có 1 con vật)
    x1, y1, x2, y2 = map(int, results.boxes.xyxy[0])
    cropped = img[y1:y2, x1:x2]

    # Resize có padding trắng giống resize3.py
    h, w = cropped.shape[:2]
    scale = min(TARGET_SIZE[0] / w, TARGET_SIZE[1] / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(cropped, (new_w, new_h))

    pad_top = (TARGET_SIZE[1] - new_h) // 2
    pad_bottom = TARGET_SIZE[1] - new_h - pad_top
    pad_left = (TARGET_SIZE[0] - new_w) // 2
    pad_right = TARGET_SIZE[0] - new_w - pad_left

    final = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right,
                               cv2.BORDER_CONSTANT, value=(255, 255, 255))

    return final