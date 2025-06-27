import os
import cv2
import json
import numpy as np
import mysql.connector
from skimage.feature import hog, graycomatrix, graycoprops

# ----------- 1. Trích xuất đặc trưng ảnh -----------
def extract_color_hist(img):
    hist = []
    for i in range(3):  # RGB
        h = cv2.calcHist([img], [i], None, [8], [0, 256])
        hist.extend(h.flatten())
    return np.array(hist)

def extract_hog(img_gray):
    return hog(img_gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)

def extract_glcm_features(img_gray):
    glcm = graycomatrix(img_gray, distances=[1],
                        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        symmetric=True, normed=True)
    features = []
    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    for prop in props:
        f = graycoprops(glcm, prop)
        features.extend(f.flatten())
    return np.array(features)

def extract_all_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Không thể đọc ảnh: {image_path}")
    img_resized = cv2.resize(img, (512, 288))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    color = extract_color_hist(img_resized)
    hog_feat = extract_hog(gray)
    glcm_feat = extract_glcm_features(gray)

    return color, hog_feat, glcm_feat

# ----------- 2. Kết nối CSDL và lưu dữ liệu -----------
def connect_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="12345678",  # sửa lại
        database="animal_images_test"
    )

def insert_features_to_db(name, path, color, hog_feat, glcm_feat):
    conn = connect_db()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO image_features4 (name, image_path, color_hist, hog, glcm)
            VALUES (%s, %s, %s, %s, %s)
        """, (
            name,
            path,
            json.dumps(color.tolist()),
            json.dumps(hog_feat.tolist()),
            json.dumps(glcm_feat.tolist())
        ))
        conn.commit()
    except mysql.connector.Error as err:
        print(f"❌ Lỗi khi lưu ảnh {name}: {err}")
    finally:
        cursor.close()
        conn.close()

# ----------- 3. Xử lý hàng loạt ảnh từ thư mục -----------
def process_image_folder(folder_path):
    supported_exts = ('.jpg', '.jpeg', '.png')
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(supported_exts):
            full_path = os.path.join(folder_path, filename)
            try:
                print(f"📷 Đang xử lý: {filename}")
                color, hog_feat, glcm_feat = extract_all_features(full_path)
                insert_features_to_db(filename, full_path, color, hog_feat, glcm_feat)
                print("✅ Đã lưu vào CSDL\n")
            except Exception as e:
                print(f"❌ Bỏ qua ảnh {filename}: {e}\n")

# ----------- 4. Chạy chính -----------
if __name__ == "__main__":
    folder_path = "./Anh_resize_last1"  # chỉnh lại đường dẫn
    process_image_folder(folder_path)
