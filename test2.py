import os
import cv2
import json
import numpy as np
import mysql.connector
from skimage.feature import graycomatrix, graycoprops, hog


# 1️⃣ Color Histogram (HSV, bins=8x8x8 = 512 chiều)
def extract_color_hist(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    return hist.flatten()

# 2️⃣ HOG (khoảng ~79.000 chiều)
def extract_hog(img_gray):
    return hog(img_gray,
               orientations=9,
               pixels_per_cell=(8, 8),
               cells_per_block=(2, 2),
               feature_vector=True)

# 3️⃣ GLCM (16 chiều)
def extract_glcm_features(img_gray):
    glcm = graycomatrix(img_gray,
                        distances=[1, 2],
                        angles=[0, np.pi/4],
                        symmetric=True,
                        normed=True)
    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    features = []
    for prop in props:
        features.extend(graycoprops(glcm, prop).flatten())
    return np.array(features[:16])  # lấy đúng 16 giá trị

# 🎯 Tổng hợp tất cả đặc trưng
def extract_all_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Không thể đọc ảnh: {image_path}")
    img_resized = cv2.resize(img, (512, 288))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    color = extract_color_hist(img_resized)
    hog_feat = extract_hog(gray)
    
     # 📊 Thống kê HOG
    print(f"HOG - {os.path.basename(image_path)} | mean: {hog_feat.mean():.4f}, std: {hog_feat.std():.4f}, nonzero: {(hog_feat != 0).sum()}")
    
    glcm_feat = extract_glcm_features(gray)

    return color, hog_feat, glcm_feat

# 📦 Kết nối MySQL
def connect_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="1234567",  # thay bằng mật khẩu bạn đặt
        database="animal_images_test"
    )

# 💾 Ghi dữ liệu vào CSDL
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

# 🔁 Xử lý hàng loạt ảnh
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

# 🚀 Chạy chính
if __name__ == "__main__":
    folder_path = "./Anh_resize_last1"
    process_image_folder(folder_path)
