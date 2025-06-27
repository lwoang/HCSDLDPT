import streamlit as st
import os
import cv2
from feature_utils import extract_color_hist, extract_hog, extract_glcm_features
from mysql_utils import fetch_all_features, find_top_k

st.set_page_config(layout="wide")
st.title("🔍 Tìm kiếm ảnh động vật tương đồng")

uploaded_file = st.file_uploader("📤 Chọn một ảnh động vật", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Lưu ảnh tạm
    input_path = "input.jpg"
    with open(input_path, "wb") as f:
        f.write(uploaded_file.read())

    st.image(input_path, caption="📷 Ảnh gốc", use_container_width=True)

    st.markdown("### 2️⃣ Tìm ảnh giống nhất từ database")

    try:
        # Đọc ảnh và resize về đúng kích thước dùng để huấn luyện trước đó 
        img = cv2.imread(input_path)
        if img is None:
            raise ValueError("Không thể đọc ảnh.")

        resized_img = cv2.resize(img, (512, 288))  #size giống DB đã dùng
        gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

        # Trích xuất đặc trưng
        color = extract_color_hist(resized_img)
        hog = extract_hog(gray)
        glcm = extract_glcm_features(gray)

        # Truy vấn DB
        db_data = fetch_all_features()
        top_matches = find_top_k((color, hog, glcm), db_data, k=3)

        st.markdown("### 🔝 Top 3 ảnh giống nhất:")
        for name, path, score in top_matches:
            col1, col2 = st.columns([1.2, 3])
            with col1:
                st.image(path, width=240)
            with col2:
                st.markdown(f"**📌 {name}**")
                st.markdown(f"🧠 Similarity: `{score:.4f}`")
                st.markdown("---")
    except Exception as e:
        st.error(f"❌ Lỗi truy vấn database: {e}")
