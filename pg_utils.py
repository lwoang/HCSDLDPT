import psycopg2
import psycopg2.extras
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

def safe_json_load(x):
    if isinstance(x, str):
        return json.loads(x)
    return x  # nếu đã là list hoặc dict thì trả nguyên

# Kết nối và đọc tất cả đặc trưng từ bảng image_features4
def fetch_all_features():
    conn = psycopg2.connect(
        host="localhost",
        user="postgres",
        password="1234567", 
        dbname="BTL_ĐPT"
    )
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cursor.execute("SELECT name, image_path, color_hist, hog, glcm FROM image_features4")

    data = []
    for row in cursor.fetchall():
        data.append({
        "name": row["name"],
        "path": row["image_path"],
        "color": np.array(safe_json_load(row["color_hist"])),
        "hog": np.array(safe_json_load(row["hog"])),
        "glcm": np.array(safe_json_load(row["glcm"]))
        })

    cursor.close()
    conn.close()
    return data

# Chuẩn hóa vector đặc trưng trước khi so sánh
def normalize_vector(input_vec, db_vecs):
    stacked = np.vstack([input_vec] + db_vecs)
    scaled = StandardScaler().fit_transform(stacked)
    return scaled[0], scaled[1:]

# Trả về Top K ảnh giống nhất
def find_top_k(query_feats, db_data, k=5, w_color=1.0, w_hog=1.0, w_glcm=1.0):
    # Tách từng đặc trưng
    c1, h1, g1 = query_feats

    # Áp dụng trọng số trực tiếp vào vector
    query_vector = np.concatenate([c1 * w_color, h1 * w_hog, g1 * w_glcm])

    db_vectors = []
    meta = []  # Lưu thông tin name/path

    for item in db_data:
        vec = np.concatenate([
            item["color"] * w_color,
            item["hog"] * w_hog,
            item["glcm"] * w_glcm
        ])
        db_vectors.append(vec)
        meta.append((item["name"], item["path"]))

    # Chuẩn hóa toàn bộ (truy vấn + DB)
    all_vectors = np.vstack([query_vector] + db_vectors)
    scaled = StandardScaler().fit_transform(all_vectors)
    query_scaled = scaled[0]
    db_scaled = scaled[1:]

    # Tính cosine similarity
    sims = cosine_similarity([query_scaled], db_scaled)[0]  # shape: (n_db,)
    
    # Sắp xếp giảm dần theo similarity
    ranked = sorted(zip(meta, sims), key=lambda x: x[1], reverse=True)
    
    # Trả về top k
    return [(name, path, score) for ((name, path), score) in ranked[:k]]
