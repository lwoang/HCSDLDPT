import mysql.connector
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

def fetch_all_features():
    conn = mysql.connector.connect(host="localhost", user="root", password="1234567", database="animal_images_test")
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT name, image_path, color_hist, hog, glcm FROM image_features4")

    data = []
    for row in cursor.fetchall():
        data.append({
            "name": row["name"],
            "path": row["image_path"],
            "color": np.array(json.loads(row["color_hist"])),
            "hog": np.array(json.loads(row["hog"])),
            "glcm": np.array(json.loads(row["glcm"]))
        })

    cursor.close()
    conn.close()
    return data

def normalize_vector(input_vec, db_vecs):
    stacked = np.vstack([input_vec] + db_vecs)
    scaled = StandardScaler().fit_transform(stacked)
    return scaled[0], scaled[1:]

def combined_similarity(color1, color2, hog1, hog2, glcm1, glcm2, w_color=0.3, w_hog=0.5, w_glcm=0.2):
    sim_color = cosine_similarity([color1], [color2])[0][0]
    sim_hog = cosine_similarity([hog1], [hog2])[0][0]
    sim_glcm = cosine_similarity([glcm1], [glcm2])[0][0]
    return w_color*sim_color + w_hog*sim_hog + w_glcm*sim_glcm

def find_top_k(query_feats, db_data, k=5):
    c1, h1, g1 = query_feats
    c1, c_db = normalize_vector(c1, [d['color'] for d in db_data])
    h1, h_db = normalize_vector(h1, [d['hog'] for d in db_data])
    g1, g_db = normalize_vector(g1, [d['glcm'] for d in db_data])

    results = []
    for i, d in enumerate(db_data):
        score = combined_similarity(c1, c_db[i], h1, h_db[i], g1, g_db[i])
        results.append((d['name'], d['path'], score))
    return sorted(results, key=lambda x: x[2], reverse=True)[:k]
