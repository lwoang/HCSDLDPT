import cv2
import numpy as np
import os
from ultralytics import YOLO

# Load YOLOv8 segmentation model (cần model seg, ví dụ: yolov8n-seg.pt)
model = YOLO("yolov8n-seg.pt")

INPUT_FOLDER = "./raw"
OUTPUT_FOLDER = "./processed"
TARGET_SIZE = (512, 288)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def apply_mask_and_white_bg(image, mask):
    # Resize mask về cùng kích thước ảnh nếu cần
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    mask = (mask > 0.5).astype(np.uint8) * 255  # nhị phân hóa

    # Tách đối tượng ra khỏi nền
    result = np.full_like(image, 255)  # nền trắng
    for c in range(3):
        result[:, :, c] = np.where(mask == 255, image[:, :, c], 255)
    return result, mask

def crop_to_mask(image, mask):
    ys, xs = np.where(mask == 255)
    if len(xs) == 0 or len(ys) == 0:
        return image  # fallback
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    return image[y1:y2, x1:x2]

def resize_with_padding(image, target_size=(512, 288)):
    h, w = image.shape[:2]
    scale = min(target_size[0]/w, target_size[1]/h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))

    pad_top = (target_size[1] - new_h) // 2
    pad_bottom = target_size[1] - new_h - pad_top
    pad_left = (target_size[0] - new_w) // 2
    pad_right = target_size[0] - new_w - pad_left

    padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right,
                                cv2.BORDER_CONSTANT, value=(255, 255, 255))
    return padded

def process_folder():
    for file in os.listdir(INPUT_FOLDER):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(INPUT_FOLDER, file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"❌ Lỗi đọc ảnh: {file}")
                continue

            results = model(img)[0]

            if results.masks is None or len(results.masks.data) == 0:
                print(f"⚠️ Không tìm thấy mask trong: {file}")
                continue

            mask = results.masks.data[0].cpu().numpy()
            img_white_bg, bin_mask = apply_mask_and_white_bg(img, mask)
            cropped = crop_to_mask(img_white_bg, bin_mask)
            final = resize_with_padding(cropped, TARGET_SIZE)

            out_path = os.path.join(OUTPUT_FOLDER, file)
            cv2.imwrite(out_path, final)
            print(f"✅ Xử lý xong: {file}")

if __name__ == "__main__":
    process_folder()
