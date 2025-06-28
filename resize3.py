from ultralytics import YOLO
import cv2
import os

# Khởi tạo mô hình YOLOv8 (detection)
model = YOLO("yolov8n.pt")  # hoặc yolov8s.pt nếu cần chính xác hơn

input_folder = "./Anh"
output_folder = "Anh_resize_last1"
os.makedirs(output_folder, exist_ok=True)

TARGET_SIZE = (512, 288)  # width, height

# Danh sách class ID trong COCO dataset là động vật 4 chân
# Chó (16), mèo (15), ngựa (17), cừu (18), bò (19), voi (20), gấu (21), hươu cao cổ (23), ngựa vằn (24)
ANIMAL_CLASS_IDS = [15, 16, 17, 18, 19, 20, 21, 23, 24]

for file in os.listdir(input_folder):
    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(input_folder, file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Lỗi đọc ảnh: {file}")
            continue

        # Detect objects
        results = model(img)[0]
        animal_boxes = []

        for box in results.boxes:
            cls_id = int(box.cls)
            if cls_id in ANIMAL_CLASS_IDS:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                animal_boxes.append((x1, y1, x2, y2))

        if len(animal_boxes) == 0:
            print(f"Không phát hiện con vật trong ảnh: {file}")
            continue

        # Lấy box lớn nhất (trường hợp có nhiều con vật)
        x1, y1, x2, y2 = sorted(animal_boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)[0]
        cropped = img[y1:y2, x1:x2]

        # Resize với padding để không méo hình
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

        # Lưu ảnh
        output_path = os.path.join(output_folder, file)
        cv2.imwrite(output_path, final)

print(f"✅ Hoàn tất. Ảnh đã lưu tại: {output_folder}")
