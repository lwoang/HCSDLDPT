import os

def rename_images_in_directory(directory_path, prefix="image", extension_filter=None):
    """
    Đổi tên tất cả các tệp ảnh trong thư mục theo dạng prefix + số thứ tự.

    Parameters:
        - directory_path (str): Đường dẫn đến thư mục chứa ảnh.
        - prefix (str): Tiền tố cho tên mới (mặc định là 'image').
        - extension_filter (list[str] | None): Chỉ đổi tên các file có phần mở rộng nhất định (ví dụ ['.jpg', '.png']).
    """
    if not os.path.isdir(directory_path):
        print(f"Thư mục không tồn tại: {directory_path}")
        return

    files = os.listdir(directory_path)
    files.sort()  # Sắp xếp để tên có thứ tự cố định

    count = 1
    for filename in files:
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            name, ext = os.path.splitext(filename)
            if extension_filter is None or ext.lower() in extension_filter:
                new_name = f"{prefix}_{count}{ext}"
                new_path = os.path.join(directory_path, new_name)
                os.rename(file_path, new_path)
                print(f"Renamed: {filename} -> {new_name}")
                count += 1

    print("Đã đổi tên xong toàn bộ ảnh.")

# Ví dụ sử dụng
rename_images_in_directory("./Anh_resize_last1", prefix="photo", extension_filter=[".jpg", ".png", ".jpeg"])
