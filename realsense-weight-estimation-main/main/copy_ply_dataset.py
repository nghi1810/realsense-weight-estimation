import os
import shutil

# Folder gốc
source_root = r"C:\Users\phand\Desktop\new\pointcloud_capture1"

# Folder đích mới
destination_root = r"C:\Users\phand\Desktop\PLY_in_BB"

# Tạo folder đích nếu chưa có
os.makedirs(destination_root, exist_ok=True)

# Duyệt từ apple_311 đến apple_351
for i in range(311, 352):

    apple_folder = f"apple_{i}"

    # Folder chứa file .ply trong source
    source_ply_folder = os.path.join(
        source_root,
        apple_folder,
        "PLY_in_BB"
    )

    # Folder đích mới
    destination_ply_folder = os.path.join(
        destination_root,
        apple_folder
    )

    # Kiểm tra folder nguồn có tồn tại không
    if not os.path.exists(source_ply_folder):
        print(f"Không tìm thấy: {source_ply_folder}")
        continue

    # Tạo folder đích
    os.makedirs(destination_ply_folder, exist_ok=True)

    # Copy toàn bộ file .ply
    for file_name in os.listdir(source_ply_folder):

        if file_name.lower().endswith(".ply"):

            source_file = os.path.join(source_ply_folder, file_name)
            destination_file = os.path.join(destination_ply_folder, file_name)

            shutil.copy2(source_file, destination_file)

    print(f"Đã copy xong {apple_folder}")

print("Hoàn tất copy toàn bộ file PLY.")