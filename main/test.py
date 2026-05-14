import os
import shutil

# =========================
# PATH
# =========================
SRC_ROOT = r"C:\Users\phand\Desktop\new\pointcloud_capture1"
DST_ROOT = r"C:\Users\phand\Desktop\new\pointcloud_capture1_new"

DST_PLY_ROOT = os.path.join(DST_ROOT, "PLY_in_BB")
os.makedirs(DST_PLY_ROOT, exist_ok=True)

# =========================
# RESTRUCTURE
# =========================
for apple_id in range(275, 311):  # 241 -> 274
    apple_name = f"apple_{apple_id}"
    
    src_dir = os.path.join(SRC_ROOT, apple_name, "PLY_in_BB")
    dst_dir = os.path.join(DST_PLY_ROOT, apple_name)
    
    if not os.path.exists(src_dir):
        print(f"❌ Missing: {src_dir}")
        continue
    
    os.makedirs(dst_dir, exist_ok=True)
    
    for file in os.listdir(src_dir):
        if file.endswith(".ply"):
            src_file = os.path.join(src_dir, file)
            dst_file = os.path.join(dst_dir, file)
            
            shutil.copy2(src_file, dst_file)
    
    print(f"✅ Done {apple_name}")