import os
import random
import shutil

# =========================
# Path ของ Dataset
# =========================
train_root = "/Volumes/SSD_M/LeafClassification_A7/Phase2 AI Leaf Classification/Train-AI-Leaf-Base"
test_root = "/Volumes/SSD_M/LeafClassification_A7/Phase2 AI Leaf Classification/Test-AI-Leaf-Base"

# จำนวนรูปที่ต้องการย้ายต่อโฟลเดอร์
NUM_IMAGES = 40

# กำหนด seed เพื่อให้สุ่มได้ผลเหมือนเดิมทุกครั้ง
random.seed(42)

# =========================
# วนทุกโฟลเดอร์ใน Train
# =========================
for folder_name in os.listdir(train_root):

    train_dir = os.path.join(train_root, folder_name)

    # ข้ามถ้าไม่ใช่โฟลเดอร์
    if not os.path.isdir(train_dir):
        continue

    # สร้าง Path ปลายทางตามชื่อโฟลเดอร์เดิม
    test_dir = os.path.join(test_root, folder_name)
    os.makedirs(test_dir, exist_ok=True)

    # อ่านเฉพาะไฟล์รูป
    images = [
        f for f in os.listdir(train_dir)
        if not f.startswith("._")
        and f.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        )
    ]

    # ถ้ามีน้อยกว่า 40 รูป ให้ย้ายทั้งหมด
    n = min(NUM_IMAGES, len(images))

    # สุ่มเลือกรูป
    selected_images = random.sample(images, n)

    print(f"\nFolder: {folder_name}")
    print(f"Moving {n} images...")

    # ย้ายรูป
    for img_name in selected_images:
        src = os.path.join(train_dir, img_name)
        dst = os.path.join(test_dir, img_name)

        shutil.move(src, dst)

        # แสดงว่ารูปไหนถูกย้ายไปไหน
        print(f"{src}  -->  {dst}")

    print(f"Done : {folder_name}")

print("\n===== Finished =====")