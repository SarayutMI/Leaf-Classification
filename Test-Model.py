import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# ---------- โหลดโมเดล ----------
model_path = "/content/drive/MyDrive/Phase2_AI_Leaf_Classification/Phase2_AI_Leaf_Classification/Model-Phase2/R50_Shape_final_V0.keras"
model = load_model(model_path)
print("✅ Model Loaded Successfully:", model_path)

# ---------- Class names ----------
all_class_names = [
    "Cordate",
    "Lanceolate",
    "Ovate",
    "Sagittate"
    # เพิ่ม class ของคุณให้ครบ
]

def predict_and_evaluate(folder_path, model, all_class_names):
    results = []
    correct = 0
    total = 0

    for filename in sorted(os.listdir(folder_path)):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        img_path = os.path.join(folder_path, filename)

        # ---------- โหลดและเตรียมภาพ ----------
        img = image.load_img(img_path, target_size=(256, 256))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # ---------- Predict ----------
        pred = model.predict(img_array, verbose=0)[0]
        top_index = np.argmax(pred)
        predicted_label = all_class_names[top_index]
        confidence = pred[top_index] * 100

        # ---------- ดึง Class จริงจากชื่อไฟล์ ----------
        # ตัวอย่าง:
        # Cordate_1.jpg     -> Cordate
        # Leaf_Cordate_1.jpg -> Leaf_Cordate
        base_name = os.path.splitext(filename)[0]
        true_class = base_name.rsplit("_", 1)[0]

        # ---------- ตรวจสอบผลลัพธ์ ----------
        is_correct = (predicted_label == true_class)

        if is_correct:
            correct += 1
        total += 1

        print(
            f"🖼️ {filename} "
            f"→ Predict: {predicted_label} ({confidence:.2f}%) "
            f"| True: {true_class} "
            f"| {'✅ ถูก' if is_correct else '❌ ผิด'}"
        )

        results.append({
            "file": filename,
            "predicted_label": predicted_label,
            "true_class": true_class,
            "confidence": confidence,
            "correct": is_correct
        })

    # ---------- Accuracy ----------
    accuracy = (correct / total * 100) if total > 0 else 0

    print("\n" + "=" * 70)
    print(f"🎯 Accuracy = {accuracy:.2f}% ({correct}/{total})")
    print("=" * 70)

    return results, accuracy


# ---------- รัน ----------
folder_path = "/content/drive/MyDrive/Phase2_AI_Leaf_Classification/Phase2_AI_Leaf_Classification/Test_Set_2_Shadow/Testset-Shape-Shadow-output"

results, accuracy = predict_and_evaluate(
    folder_path,
    model,
    all_class_names
)