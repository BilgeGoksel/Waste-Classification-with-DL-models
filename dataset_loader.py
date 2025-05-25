import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def load_dataset(data_dir, img_size=(224, 224), split_ratios=(0.7, 0.15, 0.15)):
    images = []
    labels = []
    class_names = sorted(os.listdir(data_dir))
    label_map = {name: idx for idx, name in enumerate(class_names)}

    print("Bulunan sınıflar:", class_names)

    for class_name in class_names:
        default_dir = os.path.join(data_dir, class_name, 'default')
        if not os.path.exists(default_dir):
            print(f"[!] Klasör yok: {default_dir}")
            continue

        files = os.listdir(default_dir)

        for file in files:
            if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            img_path = os.path.join(default_dir, file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"[X] Hatalı dosya: {img_path}")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, img_size)
            img = img.astype('float32') / 255.0
            images.append(img)
            labels.append(label_map[class_name])

    print(f"\nToplam {len(images)} görsel başarıyla yüklendi.")

    if len(images) == 0:
        raise ValueError("Hiç görsel yüklenemedi! Klasör yolunu veya dosyaları kontrol et.")

    images = np.array(images)
    labels = to_categorical(np.array(labels), num_classes=len(class_names))

    X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=(1 - split_ratios[0]), stratify=labels, random_state=42)
    val_ratio = split_ratios[1] / (split_ratios[1] + split_ratios[2])
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=(1 - val_ratio), stratify=y_temp, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test, class_names
