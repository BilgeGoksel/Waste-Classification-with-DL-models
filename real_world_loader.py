
import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical

def load_real_world_test_set(data_dir, img_size=(224, 224)):
    images = []
    labels = []
    class_names = sorted(os.listdir(data_dir))
    label_map = {name: idx for idx, name in enumerate(class_names)}

    for class_name in class_names:
        real_dir = os.path.join(data_dir, class_name, 'real_world')
        if not os.path.exists(real_dir):
            continue
        for file in os.listdir(real_dir):
            if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            img_path = os.path.join(real_dir, file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, img_size)
            img = img.astype('float32') / 255.0
            images.append(img)
            labels.append(label_map[class_name])

    images = np.array(images)
    labels = to_categorical(np.array(labels), num_classes=len(class_names))

    return images, labels, class_names
