from dataset_loader import load_dataset
from cnn_model import create_cnn_model
import tensorflow as tf

data_path = r"C:\Users\HP\Desktop\DERSLER\CNN_Proje\dataset\images\images"

# Veriyi yükle (sadece default klasörlerinden)
X_train, X_val, X_test, y_train, y_val, y_test, class_names = load_dataset(data_path)

# Modeli oluştur
input_shape = (224, 224, 3)
num_classes = len(class_names)
model = create_cnn_model(input_shape, num_classes)

# Modeli eğit
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=32
)

# Test seti ile değerlendirme
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Default test seti üzerindeki doğruluk: {test_acc * 100:.2f}%")

# Modeli kaydet
model.save("cnn_default_model.keras")
print("Model kaydedildi: cnn_default_model.keras")

#########################################################
# real_world klasörü için eğitim 

from real_world_loader import load_real_world_test_set
from tensorflow.keras.models import load_model

data_path = r"C:\Users\HP\Desktop\DERSLER\CNN_Proje\dataset\images\images"

model = load_model("cnn_default_model.keras")
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Real world test verilerini yükle
X_real, y_real, _ = load_real_world_test_set(data_path)

real_loss, real_acc = model.evaluate(X_real, y_real)
print(f"Real world test seti üzerindeki doğruluk: {real_acc * 100:.2f}%")

############################################################
# Data Augmentation

from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.05,
    horizontal_flip=True,
    fill_mode='nearest'
)

datagen.fit(X_train)


# CNN modelini iyleştirmeye yönelik CNN üzerine uygulanılacak 
from cnn_model import create_cnn_model

model = create_cnn_model(input_shape=(224, 224, 3), num_classes=len(class_names))

model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    steps_per_epoch=len(X_train) // 32,
    validation_data=(X_val, y_val),
    epochs=20
)

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Data augmentation sonrası default test seti doğruluğu (CNN): {test_acc * 100:.2f}%")

model.save("cnn_augmented_model.keras")





