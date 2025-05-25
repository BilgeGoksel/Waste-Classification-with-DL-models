
from dataset_loader import load_dataset
from transfer_model import create_transfer_model
from tensorflow.keras.models import load_model

# Veri yolu
data_path = r"C:\Users\HP\Desktop\DERSLER\CNN_Proje\dataset\images\images"

# Veriyi yükle
X_train, X_val, X_test, y_train, y_val, y_test, class_names = load_dataset(data_path)

# MobileNetV2 modeli oluştur
input_shape = (224, 224, 3)
num_classes = len(class_names)
model = create_transfer_model("MobileNetV2", input_shape, num_classes)

# Modeli eğit
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=32
)

# Default test seti ile değerlendir
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Default test seti üzerindeki doğruluk (MobileNetV2): {test_acc * 100:.2f}%")

# Modeli kaydet
model.save("mobilenetv2_model.h5")
print("Model kaydedildi: mobilenetv2_model.h5")

from real_world_loader import load_real_world_test_set

# Eğitilmiş mobilenetv2 modelini yükle
model = load_model("mobilenetv2_model.h5")

# Modeli derle (eğer uyarı alırsan)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Real world verisini yükle
X_real, y_real, _ = load_real_world_test_set(data_path)

# Değerlendir
real_loss, real_acc = model.evaluate(X_real, y_real)
print(f"Real world test seti üzerindeki doğruluk (MobileNetV2): {real_acc * 100:.2f}%")
