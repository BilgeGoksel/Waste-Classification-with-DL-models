from dataset_loader_resnet import load_dataset_resnet
from transfer_model import create_transfer_model
from tensorflow.keras.models import load_model

data_path = r"C:\Users\HP\Desktop\DERSLER\CNN_Proje\dataset\images\images"

# EfficientNetB0 için veri yükle
X_train, X_val, X_test, y_train, y_val, y_test, class_names = load_dataset_resnet(
    data_path, model_type='efficientnet')

# Modeli oluştur
model = create_transfer_model("EfficientNetB0", input_shape=(224, 224, 3), num_classes=len(class_names))

# Eğit
model.fit(X_train, y_train,
          validation_data=(X_val, y_val),
          epochs=10,
          batch_size=32)

# Test et
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Default test seti üzerindeki doğruluk (EfficientNetB0): {test_acc * 100:.2f}%")

# Modeli kaydet
model.save("efficientnet_model.h5")
