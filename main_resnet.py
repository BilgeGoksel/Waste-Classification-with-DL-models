from dataset_loader_resnet import load_dataset_resnet
from transfer_model import create_transfer_model

data_path = r"C:\Users\HP\Desktop\DERSLER\CNN_Proje\dataset\images\images"

X_train, X_val, X_test, y_train, y_val, y_test, class_names = load_dataset_resnet(data_path)

model = create_transfer_model("ResNet50", input_shape=(224, 224, 3), num_classes=len(class_names))

model.fit(X_train, y_train,
          validation_data=(X_val, y_val),
          epochs=20,
          batch_size=32)
