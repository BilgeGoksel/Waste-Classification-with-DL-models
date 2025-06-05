import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input, Model
from tensorflow.keras.models import load_model
from dataset_loader import load_dataset

# === 1. Autoencoder'dan encoder'ı çıkar ===
autoencoder = tf.keras.models.load_model("autoencoder_model.keras")

# "encoded" layer'ın çıktısını al (encoder çıkışı)
encoder = Model(inputs=autoencoder.input,
                outputs=autoencoder.get_layer("encoded").output)

# === 2. Veri setini yükle ===
data_dir = r"C:\Users\HP\Desktop\DERSLER\CNN_Proje\dataset\images\images"
X_train, X_val, X_test, y_train, y_val, y_test, class_names = load_dataset(data_dir)

# === 3. Görselleri encode et (özellik çıkar) ===
X_train_encoded = encoder.predict(X_train)
X_val_encoded = encoder.predict(X_val)
X_test_encoded = encoder.predict(X_test)

# === 4. Çıkan tensörleri flatten et ===
X_train_flat = X_train_encoded.reshape((X_train_encoded.shape[0], -1))
X_val_flat = X_val_encoded.reshape((X_val_encoded.shape[0], -1))
X_test_flat = X_test_encoded.reshape((X_test_encoded.shape[0], -1))

# === 5. Yeni bir MLP sınıflandırıcı oluştur ===
input_dim = X_train_flat.shape[1]

classifier = models.Sequential([
    Input(shape=(input_dim,)),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')
])

classifier.compile(optimizer='adam',
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

# === 6. Eğit ===
classifier.fit(X_train_flat, y_train,
               validation_data=(X_val_flat, y_val),
               epochs=10,
               batch_size=32)

# === 7. Değerlendir ===
loss, acc = classifier.evaluate(X_test_flat, y_test)
print(f"\n Test doğruluğu (CNN + Autoencoder özellikleri): {acc * 100:.2f}%")

# === 8. Kaydet (isteğe bağlı) ===
classifier.save("cnn_autoencoder_classifier.keras")
