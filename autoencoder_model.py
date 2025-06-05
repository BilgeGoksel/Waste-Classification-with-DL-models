import tensorflow as tf
from tensorflow.keras import layers, models
from dataset_loader import load_dataset

def build_autoencoder(input_shape=(224, 224, 3), latent_dim=128):
    # Encoder
    encoder_input = layers.Input(shape=input_shape, name='encoder_input')
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoder_input)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2, 2), padding='same', name='encoded')(x)

    # Decoder
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same', name='decoder_output')(x)

    autoencoder = models.Model(encoder_input, decoded, name='autoencoder')
    return autoencoder

if __name__ == "__main__":
    import numpy as np
    from dataset_loader import load_dataset

    # GPU kontrolü
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            print("GPU kullanılacak.")
        except:
            print("GPU ayarlanamadı, CPU kullanılacak.")
    else:
        print("GPU bulunamadı, CPU kullanılacak.")

    # Veri yolu
    data_dir = r"C:\Users\HP\Desktop\DERSLER\CNN_Proje\dataset\images\images"

    # Veriyi yükle
    X_train, X_val, X_test, y_train, y_val, y_test, class_names = load_dataset(data_dir)

    # Autoencoder sadece X verisi ile eğitilir
    X_train_auto = np.concatenate([X_train, X_val])
    X_test_auto = X_test

    # Modeli oluştur
    autoencoder = build_autoencoder(input_shape=(224, 224, 3))

    # Derle
    autoencoder.compile(optimizer='adam', loss='mse')

    # Eğit
    autoencoder.fit(
        X_train_auto, X_train_auto,
        validation_data=(X_test_auto, X_test_auto),
        epochs=10,
        batch_size=32,
        shuffle=True
    )

    # Kaydet
    autoencoder.save("autoencoder_model.keras")
