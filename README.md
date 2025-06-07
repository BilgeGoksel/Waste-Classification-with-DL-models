#  Recyclable and Household Waste Classification

Bu proje, çeşitli evsel ve geri dönüştürülebilir atık türlerini görüntüler üzerinden sınıflandırmak amacıyla derin öğrenme yöntemleriyle gerçekleştirilmiştir.

##  Veri Seti

Kullanılan veri seti: [Recyclable and Household Waste Classification](https://www.kaggle.com/datasets/alistairking/recyclable-and-household-waste-classification)

- Toplamda 30 farklı atık kategorisi içermektedir.
- Her kategori altında iki alt klasör yer alır:
  - `default`: stüdyo ortamında çekilmiş görseller
  - `real_world`: gerçek dünya koşullarında çekilmiş görseller
- Görseller `.png` formatındadır ve her sınıf yaklaşık 500 görüntü içerir.

## Uygulanan Yöntemler

Projede çeşitli derin öğrenme yaklaşımları değerlendirilmiştir:

- Temel CNN modeli (sıfırdan geliştirilmiş mimari)
- Transfer Learning:
  - MobileNetV2
  - ResNet50
  - EfficientNetB0
- AutoEncoder 

Her model hem stüdyo (default) hem de gerçek dünya (real_world) verileri üzerinde test edilmiştir.

##  Amaç

Amaç, atıkların otomatik olarak sınıflandırılmasında farklı derin öğrenme mimarilerinin başarılarını kıyaslamak ve transfer learning'in etkisini ortaya koymaktır.

## Gereksinimler

- TensorFlow
- NumPy
- OpenCV
- scikit-learn

Tüm deneyler Colab ve Anaconda ortamlarında gerçekleştirilmiştir. Eğitim kodları `.py` dosyaları olarak proje klasöründe yer almaktadır.

