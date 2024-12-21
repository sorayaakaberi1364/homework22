# مقدمه
# در این پروژه قصد داریم با استفاده از شبکه‌های عصبی کانولوشنی (CNN)
# دیتاست MNIST را که شامل تصاویر دست‌نویس ارقام از 0 تا 9 است،
# تجزیه و تحلیل کنیم. هدف اصلی ما ساخت مدلی است که بتواند
# با دقت و کارایی بالا این ارقام را شناسایی کند.

# کتابخانه‌های مورد نیاز
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

# بارگذاری دیتاست MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# پیش‌پردازش داده‌ها
# تغییر شکل تصاویر و نرمال‌سازی مقادیر
x_train = x_train.reshape((60000, 28, 28, 1)).astype('float32') / 255
x_test = x_test.reshape((10000, 28, 28, 1)).astype('float32') / 255
y_train = to_categorical(y_train, 10)  # تبدیل به فرمت دسته‌ای
y_test = to_categorical(y_test, 10)

# ساخت مدل CNN
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# کامپایل مدل
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# آموزش مدل و ذخیره تاریخچه آموزش
history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2)

# ارزیابی مدل
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)

# ترسیم نمودار دقت و از دست دادن آموزش و اعتبارسنجی
plt.figure(figsize=(12, 4))

# دقت
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# از دست دادن
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# محاسبه و ترسیم ماتریس سردرگمی
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

conf_matrix = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
