import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import math

train_data_dir = 'data/train'
validation_data_dir = 'data/test'

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=30,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    color_mode='grayscale',
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical',
    shuffle=True)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    color_mode='grayscale',
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical',
    shuffle=True)

if os.path.exists('model_file.h5'):
    print("Знайдено збережену модель. Починаємо донавчання...")
    model = load_model('model_file.h5')
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
else:
    print("Файл моделі не знайдено! Перевірте назву файлу.")

num_train_imgs = train_generator.n
num_test_imgs = validation_generator.n

print('Кількість тренувальних зображень:', num_train_imgs)
print('Кількість тестових зображень:', num_test_imgs)

steps_per_epoch = math.ceil(num_train_imgs / 32)
validation_steps = math.ceil(num_test_imgs / 32)

epochs = 15 

print(f"Починаємо донавчання на {epochs} епохах...")

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_steps)

# Зберігаємо оновлену версію під новою назвою, щоб не заплутатися
model.save('model_file_v2.h5')
print("Покращена модель збережена як model_file_v2.h5")
