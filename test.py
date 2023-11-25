import cv2
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# Загрузка обученной модели
model = load_model('handwritten_text_recognition_model.h5')

def preprocess_image(image_path):
    # Загрузка изображения и предварительная обработка
    img = Image.open(image_path).convert('L')  # Конвертация в оттенки серого
    img = img.resize((28, 28))
    img_array = np.array(img)
    img_array = img_array.reshape((1, 28, 28, 1)).astype('float32') / 255
    return img_array

def predict_handwritten_text(image_path):
    # Предварительная обработка изображения
    input_data = preprocess_image(image_path)
    predictions = model.predict(input_data)
    predicted_label = np.argmax(predictions)
    print(f'Predicted Label: {predicted_label}')

image_path = 'C:/Users/nazar/OneDrive/Desktop/3.jpg'
predict_handwritten_text(image_path)