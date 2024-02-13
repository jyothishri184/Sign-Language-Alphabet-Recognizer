import cv2
import numpy as np
from variables import *
from keras.models import load_model

model = load_model(r"C:\Users\jyoth\Downloads\SignLanguageAlphabetRecognizer\model.h5")

def pre_process(img_path):
    img_array = cv2.imread(img_path)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    img_array = cv2.resize(img_array, (IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_array.reshape(IMAGE_SIZE, IMAGE_SIZE, 1)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(img_path):
    img_array = pre_process(img_path)
    preds = model.predict(img_array)
    preds *= 100
    most_likely_class_index = int(np.argmax(preds))
    return preds.max(), LABELS[most_likely_class_index]
