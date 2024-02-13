import streamlit as st
from PIL import Image
import cv2
import numpy as np
from keras.models import load_model
import json
from streamlit_lottie import st_lottie

# Load variables
from variables import IMAGE_SIZE, MODEL_PATH, LABELS

# Load pretrained CNN Model from MODEL_PATH

model = load_model(r"C:\Users\jyoth\Downloads\SignLanguageAlphabetRecognizer\model.h5")



def pre_process(img_array):
    """
    Preprocess the image array for prediction.
    """
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    img_array = cv2.resize(img_array, (IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_array.reshape(IMAGE_SIZE, IMAGE_SIZE, 1)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(img_array):
    """
    Predict the sign language alphabet from the image array.
    """
    img_array = pre_process(img_array)
    preds = model.predict(img_array)
    preds *= 100
    most_likely_class_index = int(np.argmax(preds))
    return preds.max(), LABELS[most_likely_class_index]


def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)
    
    
def main():
    st.title("Sign Language Alphabet Recognizer")


    image = Image.open(r"C:\Users\jyoth\Downloads\SignLanguageAlphabetRecognizer\holdhand.jpg")
    st.image(image, use_column_width=True)

    
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image.", use_column_width=True)

        image_array = np.array(image)
        confidence, predicted_letter = predict_image(image_array)

        st.write(f"Predicted Letter: {predicted_letter}")
        st.write(f"Confidence: {confidence:.2f}%")

if __name__ == "__main__":
    main()
