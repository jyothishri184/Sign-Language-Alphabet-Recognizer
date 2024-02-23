import streamlit as st
from PIL import Image
import cv2
import numpy as np
from keras.models import load_model


IMAGE_SIZE = 50 
MODEL_PATH = "model.h5"
LABELS = ['A','B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M','N','O', 'P', 'Q','R', 'S', 'T','U', 'V', 'W', 'X', 'Y', 'Z']

# Minimum confidence percentage i.e allowed for prediction
THRESHOLD = 25


# Load pretrained CNN Model from MODEL_PATH
model = load_model(r"C:\Users\jyoth\Downloads\Projects\SignLanguageAlphabetRecognizer\model.h5")



def pre_process(img_array):
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    img_array = cv2.resize(img_array, (IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_array.reshape(IMAGE_SIZE, IMAGE_SIZE, 1)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(img_array):
    img_array = pre_process(img_array)
    preds = model.predict(img_array) #return ans image array with confidence for each class ---> array([[0.01, 0.95, 0.04]]) if class A,B,C --> then class B
    preds *= 100 #for representing confidence as percentage
    most_likely_class_index = int(np.argmax(preds)) #return the index of maximum value in an array
    return preds.max(), LABELS[most_likely_class_index]

    
def main():
    st.title("Sign Language Alphabet Recognizer")

    image = Image.open(r"C:\Users\jyoth\Downloads\Projects\SignLanguageAlphabetRecognizer\holdhand.jpg")
    st.image(image, use_column_width=True)

    
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        image_array = np.array(image)
        confidence, predicted_letter = predict_image(image_array)

        st.write(f"Predicted Letter: {predicted_letter}")
        st.write(f"Confidence: {confidence:.2f}%")

if __name__ == "__main__":
    main()
