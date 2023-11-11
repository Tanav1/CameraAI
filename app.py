import os
import streamlit as st
from tensorflow.keras.models import load_model
import keras.utils as image
import numpy as np

from streamlit_extras.let_it_rain import rain

# Load the pre-trained model
model = load_model('m2.h5', compile=False)

# Streamlit app
st.title('Image Classification App :camera_with_flash: :frame_with_picture:')
st.subheader('Women in AI Incubator Cohort 7 - Team Camera AI', divider='rainbow')

# File upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Sample input (for testing purposes)
if uploaded_file is None:
    st.subheader('Is it Real or Fake?')
    sample_input = '62f2dd5573c01a4523b4ace6_Deepfake-Optional-Body-of-Article.jpeg'
    img = image.load_img(sample_input)
    st.image(img, caption='Example of an AI Generated Deepfake Image', use_column_width=True)

if uploaded_file is not None:
    # Display the uploaded image
    st.subheader('Uploaded Image:')
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    # Preprocess the uploaded image
    img = image.load_img(uploaded_file, target_size=(32, 32))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Make predictions
    predictions = model.predict(img_array)
    redicted_class = np.argmax(predictions)

    binary_predictions = (predictions > 0.5).astype(int)
    #binary_predictions


    print("Model predictions:", binary_predictions)

    
    if st.button('Predict!'):
        predicted = model.predict(img_array)
        #st.success('The image is predicted to be: {}'.format(predicted))
        st.subheader('Prediction:')
        if binary_predictions == 0:
            st.error('This image is predicted to be FAKE.')
            rain(emoji='🤖', font_size=65, falling_speed=5)
        else:
            st.success('This image is predicted to be REAL.')
            st.balloons()
        

        