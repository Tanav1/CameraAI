import streamlit as st
import keras.utils as image

# Streamlit app
st.title('Image Classification App :camera_with_flash: :frame_with_picture:')
st.subheader('WAI Incubator Cohort 7 - Team Camera AI', divider='rainbow')

st.image('Blog.jpg.webp')

st.sidebar.success("Select a model from the sidebar.")

st.subheader('Focus Questions')
st.markdown(
"""
How can real and fake images be distinguished?  
\n
Should our model use only images as input, or a combination of images and text prompts?  
\n
How does model the perform on images significantly different from the training data, e.g., faces, and should the model be fine-tuned on additional data to perform better?  
\n
How can we deploy the model and interactively demo it during our presentation?  
""")

st.subheader('Datasets')

multi = """
Main Dataset:
https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images/data    
\n
Training Images: https://www.cs.toronto.edu/~kriz/cifar.html  
\n
Useful Notebooks: https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images/code   
"""
st.markdown(multi)
