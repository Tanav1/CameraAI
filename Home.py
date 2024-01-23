import streamlit as st
import keras.utils as image


#st.set_page_config(
#    page_title="Image Classification App :camera_with_flash: :frame_with_picture:",
#    page_icon="👋",
#)

# Streamlit app
st.title('Image Classification App :camera_with_flash: :frame_with_picture:')
st.subheader('Women in AI Incubator Cohort 7 - Team Camera AI', divider='rainbow')

st.image('Blog.jpg.webp')

st.sidebar.success("Select a model from the sidebar.")

st.subheader('Focus Questions')
st.markdown(
"""

How can real and fake images be distinguished?
Should our model use only images as input, or a combination of images and text prompts?
How does model the perform on images significantly different from the training data, e.g., faces, and should the model be fine-tuned on additional data to perform better?
How can we deploy the model and interactively demo it during our presentation?
""")

st.subheader('Datasets')

st.markdown(
"""
https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images/data - primary dataset
https://www.cs.toronto.edu/~kriz/cifar.html - may try to use this data to examine performance in specific classes, but this only exists for real data
https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images/code - useful notebooks
"""
)