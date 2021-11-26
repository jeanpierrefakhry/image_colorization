import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from skimage.transform import resize
from PIL import Image
import streamlit as st
from skimage.color import rgb2lab, lab2rgb
from tensorflow.keras.models import model_from_json
from io import BytesIO
import base64
import cv2

PATH = "model/"

@st.cache(allow_output_mutation= True)

def load_own_model():
    json_file = open(PATH+'model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(PATH+"checkpoint.h5")
    return model

def get_image_download_link(img):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/jpg;base64,{img_str}" download="colorized_image.jpeg">Download result</a>'
    return href

if __name__=="__main__":
    results = st.empty()
    st.image('logo-usek.jpg',width=200)
    st.markdown("<h1 style='text-align: center; color: red;'>Final Year Project</h1>", unsafe_allow_html=True)
    st.write("""
    # Artificial Intelligence for Automatic Colorization of Monochromatic Images
    ### The aim of this project is to design a tool for automatic image colorization based on modern artificial intelligence techniques, and to provide a simple user interface for the interaction with the tool.
    """)
    uploaded_img = st.file_uploader(label='Upload an image: ')
    if uploaded_img:
        st.image(uploaded_img, caption="Grayscale Image", width=None)
        results.info("Please wait for your results")
        model = load_own_model()
        black_color = [0, 0, 0]
        img1_color = []
        img_input = Image.open(uploaded_img).convert('RGB')
        img_cv = np.array(img_input)
        left = 0
        top = 0
        height = img_cv.shape[0]
        width = img_cv.shape[1]
        if img_cv.shape[0] < img_cv.shape[1]:
            top = (int)((img_cv.shape[1] - img_cv.shape[0])/2)
            height = top + img_cv.shape[0]
            width = img_cv.shape[1]
            left = 0
            img_cv = cv2.copyMakeBorder(img_cv, (int)((img_cv.shape[1] - img_cv.shape[0]) / 2),
                                        (int)((img_cv.shape[1] - img_cv.shape[0]) / 2), 0, 0, cv2.BORDER_CONSTANT,
                                        value=black_color)

        elif img_cv.shape[0] > img_cv.shape[1]:
            left = (int)((img_cv.shape[0] - img_cv.shape[1])/2)
            width = left + img_cv.shape[1]
            top = 0
            height = img_cv.shape[0]
            img_cv = cv2.copyMakeBorder(img_cv, 0, 0, (int)((img_cv.shape[0] - img_cv.shape[1]) / 2),
                                        (int)((img_cv.shape[0] - img_cv.shape[1]) / 2), cv2.BORDER_CONSTANT,
                                        value=black_color)

        img1 = img_to_array(img_cv)
        img1 = resize(img1, (256, 256))
        img1_color.append(img1)
        img1_color = np.array(img1_color, dtype=float)
        img1_color = rgb2lab(1.0 / 255 * img1_color)[:, :, :, 0]
        img1_color = img1_color.reshape(img1_color.shape + (1,))
        output1 = model.predict(img1_color)
        output1 = output1 * 128
        result = np.zeros((256, 256, 3))
        result[:, :, 0] = img1_color[0][:, :, 0]
        result[:, :, 1:] = output1[0]
        img = lab2rgb(result)
        img = resize(img, (img_cv.shape[0], img_cv.shape[1]))
        img = img[top:height, left:width]
        st.image(img, caption="Colorized Image ", clamp=True, width=None)
        img = Image.fromarray(np.uint8((img)*255))
        results.success("Image has been colorized")
        st.markdown(get_image_download_link(img), unsafe_allow_html=True)

st.write("""
        ### This Project has been worked on by Jean-Pierre Fakhry under the supervision of Prof. Charles YAACOUB and Mr. Pascal DAMIAN and is for educational purposes only
        """)

