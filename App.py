import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# App
def predictDigit(image):
    model = tf.keras.models.load_model("model/handwritten.h5")
    image = ImageOps.grayscale(image)
    img = image.resize((28,28))
    img = np.array(img, dtype='float32')
    img = img/255
    plt.imshow(img)
    plt.show()
    img = img.reshape((1,28,28,1))
    pred = model.predict(img)
    result = np.argmax(pred[0])
    return result

# Streamlit configuration
st.set_page_config(page_title='Reconocimiento de Dígitos escritos a mano', layout='wide')

# Aplicar el fondo de imagen con CSS
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://static.vecteezy.com/system/resources/previews/008/218/160/non_2x/horizontal-topographic-map-black-topographer-seamless-pattern-dark-typography-linear-background-for-mapping-and-audio-equalizer-backdrop-illustration-vector.jpg");
    background-size: cover;
}
h1, h2, button, .stButton>button {
    text-align: center;
    display: block;
    margin: 0 auto;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# Crear columnas para centrar el contenido
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.title('Reconocimiento de Dígitos escritos a mano')
    st.subheader("Dibuja el dígito en el panel y presiona 'Predecir'")

# Sidebar for customization options
with st.sidebar:
    st.title("Configuración de Dibujo")
    
    # Color picker for stroke color
    stroke_color = st.color_picker(
        'Selecciona el color del trazo', 
        '#FFFFFF',  # Default white
        key='stroke_color_picker'
    )
    
    # Slider for stroke width
    stroke_width = st.slider(
        'Selecciona el ancho de línea', 
        1, 30, 15,
        key='stroke_width_slider'
    )
    
    # Background color picker
    bg_color = st.color_picker(
        'Selecciona el color de fondo', 
        '#000000',  # Default black
        key='bg_color_picker'
    )

with col2:
    # Create canvas component with customizable settings
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        height=400,
        width=400,
        drawing_mode="freedraw",
        key="canvas",
    )

    # Add "Predict Now" button
    if st.button('Predecir'):
        if canvas_result.image_data is not None:
            input_numpy_array = np.array(canvas_result.image_data)
            input_image = Image.fromarray(input_numpy_array.astype('uint8'), 'RGBA')
            input_image.save('prediction/img.png')
            img = Image.open("prediction/img.png")
            res = predictDigit(img)
            st.header('El Dígito es : ' + str(res))
        else:
            st.header('Por favor dibuja en el canvas el dígito.')
