import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# Load model (cache it for better performance)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/handwritten.h5")

# Improved prediction function
def predict_digit(image):
    model = load_model()
    
    # Convert to grayscale and invert colors (MNIST style)
    image = ImageOps.grayscale(image)
    image = ImageOps.invert(image)
    
    # Resize and normalize
    img = image.resize((28, 28))
    img = np.array(img, dtype='float32')
    img = img / 255.0
    
    # Reshape for model input
    img = img.reshape((1, 28, 28, 1))
    
    # Make prediction
    pred = model.predict(img)
    result = np.argmax(pred[0])
    confidence = float(np.max(pred[0]))
    
    return result, confidence

# Streamlit configuration
st.set_page_config(
    page_title='Reconocimiento de Dígitos escritos a mano',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Custom CSS for styling
st.markdown(
    """
    <style>
    /* Main app background */
    .stApp {
        background-color: #f5f5f5;
    }
    
    /* Sidebar background */
    .stSidebar {
        background-color: #e1e5ee;
        padding: 1rem;
        border-radius: 10px;
    }
    
    /* Canvas container */
    .canvas-container {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Prediction result */
    .prediction-result {
        font-size: 2rem;
        font-weight: bold;
        color: #2e7d32;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Main app
st.title('Reconocimiento de Dígitos escritos a mano')
st.subheader("Dibuja un dígito (0-9) en el área de abajo y haz clic en 'Predecir'")

# Sidebar for settings
with st.sidebar:
    st.title("⚙️ Configuración")
    
    stroke_color = st.color_picker(
        'Color del trazo', 
        '#000000',  # Default black
        key='stroke_color'
    )
    
    stroke_width = st.slider(
        'Grosor del trazo', 
        1, 30, 15,
        key='stroke_width'
    )
    
    bg_color = st.color_picker(
        'Color de fondo del lienzo', 
        '#ffffff',  # Default white
        key='bg_color'
    )
    
    st.markdown("---")
    st.title("ℹ️ Acerca de")
    st.markdown("""
    Esta aplicación utiliza una red neuronal convolucional (CNN) 
    entrenada en el conjunto de datos MNIST para reconocer dígitos 
    escritos a mano.
    
    **Consejos para mejores resultados:**
    - Dibuja el dígito centrado
    - Usa trazos claros y definidos
    - Prueba con diferentes grosores de línea
    """)

# Canvas with container for better styling
with st.container():
    st.markdown("<div class='canvas-container'>", unsafe_allow_html=True)
    
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",  # Transparent fill
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        height=280,  # Larger canvas for better drawing
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )
    
    st.markdown("</div>", unsafe_allow_html=True)

# Prediction button and results
if st.button('🔮 Predecir', type='primary'):
    if canvas_result.image_data is not None:
        # Process image
        input_numpy_array = np.array(canvas_result.image_data)
        input_image = Image.fromarray(input_numpy_array.astype('uint8'), 'RGBA')
        
        # Convert to RGB and remove alpha channel
        input_image = input_image.convert('RGB')
        
        # Make prediction
        result, confidence = predict_digit(input_image)
        
        # Display results
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Resultado:")
            st.markdown(f"<p class='prediction-result'>El dígito es: {result}</p>", unsafe_allow_html=True)
            st.write(f"Confianza: {confidence:.2%}")
            
        with col2:
            # Show processed image
            processed_img = ImageOps.grayscale(input_image)
            processed_img = ImageOps.invert(processed_img)
            processed_img = processed_img.resize((150, 150))
            st.image(processed_img, caption="Imagen procesada", use_column_width=False)
    else:
        st.warning("Por favor dibuja un dígito en el lienzo primero.")
