import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Cargar el modelo y el escalador
modelo_knn = joblib.load('modelo_knn.bin')
escalador = joblib.load('escalador.bin')

# Título y descripción
st.title("Asistente IA para cardiólogos")
st.write("""
Esta aplicación predice si un paciente tiene o no un problema cardiaco. 
Utiliza un modelo de aprendizaje automático basado en KNN entrenado con datos como edad y colesterol.
Puedes ingresar los datos en la pestaña de 'Ingresar datos' y ver la predicción en 'Predicción'.
""")

# Crear las pestañas
tab = st.radio("Selecciona una pestaña:", ("Ingresar datos", "Predicción"))

if tab == "Ingresar datos":
    # Crear los inputs para la edad y el colesterol
    edad = st.slider("Edad", 18, 80, 40)
    colesterol = st.slider("Colesterol", 50, 600, 200)
    
    # Almacenar los datos en una variable de sesión
    st.session_state.edad = edad
    st.session_state.colesterol = colesterol

    st.write(f"Datos ingresados: Edad = {edad}, Colesterol = {colesterol}")
    st.write("Ahora puedes ir a la pestaña de 'Predicción' para ver el resultado.")

elif tab == "Predicción":
    # Verificar si los datos están ingresados en la sesión
    if 'edad' not in st.session_state or 'colesterol' not in st.session_state:
        st.error("Por favor, ingresa los datos primero en la pestaña 'Ingresar datos'.")
    else:
        # Crear un DataFrame con los datos de entrada
        data = pd.DataFrame({
            "edad": [st.session_state.edad], 
            "colesterol": [st.session_state.colesterol]
        })

        # Normalizar los datos de entrada usando el escalador
        data_normalizada = escalador.transform(data)

        # Realizar la predicción
        prediccion = modelo_knn.predict(data_normalizada)

        # Mostrar el resultado
        if prediccion == 1:
            st.image("https://as01.epimg.net/deporteyvida/imagenes/2017/10/28/portada/1509177885_209365_1509178036_noticia_normal.jpg", caption="Problema cardiaco", use_column_width=True)
            st.write("La predicción es que el paciente tiene un problema cardiaco.")
        else:
            st.image("https://s28461.pcdn.co/wp-content/uploads/2017/07/Tu-corazo%CC%81n-consejos-para-mantenerlo-sano-y-fuerte.jpg", caption="Corazón saludable", use_column_width=True)
            st.write("La predicción es que el paciente no tiene un problema cardiaco.")

