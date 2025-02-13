import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Cargar el modelo
modelo_knn = joblib.load('modelo_knn.bin')

# Nombres de las características
nombres_caracteristicas = ['edad', 'colesterol']

# Crear el escalador (MinMaxScaler) para la normalización explícita
escalador = MinMaxScaler()

# Función para predecir problemas cardíacos
def predecir_problema_cardiaco(edad, colesterol):
    # Crear un DataFrame con los datos de entrada, utilizando los nombres de las características
    datos_entrada = pd.DataFrame([[edad, colesterol]], columns=nombres_caracteristicas)
    
    # Normalizar los datos de entrada usando MinMaxScaler
    datos_normalizados = escalador.fit_transform(datos_entrada)
    
    # Predecir usando el modelo KNN
    prediccion = modelo_knn.predict(datos_normalizados)
    
    # Resultado de la predicción
    return prediccion[0]

# Título de la aplicación
st.title("Asistente IA para cardiólogos")

# Introducción
st.write("""
    Bienvenido al asistente de diagnóstico basado en inteligencia artificial para cardiólogos. 
    Esta aplicación predice si una persona sufre o no de problemas cardíacos, utilizando 
    factores como la edad y los niveles de colesterol. Ingrese los datos y obtendrá el diagnóstico
    junto con una imagen relacionada con el estado del corazón.
""")

# Crear un contenedor de pestañas (tabs)
tab1, tab2 = st.tabs(["Cálculo de Datos", "Predicción"])

# **Tab 1: Cálculo de Datos**
with tab1:
    st.header("Cálculo de Datos")
    
    # Entrada de datos del usuario
    edad = st.slider("Edad", 18, 80, 30)
    colesterol = st.slider("Colesterol", 50, 600, 200)

    # Mostrar los datos ingresados
    st.write(f"Edad: {edad}")
    st.write(f"Colesterol: {colesterol}")

    # Crear un DataFrame para mostrar los datos antes de la normalización
    datos_entrada = pd.DataFrame([[edad, colesterol]], columns=nombres_caracteristicas)
    st.write("Datos de entrada:")
    st.dataframe(datos_entrada)

    # Normalizar los datos usando el escalador previamente entrenado
    datos_normalizados = escalador.fit_transform(datos_entrada)

    # Mostrar los datos normalizados
    st.write("Datos normalizados:")
    st.dataframe(datos_normalizados)

# **Tab 2: Predicción**
with tab2:
    st.header("Predicción de Problemas Cardíacos")
    
    # Entrada de datos del usuario
    edad = st.slider("Edad", 18, 80, 30)
    colesterol = st.slider("Colesterol", 50, 600, 200)

    # Predicción
    if st.button("Predecir"):
        resultado = predecir_problema_cardiaco(edad, colesterol)

        if resultado == 1:
            # Mostrar imagen si tiene problema cardíaco
            st.image("https://as01.epimg.net/deporteyvida/imagenes/2017/10/28/portada/1509177885_209365_1509178036_noticia_normal.jpg", caption="Problema Cardíaco Detectado", use_column_width=True)
            st.write("**Diagnóstico**: Tiene un problema cardíaco.")
        else:
            # Mostrar imagen si no tiene problema cardíaco
            st.image("https://s28461.pcdn.co/wp-content/uploads/2017/07/Tu-corazo%CC%81n-consejos-para-mantenerlo-sano-y-fuerte.jpg", caption="Corazón saludable", use_column_width=True)
            st.write("**Diagnóstico**: No tiene problemas cardíacos.")
