import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from PIL import Image
import altair as alt
import requests
import validators
import pickle
import time
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

#[theme]
#backgroundColor="#0E117"   <- de la web en ejecucion arriba a la derecha en settings --> theme

df = pd.read_csv('vehiculos-de-segunda-mano-sample.csv')

marcas = list(df['make'].unique())

#Representamos la informacion obtenida

st.header(':blue[¿Por cuánto puedes anunciar tu vehículo?] :car:')

st.markdown('Con nuestra aplicacion puedes saber el precio óptimo para vender tu coche')
st.markdown('¡Introduce los datos!')

image = Image.open("coches.jpeg")
st.image(image, caption=None, width=400, output_format="JPEG")

marca = st.selectbox('Elige una marca', marcas)
km = st.slider('¿Cuántos kms tiene?', 0, 350000, 5000)
edad = st.slider('¿Cuántos años tiene?', 0, 30, 5)
fuel = st.selectbox('Tipo de combustible', list(df['fuel'].dropna().unique()))
shift = st.selectbox('Tipo de cambio', list(df['shift'].dropna().unique()))
st.markdown(f'Vehículo elegido: {marca}, Kms: {km}, Antigüedad: {edad}, Combustible: {fuel}, Cambio: {shift}' )

x_test = {"make": marca, "fuel": fuel, "kms": km, "shift":shift,  "antiguedad": edad}

x_test = pd.DataFrame([x_test])

#LOAD MODEL

from auxiliar import obtener_clase 

mi_instancia = obtener_clase()

prediccion = mi_instancia.predict(x_test)

#st.text(f'Precio sugerido al que puedes anunciar tu coche')

st.markdown("*Pulsa para conocer el precio sugerido al que puedes anunciar tu* ***coche***.")

#st.button("Reset", type="primary")
if st.button('Calcular:'):

    st.markdown(f'Estimación: {int(prediccion[0])} €')
    st.markdown('''Suerte !!! :money_with_wings:''')

else:
    st.write(' ')

st.divider()

st.title('Datos a tomar en cuenta')
st.components.v1.html("""<iframe title="Report Section" width="850" height="1000" src="https://app.powerbi.com/view?r=eyJrIjoiOWRjNWVmOGYtMzMwMy00YzY5LTliNTItZmVjYTVmMTNhY2FjIiwidCI6ImJiYjEzOGJhLWZjMDYtNDM2ZS04ODhlLTAyYmVjMzFlYTIzYSIsImMiOjl9" frameborder="0" allowFullScreen="true"></iframe>""", width=850, height=1000, scrolling=False)
