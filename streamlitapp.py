import streamlit as st
import pandas as pd
import subprocess
import pickle

st.title("Predicción de Fatiga - ML Pipeline")

# BOTÓN ENTRENAR (solo una vez manual)
if st.button("Entrenar modelo"):
    subprocess.run(["python", "train.py"])
    st.success("Modelo entrenado y guardado")

# cargar modelo
model = pickle.load(open("modelo_casas.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.header("Predicción")

fc = st.number_input("Frecuencia cardiaca")
pot = st.number_input("Potencia")
cad = st.number_input("Cadencia")
tiempo = st.number_input("Tiempo")
temp = st.number_input("Temperatura")
pend = st.number_input("Pendiente")
vel = st.number_input("Velocidad")

if st.button("Predecir"):
    datos = pd.DataFrame([[fc, pot, cad, tiempo, temp, pend, vel]],
                         columns=["frecuencia_cardiaca","potencia","cadencia","tiempo","temperatura","pendiente","velocidad"])

    datos = scaler.transform(datos)
    pred = model.predict(datos)[0]

    st.success(f"Fatiga predicha: {round(pred,2)}")

    if pred <= 20:
        st.info("Sin fatiga significativa")
    elif pred <= 40:
        st.warning("Esfuerzo leve")
    elif pred <= 60:
        st.warning("Fatiga moderada")
    elif pred <= 80:
        st.error("Fatiga evidente")
    else:
        st.error("Fatiga extrema")