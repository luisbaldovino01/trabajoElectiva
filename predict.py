import pandas as pd
import pickle

model = pickle.load(open("modelo_casas.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

def predict_fatiga(datos):
    datos = scaler.transform(datos)
    return model.predict(datos)