
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import pickle
import joblib
from sklearn.cluster import KMeans
from sklearn.metrics import precision_score
from sklearn import preprocessing
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler

class Modelo_XGBoostRegresion:
    def __init__(self):
        
        self.le = pickle.load(open('label_encoder.sav', 'rb'))
        self.scaler = pickle.load(open('normalizador.sav', 'rb'))
        self.model = pickle.load(open('modelo_final.sav', 'rb'))

    def predict(self, X_new):
        # X_new llega de la misma forma que el df original
        
        # Aplicar replace
        X_new["shift"] = X_new["shift"].replace({'manual': 1, 'automatic': 0})
        X_new["fuel"] = X_new["fuel"].replace({'Gasolina': 0, 'Diésel': 1, 'Eléctrico': 2, 'Otros': 3})
        
        # Aplicar self.le
        X_le = self.le.transform(X_new['make'])        
        
        X_new_le = pd.DataFrame(X_new, columns=X_new.columns)
        
        X_new_le['make'] = X_le        
        
        # Escalar los datos
        X_new_sc = self.scaler.transform(X_new_le)
        
        # Aplicar el modelo entrenado sobre los datos que acabo de preparar
        y_pred = self.model.predict(X_new_sc)
        
        # Reescalo la predicción
        y_pred_convertida = np.exp(y_pred)
        
        return y_pred_convertida


def obtener_clase():
    return Modelo_XGBoostRegresion()