import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

data = pd.read_csv("dataset_ciclismo_fatiga.csv")

x = data[["frecuencia_cardiaca", "potencia", "cadencia","tiempo", "temperatura", "pendiente", "velocidad"]]
y = data["fatiga"]

print("PROPORCIÓN 80/20")

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#regresinn Lineal
modelo_lr = LinearRegression()
modelo_lr.fit(x_train, y_train)
y_pred_lr = modelo_lr.predict(x_test)

print("\nRegresión Lineal")
print("MSE:", mean_squared_error(y_test, y_pred_lr))
print("MAE:", mean_absolute_error(y_test, y_pred_lr))
print("R2:", r2_score(y_test, y_pred_lr))

#KNN probando varios K
print("\nKNN probando k")

mejor_k = 0
mejor_r2 = -1

for k in [3,5,7,9]:
    modelo_knn_temp = KNeighborsRegressor(n_neighbors=k)
    modelo_knn_temp.fit(x_train, y_train)
    y_pred_temp = modelo_knn_temp.predict(x_test)

    r2 = r2_score(y_test, y_pred_temp)
    print("K:", k, "R2:", r2)

    if r2 > mejor_r2:
        mejor_r2 = r2
        mejor_k = k

print("Mejor K:", mejor_k)

#modelo final KNN
modelo_knn = KNeighborsRegressor(n_neighbors=mejor_k)
modelo_knn.fit(x_train, y_train)
y_pred_knn = modelo_knn.predict(x_test)

print("\nKNN FINAL")
print("MSE:", mean_squared_error(y_test, y_pred_knn))
print("MAE:", mean_absolute_error(y_test, y_pred_knn))
print("R2:", r2_score(y_test, y_pred_knn))


print("\nPROPORCIÓN 70/30")

x_train2, x_test2, y_train2, y_test2 = train_test_split(x, y, test_size=0.3, random_state=42)

scaler2 = StandardScaler()
x_train2 = scaler2.fit_transform(x_train2)
x_test2 = scaler2.transform(x_test2)

#regresión Lineal
modelo_lr2 = LinearRegression()
modelo_lr2.fit(x_train2, y_train2)
y_pred_lr2 = modelo_lr2.predict(x_test2)

print("\nRegresión Lineal")
print("MSE:", mean_squared_error(y_test2, y_pred_lr2))
print("MAE:", mean_absolute_error(y_test2, y_pred_lr2))
print("R2:", r2_score(y_test2, y_pred_lr2))

#KNN probando varios K
print("\nKNN probando k")

mejor_k2 = 0
mejor_r2_2 = -1

for k in [3,5,7,9]:
    modelo_knn_temp2 = KNeighborsRegressor(n_neighbors=k)
    modelo_knn_temp2.fit(x_train2, y_train2)
    y_pred_temp2 = modelo_knn_temp2.predict(x_test2)

    r2 = r2_score(y_test2, y_pred_temp2)
    print("K:", k, "R2:", r2)

    if r2 > mejor_r2_2:
        mejor_r2_2 = r2
        mejor_k2 = k

print("Mejor K:", mejor_k2)

# Modelo final KNN 70/30
modelo_knn2 = KNeighborsRegressor(n_neighbors=mejor_k2)
modelo_knn2.fit(x_train2, y_train2)
y_pred_knn2 = modelo_knn2.predict(x_test2)

print("\nKNN FINAL 70/30")
print("MSE:", mean_squared_error(y_test2, y_pred_knn2))
print("MAE:", mean_absolute_error(y_test2, y_pred_knn2))
print("R2:", r2_score(y_test2, y_pred_knn2))


frecuenciaCardiaca = int(input("\nDigite la frecuencia cardiaca: "))
potenciaDato = int(input("Digite la potencia: "))
cadenciaDato = int(input("Digite la cadencia: "))
tiempoDato = float(input("Digite el tiempo: "))
temperaturaDato = float(input("Digite la temperatura: "))
pendienteDato = float(input("Digite la pendiente: "))
velocidadDato = float(input("Digite la velocidad: "))

nuevo = pd.DataFrame([[frecuenciaCardiaca, potenciaDato, cadenciaDato, tiempoDato, temperaturaDato, pendienteDato, velocidadDato]],
columns=["frecuencia_cardiaca", "potencia", "cadencia", "tiempo", "temperatura", "pendiente", "velocidad"])

nuevo_scaler = scaler.transform(nuevo)

pred_lr = modelo_lr.predict(nuevo_scaler)
pred_knn = modelo_knn.predict(nuevo_scaler)

def interpretacion(valor):
    if valor <= 20:
        print("Sin fatiga significativa")
    elif valor <= 40:
        print("Esfuerzo leve")
    elif valor <= 60:
        print("Fatiga moderada")
    elif valor <= 80:
        print("Fatiga evidente")
    else:
        print("Fatiga extrema")

print("\nPredicción Regresión Lineal:", round(pred_lr[0], 2))
interpretacion(pred_lr[0])

print("\nPredicción KNN:", round(pred_knn[0], 2))
interpretacion(pred_knn[0])


print("\nCONCLUSIÓN")

r2_lr_80 = r2_score(y_test, y_pred_lr)
r2_knn_80 = r2_score(y_test, y_pred_knn)

r2_lr_70 = r2_score(y_test2, y_pred_lr2)
r2_knn_70 = r2_score(y_test2, y_pred_knn2)

print("\n80/20 -> LR:", r2_lr_80, "KNN:", r2_knn_80)
print("70/30 -> LR:", r2_lr_70, "KNN:", r2_knn_70)

if r2_knn_80 > r2_lr_80:
    print("\nEn 80/20 el mejor modelo es KNN")
else:
    print("\nEn 80/20 el mejor modelo es Regresión Lineal")

if r2_knn_70 > r2_lr_70:
    print("En 70/30 el mejor modelo es KNN")
else:
    print("En 70/30 el mejor modelo es Regresión Lineal")