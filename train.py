def entrenar_modelo():

    import pandas as pd
    import pickle
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    # cargar datos
    data = pd.read_csv("dataset_ciclismo_fatiga.csv")

    X = data[
        ["frecuencia_cardiaca", "potencia", "cadencia",
         "tiempo", "temperatura", "pendiente", "velocidad"]
    ]
    y = data["fatiga"]

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # escalado
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # modelo
    model = KNeighborsRegressor(n_neighbors=13)
    model.fit(X_train, y_train)

    # predicción
    y_pred = model.predict(X_test)

    # métricas
    metrics = {
        "MSE": mean_squared_error(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred)
    }

    # guardar modelo y scaler
    pickle.dump(model, open("modelo_fatiga.pkl", "wb"))
    pickle.dump(scaler, open("scaler.pkl", "wb"))

    # devolver métricas a Streamlit
    return metrics
