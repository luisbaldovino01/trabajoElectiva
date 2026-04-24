def entrenar_modelo():
    import pandas as pd
    import pickle
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsRegressor

    data = pd.read_csv("dataset_ciclismo_fatiga.csv")

    X = data[["frecuencia_cardiaca","potencia","cadencia","tiempo","temperatura","pendiente","velocidad"]]
    y = data["fatiga"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    model = KNeighborsRegressor(n_neighbors=13)
    model.fit(X_train, y_train)

    pickle.dump(model, open("modelo_fatiga.pkl", "wb"))
    pickle.dump(scaler, open("scaler.pkl", "wb"))
