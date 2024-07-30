from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Cargar datos de MNIST
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist["data"], mnist["target"]

# Preprocesar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.astype(float))

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Entrenar el modelo RF
rf_clf =  RandomForestClassifier(n_estimators=50)
rf_clf.fit(X_train, y_train)

# Evaluar el modelo RF
y_pred = rf_clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Guardar el modelo y el escalador
joblib.dump(rf_clf, "rf_mnist_model.pkl")
joblib.dump(scaler, "scaler.pkl")
