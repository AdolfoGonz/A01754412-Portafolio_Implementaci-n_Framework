"""

En esta parte del código se realiza la importación de las bibliotecas necesarias para llevar a cabo el preprocesamiento de datos, selección de características, construcción y evaluación de un modelo de regresión logística. Se utilizan bibliotecas como NumPy y pandas para la manipulación de datos, y Scikit-learn para implementar el modelo, seleccionar características importantes y ajustar los hiperparámetros. Además, se importan herramientas de evaluación para medir el rendimiento del modelo con métricas como precisión, recall, F1 y matriz de confusión. Finalmente, se incluye la biblioteca pickle para guardar y cargar el modelo entrenado.

"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
import pickle

"""
En esta sección, se carga el dataset que contiene los datos de calidad del aire desde un archivo CSV. 

Se define la ruta del archivo en la variable 'file_path' y se utiliza la función 'read_csv' de pandas para leer el archivo y convertirlo en un DataFrame. Este DataFrame almacenado en la variable 'data' será utilizado para entrenar el modelo, realizar análisis y generar predicciones. Asegurarse de que la ruta del archivo sea correcta es fundamental para evitar errores al cargar el dataset.

"""

file_path = '/Users/aguero/Desktop/A01754412-Portafolio_Implementaci-n_Framework/Data/Normalized_Air_Quality_Classification_Dataset.csv'
data = pd.read_csv(file_path)


"""
En esta parte del código, se separan las características (X) y la variable objetivo (y) del dataset.
 
La variable 'X' contiene todos los datos del dataset excepto la columna 'Air_Quality', que se elimina porque es la variable que se quiere predecir (la etiqueta). 'y' contiene únicamente la columna 'Air_Quality', que es el valor que el modelo intentará predecir en función de las características restantes.

Esto es una parte fundamental en la preparación de los datos para el entrenamiento de modelos de machine learning.

"""

X = data.drop('Air_Quality', axis=1)
y = data['Air_Quality']


"""
Este bloque de código se enfoca en preparar los datos numéricos para el entrenamiento del modelo.
 
Primero, se filtran solo las columnas numéricas del conjunto de datos original 'X', lo cual es esencial ya que muchos modelos de machine learning requieren que las características sean numéricas para poder procesarlas.

Después, se utiliza el 'SimpleImputer' con la estrategia de la mediana para imputar (rellenar) valores faltantes en las columnas numéricas. La mediana es una buena opción para imputar datos ya que es menos sensible a los valores atípicos  que la media. El método 'fit_transform' ajusta el imputador a los datos y transforma las características numéricas, reemplazando los valores faltantes por la mediana calculada de cada columna.
"""


X_numeric = X.select_dtypes(include=[np.number])

imputer = SimpleImputer(strategy='median')
X_numeric = imputer.fit_transform(X_numeric)


"""
En este bloque de código, se divide el conjunto de datos en tres subconjuntos: entrenamiento (train), validación (val) y prueba (test).

Primero, 'train_test_split' divide los datos en conjuntos de entrenamiento y un conjunto temporal (X_temp, y_temp) que será dividido más adelante para crear los conjuntos de validación y prueba. El 30% de los datos se asigna al conjunto temporal, y el 70% restante se utiliza para entrenar el modelo.

Luego, se vuelve a usar 'train_test_split' para dividir el conjunto temporal en dos partes iguales: el conjunto de validación y el conjunto de prueba. De este modo, se asegura que el 15% de los datos originales se use para validación y otro 15% para prueba.

El argumento 'random_state=42' garantiza que las divisiones sean reproducibles, de modo que los resultados sean consistentes en cada ejecución.
"""


X_train, X_temp, y_train, y_temp = train_test_split(
    X_numeric, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42)


"""
Este bloque escala los datos usando MinMaxScaler para normalizar las características, lo que mejora el rendimiento del modelo.
Luego, se define un modelo de regresión logística y se utiliza RFE para seleccionar las 10 características más importantes.

Finalmente, se establece una cuadrícula de parámetros (param_grid) que incluye diferentes valores de regularización, tipos de penalización y solvers, con el fin de optimizar el modelo mediante la búsqueda de los mejores hiperparámetros durante el ajuste.
"""

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


log_reg = LogisticRegression(random_state=42, max_iter=100)

rfe = RFE(log_reg, n_features_to_select=10)
X_train_rfe = rfe.fit_transform(X_train, y_train)
X_val_rfe = rfe.transform(X_val)
X_test_rfe = rfe.transform(X_test)

param_grid = [
    {'C': [0.01, 1, 10, 100], 'penalty': ['l2'], 'solver': [
        'lbfgs'], 'class_weight': [None, 'balanced']},
    {'C': [0.01, 1, 10, 100], 'penalty': ['l1'], 'solver': [
        'liblinear'], 'class_weight': [None, 'balanced']}
]


"""
En este bloque, GridSearchCV se utiliza para optimizar los hiperparámetros del modelo de regresión logística mediante validación cruzada con 5 divisiones (cv=5). Se exploran diferentes combinaciones de 'C', tipos de penalización ('l1' y 'l2'), solvers y balanceo de clases para maximizar la precisión ('scoring=accuracy'). 

Luego, se imprime el mejor conjunto de hiperparámetros encontrados.

A continuación, se evalúa el mejor modelo en el conjunto de validación, prediciendo las etiquetas y calculando las métricas de rendimiento: precisión, recall, F1 y accuracy. Estas métricas proporcionan una visión clara del desempeño del modelo en datos que no han sido utilizados para su entrenamiento.
"""
grid_search = GridSearchCV(LogisticRegression(random_state=42, max_iter=100),
                           param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_rfe, y_train)


best_model = grid_search.best_estimator_
print(f"Mejor parámetro C: {grid_search.best_params_['C']}")
print(f"Mejor penalización: {grid_search.best_params_['penalty']}")
print(f"Mejor solver: {grid_search.best_params_['solver']}")
print(f"Mejor balanceo de clases: {grid_search.best_params_['class_weight']}")


y_pred_val = best_model.predict(X_val_rfe)


accuracy_val = accuracy_score(y_val, y_pred_val)
precision_val = precision_score(y_val, y_pred_val, average='macro')
recall_val = recall_score(y_val, y_pred_val, average='macro')
f1_val = f1_score(y_val, y_pred_val, average='macro')


"""
Este bloque imprime los resultados de la evaluación del modelo en el conjunto de validación.
Se muestran las métricas clave: precisión (accuracy), precisión de las predicciones positivas (precision), recall (capacidad del modelo para identificar correctamente los positivos) y F1-Score (una media armónica entre precision y recall).

Cada una de estas métricas se imprime con una precisión de 16 decimales para mayor exactitud en la visualización de los resultados.

Además, se imprime la matriz de confusión, que compara las predicciones del modelo con los valores reales, mostrando el número de verdaderos positivos, falsos positivos, verdaderos negativos y falsos negativos, lo que proporciona una visión clara del rendimiento del modelo.
"""

print(f"Evaluación en el conjunto de validación:")
print(f"Precisión (Accuracy): {accuracy_val:.16f}")
print(f"Precisión: {precision_val:.16f}")
print(f"Recall: {recall_val:.16f}")
print(f"F1 Score: {f1_val:.16f}")

print("Matriz de confusión:\n", confusion_matrix(y_val, y_pred_val))


"""
En este bloque se evalúa el modelo en el conjunto de prueba. 
Se generan predicciones sobre los datos de prueba y se calculan las métricas de rendimiento: precisión (accuracy), precisión de predicciones positivas, recall y F1-Score, usando un promedio macro para dar igual peso a cada clase.
Finalmente, se imprimen estas métricas y la matriz de confusión, que compara las predicciones con los valores reales del conjunto de prueba.
"""
y_pred_test = best_model.predict(X_test_rfe)


accuracy_test = accuracy_score(y_test, y_pred_test)
precision_test = precision_score(y_test, y_pred_test, average='macro')
recall_test = recall_score(y_test, y_pred_test, average='macro')
f1_test = f1_score(y_test, y_pred_test, average='macro')


print(f"\nEvaluación en el conjunto de prueba:")
print(f"Precisión (Accuracy): {accuracy_test:.16f}")
print(f"Precisión: {precision_test:.16f}")
print(f"Recall: {recall_test:.16f}")
print(f"F1 Score: {f1_test:.16f}")

print("Matriz de confusión (Prueba):\n", confusion_matrix(y_test, y_pred_test))


"""
A continuación explicare por que se decidio guardar estos archivos

1. logistic_model_rfe.pkl:
    - Este archivo contiene el modelo de regresión logística ya entrenado con todos sus parámetros optimizados. 
    - Lo utilizo para hacer predicciones sin tener que volver a entrenar el modelo desde cero.

2. rfe_transformer.pkl:
    - Este archivo guarda el objeto RFE, que se utilizó para seleccionar las características más relevantes del dataset.
    - Lo cargo durante la predicción para asegurarme de que las mismas características seleccionadas en el entrenamiento se utilicen con los nuevos datos.

3. scaler.pkl:
    - Este archivo contiene el escalador (MinMaxScaler) que se utilizó para normalizar los datos de entrenamiento. 
    - Necesito cargarlo para escalar los nuevos datos de la misma manera y asegurar que el modelo reciba los datos en el mismo rango.

"""

with open('/Users/aguero/Desktop/A01754412-Portafolio_Implementaci-n_Framework/Model/rfe_transformer.pkl', 'wb') as rfe_file:
    pickle.dump(rfe, rfe_file)

with open('/Users/aguero/Desktop/A01754412-Portafolio_Implementaci-n_Framework/Model/logistic_model_rfe.pkl', 'wb') as model_file:
    pickle.dump(best_model, model_file)

with open('/Users/aguero/Desktop/A01754412-Portafolio_Implementaci-n_Framework/Model/scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)
