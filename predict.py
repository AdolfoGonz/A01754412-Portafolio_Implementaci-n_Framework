"""
Se importa la librería NumPy, abreviada como 'np', para realizar operaciones matemáticas y trabajar con matrices de datos.
También importo 'pickle', que me permite guardar y cargar objetos de Python, como el modelo entrenado y otros componentes del pipeline.
"""
import numpy as np
import pickle

"""
En este bloque cargo el modelo entrenado, el objeto RFE y el escalador MinMaxScaler que guardé previamente en archivos .pkl:

    1. 'model_path' apunta al archivo que contiene el modelo de regresión logística entrenado. Lo cargo con pickle para poder reutilizarlo y hacer predicciones sin tener que reentrenarlo.

    2. 'rfe_path' contiene el archivo del objeto RFE, el cual se usó para seleccionar las características más importantes. Al cargarlo, me aseguro de aplicar las mismas transformaciones a los nuevos datos.

    3. 'scaler_path' apunta al archivo del escalador, que se usó para normalizar los datos. Lo cargo para garantizar que los datos futuros sean escalados de la misma forma que los datos de entrenamiento.
"""
model_path = '/Users/aguero/Desktop/A01754412-Portafolio_Implementaci-n_Framework/Model/logistic_model_rfe.pkl'
rfe_path = '/Users/aguero/Desktop/A01754412-Portafolio_Implementaci-n_Framework/Model/rfe_transformer.pkl'
scaler_path = '/Users/aguero/Desktop/A01754412-Portafolio_Implementaci-n_Framework/Model/scaler.pkl'

with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

with open(rfe_path, 'rb') as rfe_file:
    rfe = pickle.load(rfe_file)

with open(scaler_path, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)


"""
Aquí creo un arreglo NumPy llamado 'features', que contiene un conjunto de características previamente escaladas (normalizadas) que serán utilizadas para hacer una predicción con el modelo de regresión logística.

Los valores en este arreglo representan un conjunto de mediciones que corresponden a una situación en la que la calidad del aire es buena.

Este conjunto de características será ingresado al modelo para predecir si la calidad del aire es buena (1) o mala (0).
"""
"""
features = np.array([[0.2343438717484787, 0.9629080691788374, 0.7546155343748744, 0.8656603791107658, 0.8959211491318471, 0.7565152164393738, 0.7924601300347778, 0.2547923734351622, 0.0489159263896771, 0.0673856481784072, 0.0539250088719914, 0.2218700464732332, 0.0, 0.0, 0.0, 0.2333098756530392, 0.036363089175756,
                    0.3485216674026032, 0.210213941270986, 0.0877438661391807, 0.5694815607760904, 0.7717871554102588, 0.80439440249228, 0.8875362190202837, 0.0459683797598271, 0.2312954534461369, 0.2484252166252645, 0.2405081657963307, 0.8260394209253437, 0.773988137778487, 0.3556228133496051, 0.0, 0.3070491926854841]])
"""

"""
Este bloque crea un arreglo NumPy llamado 'features', que contiene un conjunto de características escaladas para hacer una predicción con el modelo de regresión logística.

Los valores en este arreglo representan mediciones que corresponden a una situación en la que la calidad del aire es mala.

Este conjunto de características será utilizado por el modelo para predecir si la calidad del aire es mala (0) o buena (1).
"""


features = np.array([[0.110673923409422, 0.9148332136747168, 0.8909984273579148, 0.8995179254995496, 0.8935876314134166, 0.8847968420808114, 0.8542722204215969, 0.4453182563203886, 0.12078067892359, 0.0430389068572398, 0.0813371111351689, 0.0703389359430209, 0.0, 0.0, 0.0, 0.5764024584471832, 0.5598051579455686,
                    0.2637290981774702, 0.2764028421549014, 0.1688909234253647, 0.2479540151231041, 0.8191677103269474, 0.844713039794972, 0.8875362190202837, 0.0459683797598271, 0.10069202866066, 0.1197493157112603, 0.1264384885344016, 0.5414854090473756, 0.858758586139794, 0.6429919760720377, 0.0, 0.5244725942753523]])


"""
En este bloque de código, se realiza la predicción de la calidad del aire utilizando los siguientes pasos:
    - 1. 'features_scaled': Primero, las características se escalan utilizando el escalador MinMaxScaler previamente cargado. Esto asegura que las nuevas características estén en el mismo rango que los datos de entrenamiento.
    
    - 2. 'features_selected': Luego, se seleccionan las características más importantes usando el objeto RFE cargado, que fue entrenado para identificar las características relevantes.

    - 3. 'prediction': Se realiza la predicción usando el modelo entrenado, aplicando las características seleccionadas. El resultado será un valor binario: 1 para "buena" calidad del aire y 0 para "mala" calidad del aire.

    - 4. Finalmente, se imprime el resultado de la predicción. Si el valor predicho es 1, se indica que la calidad del aire es buena; si es 0, se indica que la calidad del aire es mala.
"""

features_scaled = scaler.transform(features)
features_selected = rfe.transform(features_scaled)
prediction = model.predict(features_selected)


if prediction == 1:
    print("La calidad del aire es Buena.")
else:
    print("La calidad del aire es Mala.")
