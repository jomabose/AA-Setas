# -*- coding: utf-8 -*-
"""
TRABAJO 3
Nombre Estudiante: José María Borrás Serrano
"""
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

# Fijamos la semilla (19)
np.random.seed(19)

###############################################################################
######## FUNCIONES DE VISUALIZACIÓN DE GŔAFICAS
###############################################################################

# Muestra una gŕafica de barras de la distribución de clases según las etiquetas
def grafica_distribucion_clases(y, titulo=None, num_clases=2):
   """Parámetros:
       y: etiquetas de los datos
       titulo: titulo del grafico (por defecto None)
       num_classes: número de clases distintas en los datos (por defecto 2)
     """

   fig, ax = plt.subplots()
   
   ax.bar(np.unique(y), np.bincount(y.astype(int)))
   ax.set_xticks(range(num_clases))
   ax.set_xlabel("clases")
   ax.set_ylabel("Nº ejemplos")
   
   if(titulo is not None):
       plt.title(titulo)
   plt.show()
   
# Muestra la matriz de correlación para un conjunto de datos
def grafica_matriz_correlacion(x, titulo=None):
   """Parámetros:
        x: conjunto de datos
        titulo: titulo del grafico (por defecto None)
   """
   fig, ax = plt.subplots()

   with np.errstate(invalid = 'ignore'):
      matriz_correlacion= np.abs(np.corrcoef(x, rowvar = False))
  
   ax.set_xlabel("características")
   ax.set_ylabel("características")
   im = ax.matshow(matriz_correlacion)
   fig.colorbar(im)

   if(titulo is not None):
       ax.title.set_text(titulo)
   plt.show()
   
# Muestra la matriz de confusión
def grafica_matriz_confusion(clasificador, x, y, titulo = None):
   """Parámetros:
        clasificador: modelo clasificador 
        x: conjunto de carácteristicas predictivas de los datos
        y: etiquetas de los datos
        titulo: titulo del grafico (por defecto None)
   """
   
   fig, ax = plt.subplots()
   disp = plot_confusion_matrix(clasificador, x, y, cmap=plt.cm.Blues, values_format = 'd', ax = ax)

   if(titulo is not None):
       disp.ax_.set_title(titulo)
   plt.show()

   plt.show()
   
###############################################################################
######## LECTURA DE LOS DATOS Y SEPARACIÓN EN TRAIN, VALIDACIÓN Y TEST
###############################################################################

# Funcion para leer los datos
def readData(file, delimiter=',', datatype = np.dtype(np.unicode)):
   """Parámetros:
       file: ruta del fichero con los datos
       delimiter: separador de los datos (por defecto ',')
       datatype: tipo de datos que leemos (por defecto se leen strings)
     """	
   data = np.genfromtxt(fname = file, dtype = datatype, delimiter = delimiter)
   
   data = np.delete(data, 11, 1) # Eliminamos la característica de la columna 11, porque le faltan el 30.5% de los valores
   data = OrdinalEncoder().fit_transform(data).astype('float64') # Pasamos los valores nominales a ordinales 
   
   x = data[:, 1:-1]
   y = data[:, 0]
	
   return x, y

# Funcion ver el porcentaje de valores perdidos en cada característica
def valoresPerdidos(data):
   """Parámetros:
       datos: datos
   """
    
   num_muestras, num_caracteristicas = data.shape
   valores_perdidos = np.zeros(num_caracteristicas)
   
   for elemento in data:
       for i in range(len(elemento)):
           if(elemento[i] == '?'):
               valores_perdidos[i] += 1
        
   porcentaje_valores_perdidos = np.zeros(num_caracteristicas)
   for i in range(num_caracteristicas):
       porcentaje_valores_perdidos[i] = 100*valores_perdidos[i]/num_muestras
       
   print(porcentaje_valores_perdidos)

# Lectura de los datos 
x_data, y_data = readData("./datos/agaricus-lepiota.data")


# Separación de los datos de entrenamiento en entrenamiento (80%) y test (20%)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, stratify=y_data)

print("Mostramos la distribución de clases en cada conjunto: \n")
grafica_distribucion_clases(y_train, "Distribución de clases en Entrenamiento")
grafica_distribucion_clases(y_test, "Distribución de clases en Test")

print("Porcentaje de datos en entrenamiento (%): ", 100*len(x_train)/(len(x_data)))
print("Porcentaje de datos en test (%): ", 100*len(x_test)/(len(x_data)))

input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
######## PREPROCESADO
###############################################################################

print("Preprocesado\n")

# Preprocesamiento a realizar, usamos Pipeline para encadenar varios procesos

preprocesamiento = [('estandarizar', StandardScaler()),
                    ('var', VarianceThreshold(0.05)),
                    ('polinomio grado 2', PolynomialFeatures(2,interaction_only=True)),
                    ('estandarizar 2', StandardScaler()),
                    ('PCA', PCA(n_components=0.95)),
                    ('estandarizar 3', StandardScaler())]

preprocesado = Pipeline(preprocesamiento)

# Realizamos un preprocesado de prueba ara comprobar que es efectivo
x_train_preprocesado = preprocesado.fit_transform(x_train)

print("Mostramos que el preprocesamiento ha sido efectivo.\n ")
print("Media de x_train antes del preprocesado: ", x_train.mean())
print("Varianza de x_train antes del preprocesado: ", x_train.std())
print("Número de características de cada dato antes del preprocesado: ", x_train.shape[1])
print("Media de x_train tras el preprocesado: ", x_train_preprocesado.mean())
print("Varianza de x_train tras el preprocesado: ", x_train_preprocesado.std())
print("Número de características de cada dato después del preprocesado: ", x_train_preprocesado.shape[1])
print("\nMatriz de correlación antes del preprocesado")
grafica_matriz_correlacion(x_train)
print("\nMatriz de correlación después del preprocesado")
grafica_matriz_correlacion(x_train_preprocesado)

input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
######## CLASIFICACIÓN Y ELECCIÓN DEL MODELO
###############################################################################

print("Clasificación\n")

# Devuelve puntuación del clasificador mediante la métrica accuracy al predecir etiquetas 
def puntuacion_precision(clasificador, x, y):
    """Parámetros:
       clasificador: clasificador ya entrenado
       x: conjunto de carácteristicas predictivas de los datos
       y: etiquetas reales de los datos
     """
    y_pred = clasificador.predict(x)
    return accuracy_score(y, y_pred)
  
# Devuelve la clasificación del modelo que obtenga una mayor puntuación en Accuracy      
def seleccionar_mejor_modelo(clasificaciones, x_train, y_train, x_val, y_val, mostrar_puntuacion=True):
    """Parámetros:
       clasificaciones: array con las clasificaciones que usar en el modelo
       x_train: conjunto de carácteristicas predictivas de los datos de entrenamiento
       y_train: etiquetas reales de los datos de entrenamiento
       x_val: conjunto de carácteristicas predictivas de los datos de validación
       y_val: etiquetas reales de los datos de validación
       mostrar_puntuacion: True para mostrar las puntuaciones de cada modelo, False para no mostrarlas
                           (True por defecto)
     """
    mejor_puntuacion = 0
    indice_mejor_modelo = 0 
    for i in range(len(clasificaciones)):
        # Cada clasificador está compuesto del preprocesamiento más la clasificación
        clasificador = Pipeline(preprocesamiento + clasificaciones[i])
        # Entrenamos el modelo según los datos de entrenamiento
        clasificador.fit(x_train, y_train)
        # Obtenemos las puntuaciones 
        puntuacion_train = puntuacion_precision(clasificador, x_train, y_train)
        puntuacion_val = puntuacion_precision(clasificador, x_val, y_val)
        # Si la puntuación es la mejor, actualizamos el mejor modelo
        if(puntuacion_val > mejor_puntuacion):
            mejor_puntuacion = puntuacion_val
            indice_mejor_modelo = i
        # si es True, entonces mostramos las puntuaciones del modelo
        if( mostrar_puntuacion):
            print("Puntuación en el clasificador de ", clasificaciones[i][0][0])
            print("Precisión en training: ", puntuacion_train)
            print("Precisión en validación: ", puntuacion_val)
            print("\n")
        
    return clasificaciones[indice_mejor_modelo]

#Clasificaciones a utilizar
clasificaciones = []

clasificaciones.append([("Ridge", 
                         RidgeClassifier(solver='sag',
                                         max_iter=500))])

clasificaciones.append([("Regresión Logística",
                         LogisticRegression(penalty='l2',
                                            solver='sag',
                                            max_iter=500,
                                            multi_class='multinomial'))])
    
clasificaciones.append([("SGD", SGDClassifier(loss = 'hinge',
                                              penalty = 'l2',
                                              max_iter=500))])

#Elegimos el mejor modelo (y mostramos las puntuaciones de cada modelo)
mejor_clasificacion = seleccionar_mejor_modelo(clasificaciones, x_train, y_train, x_test, y_test)
print("\nEl mejor clasificador ha sido ", mejor_clasificacion[0])

input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
######## ESTIMACIÓN DE LOS ERRORES
###############################################################################

# Volvemos a entrenar pero esta vez usamos el conjunto train original (train+validación) para entrenar el modelo
print("Ahora entrenamos el mejor clasificador con el conjunto de training+validación:")
mejor_clasificador = Pipeline(preprocesamiento + mejor_clasificacion)
mejor_clasificador.fit(x_train, y_train)

print("Error en training+validación: ", 1-puntuacion_precision(mejor_clasificador, x_train, y_train))
print("Error en test: ", 1-puntuacion_precision(mejor_clasificador, x_test, y_test))
grafica_matriz_confusion(mejor_clasificador, x_test, y_test, "Matriz de confusión en el conjunto test")

print("\nEstimamos el error de Eout mediante validación cruzada")
puntuaciones = cross_validate(mejor_clasificador, x_train, y_train, scoring=('accuracy'), cv=5)
print("Media de Eval tras validación cruzada: ", 1 - np.mean(puntuaciones['test_score']))
