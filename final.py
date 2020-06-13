# -*- coding: utf-8 -*-
"""
TRABAJO FINAL
Nombre Estudiantes: José María Borrás Serrano
                    Ismael Sánchez Torres
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
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import time


# Fijamos la semilla (19)
np.random.seed(1997)

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
   
   x = data[:, 1:]
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

#Preprocesamientos a utilizar
preprocesamientos = []

"""
#Preprocesamiento 0
preprocesamientos.append([('estandarizar', StandardScaler()),
                    ('var', VarianceThreshold(0.05)),
                    ('polinomio grado 2', PolynomialFeatures(2,interaction_only=True)),
                    ('estandarizar 2', StandardScaler()),
                    ('PCA', PCA(n_components=0.99)),
                    ('estandarizar 3', StandardScaler())])
"""

#Preprocesamiento 1
preprocesamientos.append([('var', VarianceThreshold()),
                          ('estandarizar', StandardScaler())])



print("Mostramos que el preprocesamiento ha sido efectivo.\n ")
print("Media de x_train antes del preprocesado: ", x_train.mean())
print("Varianza de x_train antes del preprocesado: ", x_train.std())
print("Número de características de cada dato antes del preprocesado: ", x_train.shape[1])
print("Matriz de correlación antes del preprocesado")
grafica_matriz_correlacion(x_train)

for i in range(len(preprocesamientos)):
    preprocesado = Pipeline(preprocesamientos[i])
    x_train_preprocesado = preprocesado.fit_transform(x_train)
    print("\nPreprocesamiento ", i)
    print("Media de x_train tras el preprocesado: ", x_train_preprocesado.mean())
    print("Varianza de x_train tras el preprocesado: ", x_train_preprocesado.std())
    print("Número de características de cada dato después del preprocesado: ", x_train_preprocesado.shape[1])
    print("Matriz de correlación después del preprocesado")
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
  
# Devuelve la clasificación del modelo que obtenga una mayor puntuación en Accuracy mediante validación cruzada.
# En el caso de que varios modelos empaten en la mejor puntuación,
# entonces el mejor modelo será aquel que menos tiempo haya necesitado.
def seleccionar_mejor_modelo(preprocesamientos, clasificaciones, parametros, x_train, y_train, mostrar_puntuacion=True,
                             mostrar_grafica=True):
    """Parámetros:
       preproceasmientos: array con los preprocesamientos que usar en el modelo
       clasificaciones: array con las clasificaciones que usar en el modelo
       x_train: conjunto de carácteristicas predictivas de los datos de entrenamiento
       y_train: etiquetas reales de los datos de entrenamiento
       mostrar_puntuacion: True para mostrar las puntuación de cada modelo, False para no mostrarlas
                           (True por defecto)
     """
    mejor_puntuacion = -1
    for i in range(len(clasificaciones)):
        for j in range(len(preprocesamientos)):
            # Cada clasificador está compuesto del preprocesamiento más la clasificación
            # Usamos GridSearchCV para quedarnos con el clasificador cuyos parámetros obtengan mejor resultado
            clasificador = GridSearchCV(Pipeline(preprocesamientos[j] + clasificaciones[i]), 
                                        parametros[i], scoring=('accuracy'), cv=5, n_jobs=-1).fit(x_train, y_train)
            # Obtenemos la puntuación y el tiempo
            puntuacion = clasificador.best_score_
            tiempo = clasificador.refit_time_
            # Si la puntuación es la mejor, actualizamos el mejor modelo
            if(puntuacion > mejor_puntuacion):
                mejor_puntuacion = puntuacion
                mejor_clasificador = clasificador
                mejor_tiempo = tiempo
            # Si la puntuación iguala a la mejor y ha tardado menos tiempo, actualizamos el mejor modelo
            if(puntuacion == mejor_puntuacion):
                if(tiempo < mejor_tiempo):
                    mejor_puntuacion = puntuacion
                    mejor_clasificador = clasificador
                    mejor_tiempo = tiempo
            # si es True, entonces mostramos la gráfica con la matriz de confusión
            if( mostrar_grafica):
                clasificador.fit(x_train, y_train)
                grafica_matriz_confusion(clasificador, x_train, y_train,
                                         "Matriz de confusión en el conjunto train del clasificador de {}".format( clasificaciones[i][0][0]))
            # si es True, entonces mostramos la puntuación y tiempo del modelo
            if( mostrar_puntuacion):
                print("Puntuación en el clasificador de {} con los parámetros {}".format( clasificaciones[i][0][0], clasificador.best_params_))
                print("Precisión: ",  puntuacion)
                print("Tiempo transcurrido (s): ", tiempo)
                print("\n")
                    
    return mejor_clasificador

#Clasificaciones a utilizar
clasificaciones = []
#Parámetros a probar el clasificador
parametros = []

clasificaciones.append([("Ridge", RidgeClassifier())])
parametros.append({'Ridge__alpha':[1, 10, 100], 'Ridge__solver':['saga'],'Ridge__max_iter':[500]})

clasificaciones.append([("RegresiónLogística",LogisticRegression())])
parametros.append({'RegresiónLogística__penalty':['l1', 'l2'], 
                   'RegresiónLogística__C':[1, 0.1, 0.01, 0.001],
                   'RegresiónLogística__solver':['saga'],
                   'RegresiónLogística__max_iter':[500],
                   'RegresiónLogística__multi_class':['ovr']})

clasificaciones.append([("SGD", SGDClassifier())])
parametros.append({'SGD__loss':['hinge'],
                   'SGD__penalty':['l1', 'l2'],
                   'SGD__max_iter':[500]})
    
clasificaciones.append([("SVM", SVC())])
parametros.append({'SVM__C':[1, 0.1, 0.01, 0.001],
                   'SVM__kernel':['rbf', 'poly']})

clasificaciones.append([("RandomForest", RandomForestClassifier())])
parametros.append({'RandomForest__n_estimators':[1, 2, 3, 4, 5, 10, 50, 100],
                   'RandomForest__criterion':['gini', 'entropy']})

"""
clasificaciones.append([("Perceptron multicapa", MLPClassifier(hidden_layer_sizes=(50, 1),max_iter=500))])

Puntuación en el clasificador de Perceptron multicapa con el preprocesamiento 0
Precisión: 0.999385 (+/- 0.001507)
Tiempo transcurrido (s):  17.988545179367065

Puntuación en el clasificador de Perceptron multicapa con el preprocesamiento 1
Precisión: 0.987998 (+/- 0.010687)
Tiempo transcurrido (s):  12.822283744812012
"""

"""
clasificaciones.append([("Boosting", GradientBoostingClassifier())])

Puntuación en el clasificador de Boosting con el preprocesamiento 0
Precisión: 0.998769 (+/- 0.002303)
Tiempo transcurrido (s):  61.28906488418579

Puntuación en el clasificador de Boosting con el preprocesamiento 1
Precisión: 1.000000 (+/- 0.000000)
Tiempo transcurrido (s):  1.681703805923462
"""

#Elegimos el mejor modelo (y mostramos las puntuaciones de cada modelo)
mejor_clasificador = seleccionar_mejor_modelo(preprocesamientos, clasificaciones, parametros, x_train, y_train)
print("\nEl mejor clasificador ha sido ", mejor_clasificador.best_estimator_)

input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
######## ESTIMACIÓN DE LOS ERRORES
###############################################################################

# Volvemos a entrenar pero esta vez usamos el conjunto train original (train+validación) para entrenar el modelo

print("Errores del mejor clasificador:")
mejor_clasificador.fit(x_train, y_train)

grafica_matriz_confusion(mejor_clasificador, x_test, y_test, "Matriz de confusión en el conjunto test")
print("Error en training: ", 1-puntuacion_precision(mejor_clasificador, x_train, y_train))
print("Error en test: ", 1-puntuacion_precision(mejor_clasificador, x_test, y_test))

print("\nEstimamos el error de Eout mediante validación cruzada")
puntuaciones = cross_validate(mejor_clasificador, x_train, y_train, scoring=('accuracy'), cv=5)
print("Media de Eval tras validación cruzada: %f (+/- %f)" % (1 - np.mean(puntuaciones['test_score']), puntuaciones['test_score'].std() * 2))