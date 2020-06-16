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
from sklearn.dummy import DummyClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import plot_roc_curve

# Fijamos la semilla (1997)
semilla=1997
np.random.seed(semilla)

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

   # Número de parejas de características cuya correlación es mayor que 0.3
   num = 0
   for i in range(matriz_correlacion.shape[0]):
       for j in range(i+1, matriz_correlacion.shape[1]):
           if(matriz_correlacion[i][j] > 0.3):
               num += 1
               
   ax.set_xlabel("características")
   ax.set_ylabel("características")
   im = ax.matshow(matriz_correlacion)
   fig.colorbar(im)

   if(titulo is not None):
       ax.title.set_text(titulo)
   plt.show()
   
   print("Número de parejas de características distintas cuya correlación es mayor que 0.3: ", num)

   
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

"""
# Muestra la gŕafica Precisión-Recall
def grafica_precision_recall(clasificador, x, y, titulo = '2-class Precision-Recall curve'):
   "Parámetros:
        clasificador: modelo clasificador 
        x: conjunto de carácteristicas predictivas de los datos
        y: etiquetas de los datos
        titulo: titulo del grafico (por defecto '2-class Precision-Recall curve')
   "
   
   disp = plot_precision_recall_curve(clasificador, x, y)

   if(titulo is not None):
       disp.ax_.set_title(titulo)
       
   plt.show()       


# Muestra la gŕafica curva ROC    
def grafica_roc(clasificador, x, y, titulo = 'Curva ROC'):
   "Parámetros:
        clasificador: modelo clasificador 
        x: conjunto de carácteristicas predictivas de los datos
        y: etiquetas de los datos
        titulo: titulo del grafico (por defecto 'Curva ROC')
   "

   disp = plot_roc_curve(clasificador, x, y)

   if(titulo is not None):
       disp.ax_.set_title(titulo)
       
   plt.show()
"""
   
# Código sacado de https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py
# Muestras las gráficas de la curva de aprendizaje
def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    axes : array of 3 axes, optional (default=None)
        Axes to use for plotting the curves.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(3, 1, figsize=(8, 15))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Ejemplos de entrenamiento")
    axes[0].set_ylabel("Puntuación")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Puntuación de entrenamiento")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Puntuación de validación cruzada")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Ejemplos de entrenamiento")
    axes[1].set_ylabel("Tiempos de ajuste")
    axes[1].set_title("Escalabilidad del modelo")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("Tiempos de ajuste")
    axes[2].set_ylabel("Puntuación")
    axes[2].set_title("Rendimiento del modelo")

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

preprocesamiento = ([('var', VarianceThreshold(0.05)),
                     ('estandarizar', StandardScaler())])



print("Mostramos que el preprocesamiento ha sido efectivo.\n ")
print("Media de x_train antes del preprocesado: ", x_train.mean())
print("Varianza de x_train antes del preprocesado: ", x_train.std())
print("Número de características de cada dato antes del preprocesado: ", x_train.shape[1])
print("Matriz de correlación antes del preprocesado")
grafica_matriz_correlacion(x_train)

preprocesado = Pipeline(preprocesamiento)
x_train_preprocesado = preprocesado.fit_transform(x_train)
print("\nMedia de x_train tras el preprocesado: ", x_train_preprocesado.mean())
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
def seleccionar_mejor_modelo(preprocesamiento, clasificaciones, parametros, x_train, y_train, mostrar_puntuacion=True,
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
        # Cada clasificador está compuesto del preprocesamiento más la clasificación
        # Usamos GridSearchCV para quedarnos con el clasificador cuyos parámetros obtengan mejor resultado
        clasificador = GridSearchCV(Pipeline(preprocesamiento + clasificaciones[i]), 
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
            grafica_matriz_confusion(clasificador.best_estimator_, x_train, y_train,
                                     "Matriz de confusión en el conjunto train del clasificador de {}".format( clasificaciones[i][0][0]))
            """
            grafica_precision_recall(clasificador.best_estimator_, x_train, y_train,
                                     "Gráfica Precision-Recall en el conjunto train del clasificador de {}".format( clasificaciones[i][0][0]))
            grafica_roc(clasificador, x_train, y_train,
                            "Gráfica Curva ROC en el conjunto train del clasificador de {}".format( clasificaciones[i][0][0]))
            """
            plot_learning_curve(clasificador.best_estimator_, "Curvas de Aprendizaje del clasificador de {}".format( clasificaciones[i][0][0]), x_train, y_train)
        # si es True, entonces mostramos la puntuación y tiempo del modelo
        if( mostrar_puntuacion):
            print("Puntuación en el clasificador de {} con los parámetros {}".format( clasificaciones[i][0][0], clasificador.best_params_))
            print("Precisión: ",  puntuacion)
            print("Tiempo transcurrido (s): ", tiempo)
            print("\n")
            
    return mejor_clasificador.best_estimator_

#Clasificaciones a utilizar
clasificaciones = []
#Parámetros a probar el clasificador
parametros = []

"""
clasificaciones.append([("Ridge", RidgeClassifier())])
parametros.append({'Ridge__alpha':[1, 10, 100],
                   'Ridge__solver':['saga'],
                   'Ridge__max_iter':[500],
                   'Ridge__random_state':[semilla]})
"""

# Modelo Dummy para comparar con los otros modelos
clasificaciones.append([('Dummy',DummyClassifier())])
parametros.append({})

# Modelos Lineales
clasificaciones.append([("RegresiónLogística",LogisticRegression())])
parametros.append({'RegresiónLogística__penalty':['l1', 'l2'], 
                   'RegresiónLogística__C':[1, 0.1, 0.01, 0.001],
                   'RegresiónLogística__solver':['saga'],
                   'RegresiónLogística__max_iter':[500],
                   'RegresiónLogística__multi_class':['ovr'],
                   'RegresiónLogística__random_state':[semilla]})

clasificaciones.append([("SGD", SGDClassifier())])
parametros.append({'SGD__loss':['hinge'],
                   'SGD__penalty':['l1', 'l2'],
                   'SGD__alpha':[0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001],
                   'SGD__max_iter':[500],
                   'SGD__random_state':[semilla]})
    
# Modelos no lineales
clasificaciones.append([("SVM", SVC())])
parametros.append({'SVM__C':[1, 0.1, 0.01, 0.001],
                   'SVM__kernel':['rbf', 'poly'],
                   'SVM__random_state':[semilla]})

clasificaciones.append([("RandomForest", RandomForestClassifier())])
parametros.append({'RandomForest__n_estimators':[1, 2, 3, 4, 5, 10, 50, 100],
                   'RandomForest__criterion':['gini', 'entropy'],
                   'RandomForest__random_state':[semilla]})

clasificaciones.append([("Perceptron", MLPClassifier(hidden_layer_sizes=(50, 1),max_iter=500))])
parametros.append({'Perceptron__hidden_layer_sizes':[(50,), (60,), (70,), (80,), (90,), (100,)],
                   'Perceptron__max_iter':[500],
                   'Perceptron__random_state':[semilla]})
    
clasificaciones.append([("Boosting", GradientBoostingClassifier())])
parametros.append({'Boosting__learning_rate':[0.5, 0.25, 0.1, 0.05, 0.01],
                   'Boosting__n_estimators':[10, 50, 100, 200, 500],
                   'Boosting__random_state':[semilla]})

#Elegimos el mejor modelo (y mostramos las puntuaciones de cada modelo)
mejor_clasificador = seleccionar_mejor_modelo(preprocesamiento, clasificaciones, parametros, x_train, y_train)
print("\nEl mejor clasificador ha sido ", mejor_clasificador)

input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
######## ESTIMACIÓN DE LOS ERRORES
###############################################################################

# Volvemos a entrenar pero esta vez usamos el conjunto train original (train+validación) para entrenar el modelo


#mejor_clasificador.fit(x_train, y_train)

"""
grafica_precision_recall(mejor_clasificador, x_test, y_test, "Gráfica Precision-Recall en el conjunto test")
grafica_roc(mejor_clasificador, x_test, y_test, "Gráfica Curva ROC en el conjunto test")
plot_learning_curve(mejor_clasificador, "Curvas de Aprendizaje del mejor clasificador", x_train, y_train)
"""   
         
print("Errores del mejor clasificador:")
grafica_matriz_confusion(mejor_clasificador, x_test, y_test, "Matriz de confusión en el conjunto test")
print("Error en training: ", 1-puntuacion_precision(mejor_clasificador, x_train, y_train))
print("Error en test: ", 1-puntuacion_precision(mejor_clasificador, x_test, y_test))

print("\nEstimamos el error de Eout mediante validación cruzada")
puntuaciones = cross_validate(mejor_clasificador, x_train, y_train, scoring=('accuracy'), cv=5)
print("Media de Eval tras validación cruzada: %f (+/- %f)" % (1 - np.mean(puntuaciones['test_score']), puntuaciones['test_score'].std() * 2))