# Autor: David Martínez Acha
# Fecha: 11/02/2023 14:15
# Descripción: Divide los datos para los algoritmos
# Version: 1.2
import numpy as np
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from algoritmos.utilidades.common import obtain_train_unlabelled


def data_split(x: DataFrame, y: DataFrame, is_unlabelled, p_unlabelled=0.8, p_test=0.2, is_inductive=True):
    """
    A partir de todos los datos con el nombre de sus características
    crea un conjunto de entrenamiento (con datos etiquetados y no etiquetados) y el conjunto de test.
    Si el conjunto ya tiene no etiquetados, simplemente dividirá en conjunto de test

    :param x: instancias de entrenamiento.
    :param y: etiquetas de las instancias.
    :param is_unlabelled: indica si el conjunto de datos ya contiene no etiquetados.
    :param p_unlabelled: porcentaje no etiquetados.
    :param p_test: porcentaje de test.
    :return: El conjunto de entrenamiento (x_train, y_train) que incluye datos no etiquetados
            y el conjunto de test (x_test, y_test)
    """

    x = np.array(x)
    y = np.array(y).ravel()

    if not is_unlabelled:
        x_train, x_u, y_train, y_u = train_test_split(x, y, test_size=p_unlabelled, stratify=y)
    else:
        x_train, y_train, x_u, y_u = obtain_train_unlabelled(x, y)

    if is_inductive:
        # En caso de ser inductivo, tambien habra conjunto de test
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=p_test,
                                                            stratify=y_train)
    
        x_train = np.append(x_train, x_u, axis=0)
        y_train = np.append(y_train, [-1] * len(x_u))

        return x_train, y_train, x_test, y_test
    else:
        # Si es GSSL, no habra conjunto de test
        # La evaluacion se hara con el conjunto de no etiquetados
        return x_train, y_train, x_u, y_u
