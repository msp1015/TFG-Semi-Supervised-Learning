"""Este módulo contiene la implementación del algoritmo CoForest

@Autor:     Mario Sanz Pérez
@Fecha:     13/03/2024
@Versión:   1.1
@Nombre:    CoForest.py
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier

class CoForest:
    """Algoritmo CoForest para el aprendizaje semi-supervisado.
    Basado en el estudio realizado por Zhou y Li en 2007:
    "Improve Computer-aided Diagnosis with Machine Learning Techniques
    Using Undiagnosed Samples"
    """

    def __init__(self, n, theta, random_state=None):
        """Constructor de la clase CoForest
        Inicializa los parámetros necesarios para su ejecución.
        También se inicializan las variables necesarias para el seguimiento de
        los resultados.

        Args:
            n (int): número de árboles de decisión que componen el bosque.
            theta (float): umbral de confianza para pseudo datos.
            random_state (int, optional): semilla para la generación de números
            aleatorios. Por defecto, None.
        """

        self.n = n
        self.theta = theta

        self.errores = {}
        self.confianzas = {}
        self.bosque = {}
        self.clases = []
        self.datos_arbol_ind = {}
        self.L = None
        self.y_l = None
        self.U = None

        self.estado_aleatorio = random_state
        self.rng = np.random.RandomState(self.estado_aleatorio)
        #Para probar la validez del algoritmo
        self.accuracy_por_iteracion = []

    def fit(self, L, y_l, U, X_test, y_test):
        """Entrena el algoritmo CoForest con los datos de entrenamiento,
        tanto etiquetados como no etiquetados.

        Args:
            L (array): Datos de entrenamiento etiquetados.
            y_l (array): Etiquetas de los datos de entrenamiento etiquetados.
            U (array): Datos de entrenamiento no etiquetados.
            X_test (array): Datos de prueba.
            y_test (array): Etiquetas de los datos de prueba.

        Returns:
            dict: Un diccionario que contiene los árboles de decisión entrenados.
        """
        #segun los datos en y_l podemos sacar el numero de clases únicas
        self.clases = np.unique(y_l)
        self.L = L
        self.y_l = y_l
        self.U = U

        for i in range(self.n):
            # Se inicializa el bosque con n arboles de decision
            self.bosque[i] = DecisionTreeClassifier(
                max_features="log2",
                random_state=self.rng)
            L_i, y_l_i, indices_datos= self.bootstrap(L, y_l)
            self.datos_arbol_ind[i] = indices_datos
            self.bosque[i].fit(L_i, y_l_i)
            self.errores[i] = [0.5]
            self.confianzas[i] = [min(0.1 * len(L), 100)]

        t = 0
        e = self.errores
        W = self.confianzas
        hay_cambios = True

        while hay_cambios:
            #Seguimiento de los resultados, empezando en la iteración t=0
            self.accuracy_por_iteracion.append(self.score(X_test, y_test))
            t = t + 1
            hay_cambios_en_arbol = [False] * self.n
            pseudo_datos_etiquetados = {}

            for i, arbol_Hi in self.bosque.items():
                error_actual = self.estimar_error(arbol_Hi, L, y_l,
                                                  self.datos_arbol_ind)
                W_actual = W[i][t-1]
                pseudo_datos = []
                pseudo_etiquetas_datos = []
                if error_actual < e[i][t-1]:
                    if error_actual == 0:
                        Wmax = self.theta * U.shape[0]
                    else:
                        Wmax = e[i][t-1] * W[i][t-1] / error_actual

                    U_muestras = self.submuestrear(arbol_Hi, U, Wmax)
                    W_actual = 0
                    for x_u in U_muestras:
                        confianza, most_agreed_class = self.calcula_confianza(arbol_Hi, U[x_u, :])
                        if confianza > self.theta:
                            hay_cambios_en_arbol[i] = True
                            pseudo_datos.append(U[x_u, :])
                            pseudo_etiquetas_datos.append(most_agreed_class)
                            W_actual += confianza

                e[i].append(error_actual)
                W[i].append(W_actual)
                pseudo_datos_etiquetados[i] = (pseudo_datos, pseudo_etiquetas_datos)

            for i, arbol_Hi in self.bosque.items():
                if hay_cambios_en_arbol[i]:
                    if e[i][t] * W[i][t] < e[i][t-1] * W[i][t-1]:
                        self.reentrenar_arbol(i, L, y_l, pseudo_datos_etiquetados[i],
                                              self.datos_arbol_ind[i])

            if not any(hay_cambios_en_arbol):
                hay_cambios = False

        self.errores = e
        self.confianzas = W

        return self.bosque

    def bootstrap(self, L, y_l, p=0.7):
        """Genera un conjunto de datos aleatorios con reemplazamiento.
        Utiliza el random state de la clase para garantizar la reproducibilidad.

        Args:
            L (array): Datos de entrenamiento etiquetados.
            y_l (array): Vector de etiquetas correspondientes a los datos de entrada.
            p (float, opcional): Proporción de datos a generar en el conjunto de datos 
                                aleatorios. Valor predeterminado es 0.7.

        Returns:
            tuple: Una tupla que contiene el conjunto de datos aleatorios (L_i),
            las etiquetas correspondientes (y_l_i) y los índices de los datos
            seleccionados (datos_aleatorios) sobre el conjunto L.
        """

        datos_aleatorios = self.rng.choice(L.shape[0], size=int(p*L.shape[0]), replace=True)
        L_i = L[datos_aleatorios, :]
        y_l_i = y_l[datos_aleatorios]

        return L_i, y_l_i, datos_aleatorios

    def estimar_error(self, Hi, L, y_l, datos_arbol_indiv):
        """Calcula el error estimado para una muestra dada.
        El error calculado es el Out Of Bag (OOB) error.

        Args:
            Hi (DecisionTreeClassifier): El árbol de decisión Hi.
            L (lista): Lista de muestras de entrenamiento.
            y_l (lista): Lista de etiquetas de los datos de entrenamiento.
            datos_arbol_indiv (diccionario): Diccionario que contiene las filas
            de entrenamiento utilizadas por cada árbol.

        Returns:
            float: El error estimado para la muestra dada.
        """
        errores = []
        L = np.array(L)
        for muestra, etiqueta in zip(L, y_l):
            n_votos = 0
            n_aciertos = 0
            for i, arbol in self.bosque.items():
                filas_entrenadas = L[datos_arbol_indiv[i]]
                usado_en_entrenamiento = np.any(np.all(muestra == filas_entrenadas,
                                                        axis=1))
                if arbol is not Hi and not usado_en_entrenamiento:
                    if arbol.predict([muestra])[0] == etiqueta:
                        n_aciertos += 1
                    n_votos += 1
            if n_votos > 0:
                errores.append(1 - (n_aciertos / n_votos))

        return np.mean(errores)

    def submuestrear(self, Hi, U, W):
        """Submuestrea una matriz de datos no etiquetados.

        Args:
            Hi (DecisionTreeClassifier): El árbol de decisión Hi.
            U (array): Matriz de características de los datos no etiquetados.
            W (float): Peso máximo total permitido para el submuestreo.

        Returns:
            array: Matriz de índices de las filas submuestreadas.
        """

        W_i = 0
        U_submuestreado = []

        while W_i < W:
            fila_aleatoria = self.rng.choice(U.shape[0])
            W_i += self.calcula_confianza(Hi, U[fila_aleatoria, :])[0]
            U_submuestreado.append(fila_aleatoria)

        return np.array(U_submuestreado)

    def calcula_confianza(self, Hi, sample):
        """
        Calcula la confianza de la predicción del conjunto concominante
        del árbol Hi.

        Args:
            Hi (DecisionTreeClassifier): El árbol de decisión Hi del bosque.
            sample (array): La muestra de entrada para realizar la predicción.

        Raises:
            ValueError: Se lanza si el modelo no ha sido ajustado.

        Returns:
            float: El valor de confianza de la predicción de Hi.
            int: La clase más predicha por los árboles del bosque.
        """
        #comprobar si hay elementos en el vector clases
        if not np.any(self.clases):
            raise ValueError("No se ha ajustado el modelo")
        contador = {i: 0 for i in self.clases}

        for arbol in self.bosque.values():
            if arbol is not Hi:
                contador[arbol.predict([sample])[0]] += 1

        max_votos = max(contador.values())
        clase_mas_predicha = max(contador, key=contador.get)

        return max_votos / (len(self.bosque) - 1), clase_mas_predicha

    def reentrenar_arbol(self, i, L, y_l, pseudo_datos_etiquetados, datos_arbol_indiv):
        """Reentrena un árbol específico del bosque con los pseudo datos.

        Args:
            i (int): Índice del árbol a reentrenar.
            L (array): Datos de entrenamiento etiquetados.
            y_l (array): Etiquetas del conjunto L.
            pseudo_datos_etiquetados (tuple): Conjunto de datos y etiquetas de los
                                                pseudo datos.
            datos_arbol_indiv (array): Índices de los datos utilizados 
                                        para entrenar el árbol individual.
        """

        X = np.concatenate((L[datos_arbol_indiv], pseudo_datos_etiquetados[0]))
        Y = np.concatenate((y_l[datos_arbol_indiv], pseudo_datos_etiquetados[1]))
        self.bosque[i] = self.bosque[i].fit(X, Y)

    def obtener_errores(self):
        """Obtiene los errores que han ido produciendo los árboles
        en cada iteración.

        Returns:
            dict: Un diccionario con los errores del modelo.
        """
        return self.errores

    def obtener_confianzas(self):
        """Obtiene las confianzas que han ido produciendo los árboles
            en cada iteración.

            Returns:
                dict: Un diccionario con las confianzas del modelo.
            """
        return self.confianzas

    def prediccion_unica(self, sample):
        """
        Predice la etiqueta de clase para una muestra individual.

        Args:
            sample (array): La muestra de entrada a clasificar.

        Returns:
            Int: La etiqueta de clase predicha para la muestra de entrada.
        """
        count = {i: 0 for i in self.clases}
        for i in (tree.predict([sample])[0] for tree in self.bosque.values()):
            count[i] += 1
        return max(count, key=count.get)

    def predict(self, samples):
        """Realiza predicciones para las muestras dadas.

        Args:
            samples (array): Las muestras de entrada para las cuales se realizarán
                            las predicciones.

        Returns:
            array: Un array con las predicciones correspondientes a cada muestra de entrada.
        """
        samples = (lambda x: np.expand_dims(x, axis=0) if x.ndim == 1 else x)(
            samples
        )
        return np.array([self.prediccion_unica(sample) for sample in samples])

    def score(self, X_test, y_test):
        """Calcula la precisión del modelo en los datos de prueba.

        Args:
            X_test (array): Los datos de prueba para realizar las predicciones.
            y_test (array): Las etiquetas verdaderas correspondientes a los datos de prueba.

        Returns:
            float: La precisión del modelo en los datos de prueba.
        """
        y_predicciones = self.predict(X_test)
        return np.count_nonzero(y_predicciones == y_test) / len(y_test)

    def get_accuracy_por_iteracion(self):
        """Devuelve la precisión por iteración.

        Devuelve:
            list: Una lista de valores de precisión por iteración.
        """
        return self.accuracy_por_iteracion
