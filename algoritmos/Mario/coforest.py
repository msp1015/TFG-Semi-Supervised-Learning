'''
@Autor:     Mario Sanz Pérez	
@Fecha:     21/02/2024
@Versión:   1.0
@Nombre:    CoForest.py
'''

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

class CoForest:
    '''
    Algoritmo de aprendizaje semi-supervisado CoForest
    '''
    def __init__(self, n, theta, random_state=None):
        '''
        Constructor de la clase CoForest
        
        El parametro random_state por defecto esta a None, 
        de tal manera conseguimos datos aleatorios en cada ejecucion.
        
        Parameters:
            n: int
                Numero de arboles de decision
            theta: float
                Umbral de confianza
        '''
        self.n = n  #numero de arboles de decision
        self.theta = theta #umbral de confianza
        self.bosque = {} #diccionario para el random forest
        self.clases = [] #lista de clases unicas
        self.datos_arbol_ind = {} #diccionario para los datos de entrenamiento de cada arbol
        self.errores = {} #diccionario para los errores de cada arbol
        self.confianzas = {} #diccionario para las confianzas de cada arbol

        self.L = None
        self.y_l = None
        self.U = None
        
        self.estado_aleatorio = random_state
        self.rng = np.random.RandomState(self.estado_aleatorio)
        #Para probar la validez del algoritmo
        self.accuracy_por_iteracion = []

    def fit(self, L, y_l, U, X_test, y_test):
        '''
        Ajusta el modelo a los datos de entrenamiento
        '''
        #segun los datos en y_l podemos sacar el numero de clases
        self.clases = np.unique(y_l)
        self.L = L
        self.y_l = y_l
        self.U = U
        
        for i in range(self.n):
            # Se inicializa el bosque con n arboles de decision
            self.bosque[i] = DecisionTreeClassifier(
                max_features = "log2",
                random_state=self.estado_aleatorio)
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
            t = t + 1
            hay_cambios_en_arbol = [False] * self.n
            pseudo_datos_etiquetados = {} # Diccionario para los datos que se etiquetaran

            for i, arbol_Hi in self.bosque.items():
                error_actual = self.estimar_error(arbol_Hi, L, y_l, self.datos_arbol_ind[i])
                W_actual = W[i][t-1]
                pseudo_datos = [] # Para cada arbol se almacenan los datos que se etiquetaran
                pseudo_etiquetas_datos = [] #Etiquetas que acompañan a los datos que se etiquetaran

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


            self.accuracy_por_iteracion.append(self.score(X_test, y_test))

        self.errores = e
        self.confianzas = W


        return self.bosque

    def bootstrap(self, L, y_l, p=0.7):
        '''
        La tecnica del bootstrapping es una tecnica de remuestreo que consiste en seleccionar
        aleatoriamente una muestra de datos de un conjunto de datos, con reemplazamiento.
        '''
        #generar indices aleatorios con reemplazamiento
        
        datos_aleatorios = self.rng.choice(L.shape[0], size=int(p*L.shape[0]), replace=True)
        L_i = L[datos_aleatorios, :]
        y_l_i = y_l[datos_aleatorios]

        return L_i, y_l_i, datos_aleatorios

    def estimar_error(self, Hi, L, y_l, datos_arbol_indiv):
        '''
        Estima el error del arbol Hi. Out Of Bag Error
        '''
        errores = []
        L = np.array(L)
        for muestra, etiqueta in zip(L, y_l):
            n_votos = 0
            n_aciertos = 0

            for arbol in self.bosque.values():
                filas_entrenadas = L[datos_arbol_indiv]

                usado_en_entrenamiento = np.any(np.all(muestra.reshape(1, -1) == filas_entrenadas, axis=1))

                if arbol is not Hi and not usado_en_entrenamiento:
                    if arbol.predict([muestra])[0] == etiqueta:
                        n_aciertos += 1
                    n_votos += 1

            if n_votos > 0:
                errores.append(1 - (n_aciertos / n_votos))

        return np.mean(errores)

    def submuestrear(self, Hi, U, W):
        '''
        Submuestreo de U
        '''
        W_i = 0
        U_submuestreado = []
        
        while W_i < W:
            fila_aleatoria = self.rng.choice(U.shape[0])
            W_i += self.calcula_confianza(Hi, U[fila_aleatoria, :])[0]
            U_submuestreado.append(fila_aleatoria)

        return np.array(U_submuestreado)


    def calcula_confianza(self, Hi, sample):
        '''
        '''
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
        '''
        Reeduca el arbol i con nuevos pseudo datos
        '''
        X = np.concatenate((L[datos_arbol_indiv], pseudo_datos_etiquetados[0]))
        Y = np.concatenate((y_l[datos_arbol_indiv], pseudo_datos_etiquetados[1]))
        self.bosque[i] = self.bosque[i].fit(X, Y)

    def obtener_errores(self):
        return self.errores

    def obtener_confianzas(self):
        return self.confianzas

    def prediccion_unica(self, sample):

        count = {i: 0 for i in self.clases}
        for i in (tree.predict([sample])[0] for tree in self.bosque.values()):
            count[i] += 1
        return max(count, key=count.get)

    def predict(self, samples):
        samples = (lambda x: np.expand_dims(x, axis=0) if x.ndim == 1 else x)(
            samples
        )
        return np.array([self.prediccion_unica(sample) for sample in samples])

    def score(self, X_test, y_test):
        y_predicciones = self.predict(X_test)

        return np.count_nonzero(y_predicciones == y_test) / len(y_test)

    def get_accuracy_por_iteracion(self):
        return self.accuracy_por_iteracion



