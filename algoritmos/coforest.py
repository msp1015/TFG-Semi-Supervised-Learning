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
    def __init__(self, n, theta):
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
        self.forest = {} #diccionario para el random forest
        self.clases = [] #lista de clases unicas
        self.datos_arbol_ind = {} #diccionario para los datos de entrenamiento de cada arbol
        self.errores = {} #diccionario para los errores de cada arbol
        self.confianzas = {} #diccionario para las confianzas de cada arbol
        
        self.L = None
        self.y_l = None
        self.U = None
        
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
            self.forest[i] = DecisionTreeClassifier()
            L_i, y_l_i, indices_datos= self.bootstrap(L, y_l)
            self.datos_arbol_ind[i] = indices_datos
            self.forest[i].fit(L_i, y_l_i)
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
            
            for i, Hi in self.forest.items():
                error_actual = self.estimate_error(Hi, L, y_l, self.datos_arbol_ind[i])
                W_actual = W[i][t-1]
                pseudo_datos = [] # Para cada arbol se almacenan los datos que se etiquetaran
                pseudo_etiquetas_datos = [] # Etiquetas que acompañan a los datos que se etiquetaran
                
                if error_actual < e[i][t-1]:
                    
                    if error_actual == 0:
                        Wmax = self.theta * U.shape[0]
                    else:
                        Wmax = e[i][t-1] * W[i][t-1] / error_actual
                    U_samples = self.subsample(Hi, U, Wmax)
                    W_actual = 0
                    for x_u in U_samples:
                        confidence, most_agreed_class = self.confidence(Hi, U[x_u, :])
                        
                        if confidence > self.theta:
                            hay_cambios_en_arbol[i] = True
                            pseudo_datos.append(U[x_u, :])
                            pseudo_etiquetas_datos.append(most_agreed_class)
                            
                            W_actual += confidence
                e[i].append(error_actual)
                W[i].append(W_actual)                
                pseudo_datos_etiquetados[i] = (pseudo_datos, pseudo_etiquetas_datos)
                
            
            for i, Hi in self.forest.items():
                if hay_cambios_en_arbol[i]:
                    
                    if e[i][t] * W[i][t] < e[i][t-1] * W[i][t-1]:
                        self.learn_random_tree(i, L, y_l, pseudo_datos_etiquetados[i], self.datos_arbol_ind[i])
            
            if not any(hay_cambios_en_arbol):
               
                hay_cambios = False
                
            
            self.accuracy_por_iteracion.append(self.score(X_test, y_test))
                    
        self.errores = e
        self.confianzas = W
        
        
        return self.forest   
        
    def bootstrap(self, L, y_l, p=0.65):
        '''
        La tecnica del bootstrapping es una tecnica de remuestreo que consiste en seleccionar
        aleatoriamente una muestra de datos de un conjunto de datos, con reemplazamiento.
        '''
        #generar indices aleatorios con reemplazamiento
        datos_random = np.random.choice(L.shape[0], size=int(p*L.shape[0]), replace=True)
        L_i = L[datos_random, :]
        y_l_i = y_l[datos_random]
        
        return L_i, y_l_i, datos_random
    
    def estimate_error(self, Hi, L, y_l, datos_arbol_indiv):
        '''
        Estima el error del arbol Hi. Out Of Bag Error
        '''
        errors = []
        L = np.array(L)
        for sample, tag in zip(L, y_l):
            n_votes = 0
            n_hits = 0

            for tree in self.forest.values():
                rows_training = L[datos_arbol_indiv]
                
                used_training = np.any(np.all(sample.reshape(1, -1) == rows_training, axis=1))
                
                if tree is not Hi and not used_training:
                    if tree.predict([sample])[0] == tag:
                        n_hits += 1
                    n_votes += 1
           
            if n_votes > 0:
                errors.append(1 - (n_hits / n_votes))

        return np.mean(errors)
    
    def subsample(self, Hi, U, W):
        '''
        Submuestreo de U
        '''
        W_i = 0
        U_subsampled = []

        while W_i < W:
            rand_row = np.random.choice(U.shape[0])
            W_i += self.confidence(Hi, U[rand_row, :])[0]
            U_subsampled.append(rand_row)

        return np.array(U_subsampled)
    
    
    def confidence(self, Hi, sample):
        #comprobar si hay elementos en el vector clases
        if not np.any(self.clases):
            raise ValueError("No se ha ajustado el modelo")
        count = {i: 0 for i in self.clases}

        for tree in self.forest.values():
            if tree is not Hi:
                count[tree.predict([sample])[0]] += 1

        max_agreement = max(count.values())
        most_agreed_class = max(count, key=count.get)

        return max_agreement / (len(self.forest) - 1), most_agreed_class
    
    def learn_random_tree(self, i, L, y_l, pseudo_datos_etiquetados, datos_arbol_indiv):
        '''
        Reeduca el arbol i con nuevos pseudo datos
        '''
        X = np.concatenate((L[datos_arbol_indiv], pseudo_datos_etiquetados[0]))
        Y = np.concatenate((y_l[datos_arbol_indiv], pseudo_datos_etiquetados[1]))
        self.forest[i] = self.forest[i].fit(X, Y)
        
    def get_errores(self):
        return self.errores
    
    def get_confianzas(self):
        return self.confianzas
       
    def single_predict(self, sample):
    
        count = {i: 0 for i in self.clases}
        for i in (tree.predict([sample])[0] for tree in self.forest.values()):
            count[i] += 1
        return max(count, key=count.get)

    def predict(self, samples):
        samples = (lambda x: np.expand_dims(x, axis=0) if x.ndim == 1 else x)(
            samples
        )
        return np.array([self.single_predict(sample) for sample in samples])
    
    def score(self, X_test, y_test):
        y_predictions = self.predict(X_test)
       
        return np.count_nonzero(y_predictions == y_test) / len(y_test)
    
    def get_accuracy_por_iteracion(self):
        return self.accuracy_por_iteracion

    


# from sklearn.model_selection import train_test_split
# from sklearn.datasets import load_iris, load_breast_cancer, load_digits
# import matplotlib.pyplot as plt

# # #separamos los datos en X y Y
# # x = load_iris().data
# # y = load_iris().target

# x = load_breast_cancer().data
# y = load_breast_cancer().target

# # x = load_digits().data
# # y = load_digits().target


# #separamos los datos en train y test
# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, stratify=y)
# #X_test e y_test se queda como esta para hacer el score despues (datos desconocidos)

# #La siguiente distincion es para que el algoritmo coja gran parte de datos y no los etiquete (U)
# L, U, y_l, y_u = train_test_split(X_train, y_train, test_size=0.8, stratify=y_train)

# alg = CoForest(n=6, theta=0.75)
# alg.fit(L, y_l, U)
# error = alg.get_errores()
# confianzas = alg.get_confianzas()
# accuracies = alg.get_accuracy_por_iteracion()

# print(error)
# print(confianzas)
# print(accuracies)

# #dibujamos las 3 graficas
# plt.plot(range(len(accuracies)), accuracies, label='CoForest', marker='o')
# plt.ylim(0.75,1)
# plt.xticks(range(len(accuracies)))
# plt.xlabel('Iteraciones')
# plt.ylabel('Accuracy')
# plt.title('Accuracy por iteracion')
# plt.legend()
# plt.show()

# # for i, vector in error.items():
# #     plt.plot(range(len(vector)), vector, label='Arbol {}'.format(i), marker='o')
# # plt.xlabel('Iteraciones')
# # plt.ylabel('Error')
# # plt.ylim(0,0.1)
# # plt.title('Error por iteracion')
# # plt.legend()
# # plt.show()
      
# # for i, vector in confianzas.items():
# #     plt.plot(range(len(vector)), vector, label='Arbol {}'.format(i), marker='o')
# # plt.xlabel('Iteraciones')
# # plt.ylabel('Confianza')
# # plt.title('Confianza por iteracion')
# # plt.legend()
# # plt.show()

