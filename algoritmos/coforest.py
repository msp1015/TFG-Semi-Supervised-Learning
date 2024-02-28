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
        
        
    def fit(self, L, y_l, U):
        '''
        Ajusta el modelo a los datos de entrenamiento
        '''
        #segun los datos en y_l podemos sacar el numero de clases
        self.clases = np.unique(y_l)
        self.L = L
        self.y_l = y_l
        self.U = U
        #print("Inicializando el bosque...")
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
            
            #print("\t\tEntramos bucle while en el paso t ", t, " con errores ", e, " y confianzas ", W)
            for i, Hi in self.forest.items():
                error_actual = self.estimate_error(Hi, L, y_l, self.datos_arbol_ind[i])
                W_actual = 0
                W[i].append(W_actual)
                # print("\t\t\t Arbol ", i, " con error ", e[i][t], " en el paso t ", t)
                # print("\t\t\t Arbol ", i, " con confianza ", W[i][t], " en el paso t ", t)
                pseudo_datos = [] # Para cada arbol se almacenan los datos que se etiquetaran
                pseudo_etiquetas_datos = [] # Etiquetas que acompañan a los datos que se etiquetaran
                
                if error_actual < e[i][t-1]:
                    e[i].append(error_actual)
                    #print("\t\t\t\tEl arbol ", i, " ha mejorado su rendimiento, antes e", e[i][t-1], " ahora e", e[i][t])
                    #print("\t\t\t\tPreparando submuestreo de U... para el arbol ", i, " en el paso ", t)
                    if e[i][t] == 0.0:
                        Wmax = self.theta * U.shape[0]
                    else:
                        Wmax = e[i][t-1] * W[i][t-1] / e[i][t]
                    U_samples = self.subsample(Hi, U, Wmax)
                    #print("\t\t\t\tEntramos en el submuestreo de U, bucle for ")
                    for x_u in U_samples:
                        confidence, most_agreed_class = self.confidence(Hi, U[x_u, :])
                        
                        if confidence > self.theta:
                            #print("\t\t\t\t\tConfianza ", confidence, " mayor que el umbral ", self.theta, " para la muestra ", x_u)  
                            hay_cambios_en_arbol[i] = True
                            pseudo_datos.append(U[x_u, :])
                            pseudo_etiquetas_datos.append(most_agreed_class)
                            #print("\t\t\t\t\tEtiquetamos la muestra ", x_u, " con la clase ", most_agreed_class)
                            #print("\t\t\t\t\tSumammos la confianza)")
                            W_actual += confidence   
                else:
                    e[i].append(e[i][t-1])
                
                if W_actual != 0:
                    W[i][t] = W_actual
                else:
                    W[i][t] = W[i][t-1]                
                pseudo_datos_etiquetados[i] = (pseudo_datos, pseudo_etiquetas_datos)
                
                #print("\t\t\tPseudo datos etiquetados para el arbol ", i, " en el paso ", t, " son ", pseudo_datos_etiquetados[i])    
            # print("\t\tErrores en el paso ", t, " son ", e)
            #print("\t\tConfianzas en el paso ", t, " son ", W)
            #print("\t\tEntramos en el bucle for para reeducar el arbol")
            for i, Hi in self.forest.items():
                #print(hay_cambios_en_arbol)
                if hay_cambios_en_arbol[i]:
                    #print("\t\t\t\tHay cambios en el arbol ", i, " en el paso ", t)
                    if e[i][t] * W[i][t] < e[i][t-1] * W[i][t-1]:
                        #print("\t\t\t\t\tEl arbol ", i, " ha mejorado su rendimiento, antes e", e[i][t-1], " ahora e", e[i][t])
                        self.learn_random_tree(i, L, y_l, pseudo_datos_etiquetados[i], self.datos_arbol_ind[i])
            
            if not any(hay_cambios_en_arbol):
                #print("\t\t\tNo hay cambios en ningun arbol en el paso ", t)
                hay_cambios = False        
        self.errores = e
        self.confianzas = W
        
        
        return self.forest   
        
    def bootstrap(self, L, y_l, random_state=None, p=0.75):
        '''
        La tecnica del bootstrapping es una tecnica de remuestreo que consiste en seleccionar
        aleatoriamente una muestra de datos de un conjunto de datos, con reemplazamiento.
        '''
        #generar indices aleatorios con reemplazamiento
        # print("Antes del bootstrapping")
        # print(L)
        # print(y_l)
        datos_random = np.random.choice(L.shape[0], size=int(p*L.shape[0]), replace=True)
        L_i = L[datos_random, :]
        y_l_i = y_l[datos_random]
        
        # print("Despues del bootstrapping")
        # print(L_i)
        # print(y_l_i)
        
        return L_i, y_l_i, datos_random
    
    def estimate_error(self, Hi, L, y_l, datos_arbol_indiv):
        '''
        Estima el error del arbol Hi
        '''
        errors = []
        L = np.array(L)
        # print("L: ", L)
        for sample, tag in zip(L, y_l):
            #print("Sample: ", sample)
            n_votes = 0
            n_hits = 0

            for i, tree in self.forest.items():
                rows_training = L[datos_arbol_indiv[i]]
                #print("Rows training: ", rows_training)
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
        #print("Reeducando arbol ")
        X = np.concatenate((L[datos_arbol_indiv], pseudo_datos_etiquetados[0]))
        Y = np.concatenate((y_l[datos_arbol_indiv], pseudo_datos_etiquetados[1]))
        #print("Nuevos Datos para el arbol ", i, " son ", X, Y)
        self.forest[i] = self.forest[i].fit(X, Y)
        
        
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
        # print("Datos a obtener: ", y_test)
        # print("(************************************)")
        # print("Predicciones: ", y_predictions)
        return np.count_nonzero(y_predictions == y_test) / len(y_test)
    

    
   
   
   
   
    
# # from sklearn.datasets import load_iris
# # from sklearn.model_selection import train_test_split
# # #leer el archivo
# # x = load_iris().data
# # y = load_iris().target

# # #separamos los datos en train y test
# # X_train, U, y_train, y_test = train_test_split(x, y, test_size=0.7, stratify=y)

# # X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3, stratify=y_train)

# # X_train = np.append(X_train, U, axis=0)
# # y_train = np.append(y_train, [-1] * U.shape[0])


# # alg = CoForest(n=10, theta=0.75)
# # forest = alg.fit(X_train, y_train, U)
# # score = alg.score(U, y_test)
# # print(score)

# # L = X_train.shape[0]
# # print(L)

# # print("previo al bootstrapping:")
# # print(X_train)
# # print(y_train)

# # alg = CoForest(n=10, theta=0.75)
# # L, y_l = alg.bootstrap(X_train, y_train)
# # print("tras el bootstrapping:")
# # print(L)
# # print(y_l)

# from sklearn.model_selection import train_test_split
# from sklearn.datasets import load_iris
# import matplotlib.pyplot as plt

# #separamos los datos en X y Y
# x = load_iris().data
# y = load_iris().target
# scores = []

# # unlabeled_size = [0.4, 0.5, 0.6, 0.7, 0.8]
# # for i in unlabeled_size:
# #     #separamos los datos en train y test
# #     X_train, X_test, y_train, y_test = train_test_split(x, y, test_size= 0.15, stratify=y)
# #     #X_test e y_test se queda como esta para hacer el score despues (datos desconocidos)
    
# #     #La siguiente distincion es para que el algoritmo coja gran parte de datos y no los etiquete (U)
# #     L, U, y_l, y_u = train_test_split(X_train, y_train, test_size=i, stratify=y_train)
# #     coforest = CoForest(n=30, theta=0.75)
# #     coforest.fit(L, y_l, U)
# #     print(coforest.score(X_test, y_test))
# #     scores.append(coforest.score(X_test, y_test))


# # #construimos un grafico con los resultados
# # #ajustar eje y

# # plt.plot(unlabeled_size, scores)
# # plt.ylim(0.7, 1)
# # plt.xlabel('Unlabeled size')
# # plt.ylabel('Accuracy')
# # plt.title('Accuracy vs Unlabeled size')
# # plt.show()


#  #separamos los datos en train y test
# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, stratify=y)
# #X_test e y_test se queda como esta para hacer el score despues (datos desconocidos)

# #La siguiente distincion es para que el algoritmo coja gran parte de datos y no los etiquete (U)
# L, U, y_l, y_u = train_test_split(X_train, y_train, test_size=0.7, stratify=y_train)

# # for i,j in zip(load_iris().data, load_iris().target):
# #     print(i,j)
# # print("****************************************")
# # print("****************************************")
# # print("Datos para testear el algoritmo")
# # for i, j in zip(X_test, y_test):
# #     print(i,j)

# # print("****************************************")
# # print("****************************************")
# # print("Datos para entrenar el algoritmo (etiquetados)")
# # for i, j in zip(L, y_l):
# #     print(i,j)
# # print("****************************************")


# # alg = CoForest(n=10, theta=0.75)
# # # for i in range(50,100,5):
# # #     alg.fit(L, y_l, U)
# # #     scores.append(alg.score(X_test, y_test))

# # alg.fit(L, y_l, U)
# # print(alg.score(X_test, y_test))
    
    
    

