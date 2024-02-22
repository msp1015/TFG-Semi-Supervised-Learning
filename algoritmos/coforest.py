'''
@Autor:     Mario Sanz Pérez	
@Fecha:     21/02/2024
@Versión:   1.0
@Nombre:    CoForest.py
'''

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from pandas import DataFrame

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
        self.n = n
        self.theta = theta
        self.forest = {}
        self.clases = []
        
    def fit(self, L, y_l, U):
        '''
        Ajusta el modelo a los datos de entrenamiento
        '''
        #segun los datos en y_l podemos sacar el numero de clases
        self.clases = np.unique(y_l)
        
        datos_arbol_individuales = {}
        e_init = [0.5] * self.n
        W_init = [min(0.1 * len(L), 100)] * self.n
        for i in range(self.n):
            # Se inicializa el bosque con n arboles de decision
            self.forest[i] = DecisionTreeClassifier()
            L_i, y_l_i, indices_datos= self.bootstrap(L, y_l)
            datos_arbol_individuales[i] = indices_datos
            
            self.forest[i].fit(L_i, y_l_i)
        
        t = 0
        e = e_init
        W = [0] * self.n
        hay_cambios = True
        while hay_cambios:
            t = t + 1
            
            for i, Hi in self.forest.items():
                e[i] = self.estimate_error(Hi, L, y_l, datos_arbol_individuales[i])
                W[i] = W_init[i]
                pseudo_datos = []
                pseudo_etiquetas_datos = []
                if e[i] < e_init[i]:
                    U_samples = self.sub_sample(Hi, U, e_init[i] * W_init[i] / e[i])
                    
                    for x_u in U_samples:
                        confidence, most_agreed_class = self.confidence(Hi, U[x_u, :])
                        if confidence > self.theta:
                            pseudo_datos.append(U[x_u, :])
                            pseudo_etiquetas_datos.append(most_agreed_class)
                            W[i] += confidence
            for i, Hi in self.forest.items():
                if e[i] * W[i] < e_init[i] * W_init[i]:
                    self.learn_random_tree(i, L, y_l, pseudo_datos, pseudo_etiquetas_datos, datos_arbol_individuales)
                    
            e_init = e
            W_init = W
        return self.forest    
        
    def bootstrap(self, L, y_l, random_state=None, p=0.75):
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
        Estima el error del arbol Hi
        '''
        errors = []

        for sample, tag in zip(L, y_l):
            n_votes = 0
            n_hits = 0

            for i, tree in self.forest.items():
                rows_training = L[datos_arbol_indiv[i]]
                used_training = np.any(np.all(sample == rows_training, axis=1))

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
            W_i += self.concomitant_confidence(Hi, U[rand_row, :])[0]
            U_subsampled.append(rand_row)

        return np.array(U_subsampled)
    
    
    def confidence(self, Hi, sample):
        
        count = {i: 0 for i in self.clases}

        for tree in self.forest.values():
            if tree is not Hi:
                count[tree.predict([sample])[0]] += 1

        max_agreement = max(count.values())
        most_agreed_class = max(count, key=count.get)

        return max_agreement / (len(self.ensemble) - 1), most_agreed_class
    
    def learn_random_tree(self, i, L, y_l, pseudo_data, pseudo_labels, datos_arbol_indiv):
        '''
        Reeduca el arbol i con nuevos pseudo datos
        '''
        X = np.concatenate((L[datos_arbol_indiv[i]], pseudo_data))
        Y = np.concatenate((y_l[datos_arbol_indiv[i]], pseudo_labels))
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
        return np.count_nonzero(y_predictions == y_test) / len(y_test)
    
    
   
   
   
   
    
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# #leer el archivo
# x = load_iris().data
# y = load_iris().target

# #separamos los datos en train y test
# X_train, U, y_train, y_test = train_test_split(x, y, test_size=0.7, stratify=y)

# X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3, stratify=y_train)

# X_train = np.append(X_train, U, axis=0)
# y_train = np.append(y_train, [-1] * U.shape[0])


# alg = CoForest(n=10, theta=0.75)
# forest = alg.fit(X_train, y_train, U)
# score = alg.score(U, y_test)
# print(score)

# L = X_train.shape[0]
# print(L)

# print("previo al bootstrapping:")
# print(X_train)
# print(y_train)

# alg = CoForest(n=10, theta=0.75)
# L, y_l = alg.bootstrap(X_train, y_train)
# print("tras el bootstrapping:")
# print(L)
# print(y_l)

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

#separamos los datos en X y Y
x = load_iris().data
y = load_iris().target
scores = []

unlabeled_size = [0.4, 0.5, 0.6, 0.7, 0.8]
for i in unlabeled_size:
    #separamos los datos en train y test
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size= 0.15, stratify=y)
    #X_test e y_test se queda como esta para hacer el score despues (datos desconocidos)
    
    #La siguiente distincion es para que el algoritmo coja gran parte de datos y no los etiquete (U)
    L, U, y_l, y_u = train_test_split(X_train, y_train, test_size=i, stratify=y_train)
    coforest = CoForest(n=30, theta=0.75)
    coforest.fit(L, y_l, U)
    print(coforest.score(X_test, y_test))
    scores.append(coforest.score(X_test, y_test))


#construimos un grafico con los resultados
#ajustar eje y

plt.plot(unlabeled_size, scores)
plt.ylim(0.7, 1)
plt.xlabel('Unlabeled size')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Unlabeled size')
plt.show()

