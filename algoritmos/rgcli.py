import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KDTree
from scipy.spatial.distance import euclidean

class RGCLI:
    """ Algoritmo de construcción de grafos RGCLI basado en el articulo:
        'RGCLI: Robust Graph that Considers Labeled Instances for Semi-
        Supervised Learning' de los autores:
        - Lilian Berton, Alan Valejo, Thiago de Paulo Faleiros, Jorge Valverde-Rebaza
        Alnea de Andrade Lopes
    """
    def __init__(self, datos_se, datos_e, etiquetas, ke=10, ki=3):
        """ Inicializa el algoritmo RGCLI

        Args:
        - datos_e: np.array
            datos etiquetados
        - datos_se: np.array
            datos sin etiquetar
        - etiquetas: np.array
            Todas las etiquetas
        - Ke: int
            Número de vecinos más cercanos
        - Ki: int
            Número de vecinos más cercanos para el RGCLI
        """
        self.nodos = np.concatenate((datos_e, datos_se), axis=0)
        self.nodos_etiquetados = np.array(range(len(datos_e)))
        self.nodos_sin_etiquetar = np.array(range(len(datos_e), len(datos_e) + len(datos_se)))

        self.etiquetas_etiquetados = etiquetas[:len(datos_e)]
        self.etiquetas_modificadas = np.concatenate((self.etiquetas_etiquetados, np.full(len(datos_se), -1)))

        self.ke = ke
        self.ki = ki
        self.V = list(range(len(self.nodos)))
        self.E = []
        self.W = {}
        self.kdtree = KDTree(self.nodos)
        self.l_kdtree = KDTree(self.nodos[self.nodos_etiquetados, :])
        self.kNN = {}
        self.F = {}
        self.L = {}
        self.grafo_knn = {}
        self.grafoFinal = {}

    def search_knn(self):
        """
        Construye el grafo de vecinos mas cercanos.

        Para cada nodo en self.V, este método:
        - Encuentra todos los vecinos del nodo en el espacio de características, 
            utilizando un kdtree para la búsqueda eficiente.
        - Almacena los primeros `ke` vecinos en `self.kNN` y `self.grafo_knn`.
        - Encuentra los vecinos etiquetados del nodo y almacena el más cercano en `self.L`.
        - Almacena el vecino `ke`-ésimo más lejano en `self.F`.

        Returns:
            dict: Un diccionario que mapea cada nodo en self.V a sus `ke` vecinos más cercanos.
        """
        for v in self.V:
            all_neighbors = self.kdtree.query([self.nodos[v]], k=len(self.nodos), return_distance=False)[0]
            self.kNN[v] = all_neighbors[1:self.ke + 1].tolist()
            self.grafo_knn[v] = self.kNN[v]

            labeled_neighbors = self.l_kdtree.query([self.nodos[v]], k=2, return_distance=False)[0]
            self.L[v] = self.nodos_etiquetados[labeled_neighbors[labeled_neighbors != v][0]]
            self.F[v] = all_neighbors[-self.ke]
        return self.kNN

    def search_rgcli(self):
        """
        Construye un grafo final con ayuda de los datos etiquetados.

        Para cada nodo en self.V, este método realiza lo siguiente:
        - Calcula una medida de distancia, epsilon, para cada vecino del nodo.
        - Selecciona los `ki` vecinos con la menor medida epsilon y los añade a `self.E`.
        - Asigna un peso de 1 a cada uno de estos vecinos en `self.W`.
        - Añade estos vecinos al grafo final `self.grafoFinal`.

        El grafo final es un grafo no dirigido, cada nodo está conectado a sus `ki` vecinos.
        """
        for vi in self.V:
            epsilon = dict()
            for vj in self.kNN[vi]:
                if euclidean(self.nodos[vi], self.nodos[vj]) <= euclidean(self.nodos[vj], self.nodos[self.F[vj]]):
                    e = (vi, vj)
                    epsilon[e] = euclidean(self.nodos[vi], self.nodos[vj]) + euclidean(self.nodos[vj], self.nodos[self.L[vj]])
            E_estrella = sorted(epsilon, key=epsilon.get)[:self.ki]
            self.E.extend(E_estrella)
            for e in E_estrella:
                self.W[e] = 1
                if e[0] not in self.grafoFinal:
                    self.grafoFinal[e[0]] = []
                if e[1] not in self.grafoFinal[e[0]]:
                    self.grafoFinal[e[0]].append(e[1])
                    
                if e[1] not in self.grafoFinal:
                    self.grafoFinal[e[1]] = []
                if e[0] not in self.grafoFinal[e[1]]:
                    self.grafoFinal[e[1]].append(e[0])

    def construir_grafo(self):
        """
        Metodo que ejecuta los pasos del algoritmo RGCLI

        Returns:
            dict, dict: Un diccionario para el grafo del primer paso y otro para el grafo final.   
        """
        self.search_knn()
        self.search_rgcli()
        return self.grafo_knn, self.grafoFinal
