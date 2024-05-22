import numpy as np
from sklearn.neighbors import KDTree
from concurrent.futures import ThreadPoolExecutor
from scipy.spatial.distance import minkowski

class Rgcli:
    def __init__(self, X, puntos_etiquetados, ke=15, ki=5, nt=4):
        self.X = X
        self.puntos_etiquetados = puntos_etiquetados
        self.ke = ke
        self.ki = ki
        self.nt = nt
        self.n = X.shape[0]
        self.etiquetas = np.array([1 if i in puntos_etiquetados else 0 for i in range(self.n)])
        self.arbol_knn = KDTree(X)
        self.kNN = {}
        self.F = {}
        self.L = {}
        self.grafo = {i: [] for i in range(self.n)}
        
    def searchKNN(self, T):
        for vi in T:
            self.kNN[vi] = self.arbol_knn.query([self.X[vi]], k=self.ke, return_distance=False)[0]
            dists, indices = self.arbol_knn.query([self.X[vi]], k=self.ke)
            nearest_labeled = [index for index in indices[0] if self.etiquetas[index] == 1]
            self.L[vi] = nearest_labeled[0] if nearest_labeled else None
            self.F[vi] = indices[0][-1]
            
    def searchRGCLI(self, T):
        for vi in T:
            epsilon = {}
            for vj in self.kNN[vi]:
                if np.linalg.norm(self.X[vi] - self.X[vj]) <= np.linalg.norm(self.X[vj] - self.X[self.F[vj]]):
                    e = (vi, vj)
                    epsilon[e] = np.linalg.norm(self.X[vi] - self.X[vj]) + np.linalg.norm(self.X[vj] - self.X[self.L[vj]])
            
            sorted_edges = sorted(epsilon.items(), key=lambda item: item[1])
            selected_edges = sorted_edges[:self.ki]
            for (vi, vj), _ in selected_edges:
                if vj not in self.grafo[vi] and vi != vj:
                    self.grafo[vi].append(vj)
                if vi not in self.grafo[vj] and vi != vj:
                    self.grafo[vj].append(vi)
                
    def ajustar(self):
        V = range(self.n)
        T = np.array_split(list(V), self.nt)
        print(T)
        with ThreadPoolExecutor(max_workers=self.nt) as executor:
            futures = [executor.submit(self.searchKNN, Ti) for Ti in T]
            for future in futures:
                future.result()
                
            futures = [executor.submit(self.searchRGCLI, Ti) for Ti in T]
            for future in futures:
                future.result()
        
        return self.grafo

# Uso de la clase RGCLI
X = np.random.rand(100, 2)  # 100 puntos en un espacio bidimensional
y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # Primeros 10 puntos están etiquetados

rgcli = Rgcli(X, y, ke=5, ki=5, nt=4)
grafo = rgcli.ajustar()

print(grafo)

import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()
for u, vecinos in grafo.items():
    for v in vecinos:
        G.add_edge(u, v)

nx.draw(G, with_labels=True)
plt.show()
