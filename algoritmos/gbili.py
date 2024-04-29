"""Este módulo contiene la implementación del algoritmo CoForest

@Autor:     Mario Sanz Pérez
@Fecha:     23/04/2024
@Versión:   1.2
@Nombre:    Gbili.py
"""""

import numpy as np
from scipy.spatial import distance_matrix
from collections import defaultdict, deque

class Gbili:
    """_summary_
    """
    def __init__(self, vertices, labels, K):
        self.vertices = vertices
        self.labels = labels
        self.K = K
        self.distance_matrix = distance_matrix(self.vertices, self.vertices)
        self.graph = defaultdict(list)
    
    def find_knn(self):
        n = len(self.vertices)
        knn_list = defaultdict(list)
        for i in range(n):
            distances = [(self.distance_matrix[i][j], j) for j in range(n) if i != j]
            distances.sort()
            knn_list[i] = [idx for _, idx in distances[:self.K]]
        print(knn_list)
        return knn_list
    
    def find_mknn(self, knn_list):
        mknn_list = defaultdict(list)
        for i in knn_list:
            for j in knn_list[i]:
                if i in knn_list[j]:
                    mknn_list[i].append(j)
        print(mknn_list)
        return mknn_list
    
    def connect_min_distance(self, mknn_list):
        for i in mknn_list:
            for j in mknn_list[i]:
                min_distance = float('inf')
                min_link = None
                for l in self.labels:
                    d = self.distance_matrix[i][j] + self.distance_matrix[j][l]
                    if d < min_distance:
                        min_distance = d
                        min_link = (i, j)
                if min_link:
                    self.graph[min_link[0]].append(min_link[1])
                    self.graph[min_link[1]].append(min_link[0])
        print(self.graph)
        
    def find_components(self):
        visited = {}
        component = {}
        comp_id = 0
        def bfs(v):
            queue = deque([v])
            visited[v] = True
            component[v] = comp_id
            while queue:
                current = queue.popleft()
                for neighbor in self.graph[current]:
                    if not visited.get(neighbor, False):
                        visited[neighbor] = True
                        component[neighbor] = comp_id
                        queue.append(neighbor)
        
        for v in range(len(self.vertices)):
            if v not in visited:
                bfs(v)
                comp_id += 1
        print(component)
        return component
    
    def link_components(self, component):
        comp_label = {}
        for comp in set(component.values()):
            comp_label[comp] = any(self.labels.get(idx) for idx in component if component[idx] == comp)
        
        for v in range(len(self.vertices)):
            if not comp_label[component[v]]:
                for k in self.graph[v]:
                    if comp_label[component[k]]:
                        self.graph[v].append(k)
                        self.graph[k].append(v)
                        break
    
    def solve(self):
        knn_list = self.find_knn()
        mknn_list = self.find_mknn(knn_list)
        self.connect_min_distance(mknn_list)
        component = self.find_components()
        self.link_components(component)
        return self.graph

vertices = np.random.rand(10, 2)
labels = {0: 'A', 5: 'B'}
solver = Gbili(vertices, labels, 3)
graph = solver.solve()
print(graph)
