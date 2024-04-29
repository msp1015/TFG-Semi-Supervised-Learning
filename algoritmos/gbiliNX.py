import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.spatial import distance_matrix
from collections import defaultdict

class GbiliNX:
    def __init__(self, vertices, labels, K):
        self.vertices = vertices 
        self.labels = labels     
        self.K = K              
        self.G = nx.Graph()       
        self.distance_matrix = distance_matrix(vertices, vertices)
        
    def visualize_graph(self, title):
        pos = {i: self.vertices[i] for i in range(len(self.vertices))}
        plt.figure(figsize=(8, 6))
        nx.draw(self.G, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=500, font_size=15, font_weight='bold')

        labeled_nodes = [node for node in self.G.nodes if node in self.labels]
        nx.draw_networkx_nodes(self.G, pos, nodelist=labeled_nodes, node_color='salmon')
        label_pos = {node: (pos[node][0], pos[node][1] + 0.05) for node in labeled_nodes}
        nx.draw_networkx_labels(self.G, label_pos, labels={node: self.labels[node] for node in labeled_nodes}, font_color='darkred')

        plt.title(title)
        plt.show()

    def find_knn(self):
        n = len(self.vertices)
        for i in range(n):
            distances = sorted(((self.distance_matrix[i][j], j) for j in range(n) if i != j), key=lambda x: x[0])
            for _, idx in distances[:self.K]:
                self.G.add_edge(i, idx)
        # self.visualize_graph("Graph After k-NN")
        pos = {i: self.vertices[i] for i in range(len(self.vertices))}

        plt.figure(figsize=(8, 6))
        nx.draw(self.G, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=500, font_size=15, font_weight='bold')

        # Resaltar los nodos etiquetados
        labelled_nodes = [node for node in self.G.nodes() if node in labels]
        nx.draw_networkx_nodes(self.G, pos, nodelist=labelled_nodes, node_color='salmon')

        # Dibujar las etiquetas de los nodos etiquetados
        label_pos = {node: (pos[node][0], pos[node][1] + 0.05) for node in labelled_nodes}  # Ajusta la posición de la etiqueta
        nx.draw_networkx_labels(self.G, label_pos, labels={node: labels[node] for node in labelled_nodes}, font_color='darkred')

        plt.title('Visualización del Grafo con NetworkX')
        plt.show()
        
    def find_mknn(self):
        mknn_list = defaultdict(list)
        for i in self.G.nodes:
            neighbors_i = list(self.G.neighbors(i))
            for j in neighbors_i:
                if i in self.G.neighbors(j): 
                    mknn_list[i].append(j)
        # self.visualize_graph("Graph After Mutual k-NN")
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
                    self.G.add_edge(min_link[0], min_link[1])
        # self.visualize_graph("Graph After Connecting Min Distance")

    def link_components(self):
        components = list(nx.connected_components(self.G))
        comp_label = {frozenset(c): any(v in self.labels for v in c) for c in components}
        for comp in components:
            if not comp_label[frozenset(comp)]:
                for v in comp:
                    for k in self.G.nodes():
                        if k not in comp and comp_label.get(frozenset(nx.node_connected_component(self.G, k)), False):
                            self.G.add_edge(v, k)
                            break
    
        # self.visualize_graph("Final Graph After Linking Components")

    def solve(self):
        self.find_knn()
        mknn_list = self.find_mknn()
        self.connect_min_distance(mknn_list)
        self.link_components()
        return self.G


#np.random.seed(0)
vertices = np.random.rand(35, 2)  
labels = {0: 'A', 5: 'B', 7: 'C', 9: 'D', 3: 'E', 8: 'F', 12: 'G', 15: 'H', 18: 'I', 20: 'J', 22: 'K', 25: 'L', 27: 'M', 29: 'N'} 
solver = GbiliNX(vertices, labels, 2)
graph = solver.solve()
# Posiciones de los nodos basadas en sus coordenadas (si son representativas de la estructura del grafo)
pos = {i: vertices[i] for i in range(len(vertices))}

plt.figure(figsize=(8, 6))
nx.draw(graph, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=500, font_size=15, font_weight='bold')

# Resaltar los nodos etiquetados
labelled_nodes = [node for node in graph.nodes() if node in labels]
nx.draw_networkx_nodes(graph, pos, nodelist=labelled_nodes, node_color='salmon')

# Dibujar las etiquetas de los nodos etiquetados
label_pos = {node: (pos[node][0], pos[node][1] + 0.05) for node in labelled_nodes}  # Ajusta la posición de la etiqueta
nx.draw_networkx_labels(graph, label_pos, labels={node: labels[node] for node in labelled_nodes}, font_color='darkred')

plt.title('Visualización del Grafo con NetworkX')
plt.show()
