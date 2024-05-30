"""Este módulo contiene la implementación del algoritmo Gbili

@Autor:     Mario Sanz Pérez
@Fecha:     08/05/2024
@Versión:   1.1
@Nombre:    gbili.py
"""""

from matplotlib import pyplot as plt
import networkx as nx
from collections import deque, defaultdict
import numpy as np
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from copy import deepcopy
#from localglobalconsistency import LGC
class Gbili:
    """ Algoritmo de construccion de grafos GBILI basado en el artículo:
    'Graph construction based on labeled instances for
    Semi-Supervised Learning' de los autores: 
    Lilian Berton y Alneu de Andrade Lopes
    """

    def __init__(self, datos_se, datos_e, etiquetas, K):
        """
        Constructor de la clase.

        Args:
        - datos_se: numpy.ndarray
            Datos sin etiquetar.
        - datos_e: numpy.ndarray
            Datos etiquetados.
        - etiquetas: numpy.ndarray
            Todas las etiquetas.
        - K: int
            Número de vecinos más cercanos a considerar.

        Inicializa los atributos:
        - nodos: numpy.ndarray
            Todos los nodos (datos reales), tanto etiquetados como sin etiquetar.
        - nodos_etiquetados: numpy.ndarray
            Índices de los nodos etiquetados.
        - nodos_sin_etiquetar: numpy.ndarray
            Índices de los nodos sin etiquetar.
        - K: int
            Número de vecinos más cercanos a considerar.
        - matriz_distancias: numpy.ndarray
            Matriz de distancias entre los nodos.
        - grafo: dict
            Grafo que representa las conexiones entre los nodos.
        - etiquetas_etiquetados: numpy.ndarray
            Etiquetas correspondientes a los nodos etiquetados.
        """

        self.nodos = np.concatenate((datos_e, datos_se), axis=0)
        self.nodos_etiquetados = np.array(range(len(datos_e)))
        self.nodos_sin_etiquetar = np.array(range(len(datos_e), len(datos_e) + len(datos_se)))
        self.K = K
        self.matriz_distancias = distance_matrix(self.nodos, self.nodos)
        self.grafo = {}

        
        self.etiquetas_etiquetados = etiquetas[:len(datos_e)]
        # TODO: borrar
        self.etiquetas_modificadas = np.concatenate((self.etiquetas_etiquetados, np.full(len(datos_se), -1)))
    def construir_grafo(self):
        """
        Indica el proceso de construcción del grafo GBILI.
        Se realizan los siguientes pasos:
        1. Encuentra los k vecinos más cercanos de cada vértice.
        2. Encuentra los pares mutuos de vecinos más cercanos.
        3. Conecta los vértices con la mínima distancia.
        4. Encuentra los componentes del grafo.
        5. Conecta los componentes del grafo basándose en la presencia de etiquetas.

        Returns:
            dict: grafo resultante de la construcción.
        """
        # Configurar la figura y los ejes
        # fig, axs = plt.subplots(2, 2, figsize=(12, 6))  # 2 fila, 2 columnas
        # colores_map = np.array(self.etiquetas_modificadas)

        lista_knn = self.encuentra_knn()
        # self.dibujar_grafo(lista_knn, colores_map, axs[0, 0], "K-NN")

        lista_mknn = self.encuentra_mknn(lista_knn)
        # self.dibujar_grafo(lista_mknn, colores_map, axs[0, 1], "Mutual K-NN")

        self.conectar_minima_distancia(lista_mknn)
        grafo_1 = deepcopy(self.grafo)

        # Componente = subgrafo
        componentes = self.encontrar_componentes()   
        # print("Componentes antes de conectar: ", componentes) 
        self.conectar_componentes(componentes, lista_knn)
        # print("GRAFO CON COMPONENTES CONECTADOS: ", self.grafo)
        # print()
        grafo_2 = deepcopy(self.grafo)
        # print("Componentes despues de conectar: ", self.encontrar_componentes())

        # self.dibujar_grafo(grafo_1, colores_map, axs[1, 0], "Grafo antes de conectar componentes")
        # self.dibujar_grafo(grafo_2, colores_map, axs[1, 1], "Grafo después de conectar componentes")
        # Crear una leyenda
        custom_lines = [Line2D([0], [0], color='yellow', lw=3),
                        Line2D([0], [0], color='blue', lw=3),
                        Line2D([0], [0], color='green', lw=3),
                        Line2D([0], [0], color='grey', lw=3)]

        # fig.legend(custom_lines, ['0', '1', '2', 'Desconocido'])
        # plt.show()
        return self.grafo

    def encuentra_knn(self):
        """ Encuentra los k vecinos más cercanos de cada vértice.
        <<knn>>: `k-nearest neighbors`

        Returns:
            dict: contiene los k vecinos más cercanos de cada vértice.
        """
        n = len(self.nodos)
        lista_knn = {}
        for i in range(n):
            # Guarda una tupla con la distancia y el indice del vertice
            distancias = [(self.matriz_distancias[i][j], j) for j in range(n) if i != j]
            distancias.sort()
            # Seleccionar los K vecinos más cercanos
            lista_knn[i] = [ind_v for _, ind_v in distancias[:self.K]]
        return lista_knn

    def encuentra_mknn(self, lista_knn):
        """ Para los vértices en la lista de k-NN
        encuentra los pares mutuos de vecinos más cercanos.
        Mutuo k-NN: Si i es vecino más cercano de j y 
                    j es vecino más cercano de i.
        Args:
            lista_knn (dict): contiene los k vecinos más cercanos de cada vértice.

        Returns:
            dict: contiene los pares mutuos de vecinos más cercanos.
        """
        lista_mknn = {}
        for i in lista_knn:
            lista_mknn[i] = []
            for j in lista_knn[i]:
                if i in lista_knn[j]:
                    lista_mknn[i].append(j)
        return lista_mknn

    def conectar_minima_distancia(self, lista_mknn):
        """ Conecta los vértices con la mínima distancia.
        Almacena esos enlaces en el diccionario que representa el grafo.
        El grafo es no dirigido, por lo que se conectan ambos nodos.

        Args:
            lista_mknn (dict): contiene los pares mutuos de vecinos más cercanos.
        """
        for i in lista_mknn:
            # Si el nodo no tiene vecinos mutuos, se agregan como elementos solitarios
            if not lista_mknn[i]:
                self.grafo[i] = []
                continue
            for j in lista_mknn[i]:
                min_distancia = float('inf')
                min_enlace = None
                for l in self.nodos_etiquetados:
                    d = self.matriz_distancias[i][j] + self.matriz_distancias[j][l]
                    if d < min_distancia:
                        min_distancia = d
                        min_enlace = (i, j)
            if min_enlace:
                if min_enlace[0] not in self.grafo:
                    self.grafo[min_enlace[0]] = []
                if min_enlace[1] not in self.grafo:
                    self.grafo[min_enlace[1]] = []
                if min_enlace[1] not in self.grafo[min_enlace[0]]:
                    self.grafo[min_enlace[0]].append(min_enlace[1])
                if min_enlace[0] not in self.grafo[min_enlace[1]]:
                    self.grafo[min_enlace[1]].append(min_enlace[0])

    def encontrar_componentes(self):
        """ Encuentra los componentes del grafo.
        Utiliza BFS (Breadth First Search) o busqueda en anchura 
        para encontrar los componentes.

        Returns:
            dict: contiene los componentes del grafo.
        """
        visitados = {}
        componentes = {}
        id_comp = 0

        def bfs(v):
            """Realiza una búsqueda en anchura a partir de un vértice dado.

            Esta función implementa el algoritmo de búsqueda en anchura
            (BFS por sus siglas en inglés) en un grafo. 
            Comienza la búsqueda desde el vértice 'v' y visita todos los 
            vértices alcanzables desde 'v'.

            Args:
                v (int): El vértice de inicio para la búsqueda en anchura.
            """
            cola = deque([v])
            visitados[v] = True
            componentes[v] = id_comp
            while cola:
                actual = cola.popleft()
                for vecino in self.grafo.get(actual, []):
                    if not visitados.get(vecino, False):
                        visitados[vecino] = True
                        componentes[vecino] = id_comp
                        cola.append(vecino)

        for v in range(len(self.nodos)):
            if v not in visitados:
                bfs(v)
                id_comp += 1
        return componentes
 
    def conectar_componentes(self, comp, lista_knn):
        """
        Conecta los componentes del grafo basándose en la presencia de etiquetas.

        Args:
            comp (dict): contiene los componentes del grafo con cada nodo mapeado a su componente.
            lista_knn (dict): contiene los k vecinos más cercanos de cada vértice.
        """
        # Invertir el diccionario de componentes para agrupar nodos por componente
        comp_a_nodos = defaultdict(list)
        for nodo, componente in comp.items():
            comp_a_nodos[componente].append(nodo)

        # Determinar si cada componente tiene nodos etiquetados
        comp_etiquetados = {}
        for componente, nodos in comp_a_nodos.items():
            comp_etiquetados[componente] = any(nodo in self.nodos_etiquetados for nodo in nodos)

        # Conectar componentes según las condiciones dadas
        for v in range(len(self.nodos)):
            componente_v = comp[v]
            # Verificar si la componente de v no tiene nodos etiquetados
            if not comp_etiquetados[componente_v]:
                for vk in lista_knn[v]:
                    componente_vk = comp[vk]
                    # Verificar si la componente de vk tiene nodos etiquetados
                    if comp_etiquetados[componente_vk]:

                        if vk not in self.grafo[v]:
                            self.grafo[v].append(vk)
                        if v not in self.grafo[vk]:
                            self.grafo[vk].append(v)

    def dibujar_grafo(self, grafo, colores_map, ax, titulo):
        # Crear un objeto grafo de NetworkX
        G = nx.Graph()

        # Añadir los nodos y las aristas desde el diccionario
        for nodo, vecinos in grafo.items():
            G.add_node(nodo)  # Aunque no es necesario añadir explícitamente los nodos
            for vecino in vecinos:
                G.add_edge(nodo, vecino)
                
        G_sorted = nx.Graph()
        G_sorted.add_nodes_from(sorted(G.nodes(data=True)))
        G_sorted.add_edges_from(G.edges(data=True))      
        colors = list(map(lambda x: 'grey' if x==-1 else 'yellow' if x==0 else 'blue' if x==1 else 'green', colores_map))        # Dibujar el grafo
        nx.draw(G_sorted, ax=ax, with_labels=True, node_color=colors, edge_color='gray', node_size=50, font_size=5, font_weight='bold')
            
        ax.set_title(titulo)
        

    
    
## Ejemplo de uso
# from sklearn.datasets import load_iris, load_breast_cancer
# from sklearn.model_selection import train_test_split
# iris = load_iris()
# breast_cancer = load_breast_cancer()
# x = iris.data
# y = iris.target
# K = 10

# L, U, L_, U_ = train_test_split(x, y, test_size=0.7, stratify=y, random_state=42)

# todas_etiquetas = np.concatenate((L_, U_))


# solver = Gbili(U, L,todas_etiquetas, K)
# grafo = solver.construir_grafo()

# inferecia = LGC(grafo, solver.nodos, solver.etiquetas_etiquetados, alpha=0.99, tol=0.1, max_iter=10000)
# predicciones = inferecia.inferir_etiquetas()

# predicciones[len(L):]
# etiquetas_reales = todas_etiquetas[len(L):]
# accuracy = np.mean(predicciones[len(L):] == etiquetas_reales)
# print("Predicciones: ", predicciones[len(L):])
# print("Etiquetas reales: ", etiquetas_reales)
# print(f"Accuracy: {accuracy}")