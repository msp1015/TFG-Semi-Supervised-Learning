"""Este módulo contiene la implementación del algoritmo Gbili

@Autor:     Mario Sanz Pérez
@Fecha:     08/05/2024
@Versión:   1.1
@Nombre:    gbili.py
"""""

from collections import deque, defaultdict
import numpy as np
from scipy.spatial import distance_matrix
from copy import deepcopy

class Gbili:
    """ Algoritmo de construccion de grafos GBILI basado en el artículo:
    'Graph construction based on labeled instances for
    Semi-Supervised Learning' de los autores: 
    Lilian Berton y Alneu de Andrade Lopes
    """

    def __init__(self, datos_se, datos_e, etiquetas, k_vecinos=5):
        """
        Constructor de la clase.

        Args:
        - datos_se: numpy.ndarray
            Datos sin etiquetar.
        - datos_e: numpy.ndarray
            Datos etiquetados.
        - etiquetas: numpy.ndarray
            Todas las etiquetas.
        - k_vecinos: int
            Número de vecinos más cercanos a considerar.

        Inicializa los atributos:
        - nodos: numpy.ndarray
            Todos los nodos (datos reales), tanto etiquetados como sin etiquetar.
        - nodos_etiquetados: numpy.ndarray
            Índices de los nodos etiquetados.
        - nodos_sin_etiquetar: numpy.ndarray
            Índices de los nodos sin etiquetar.
        - k_vecinos: int
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
        self.k_vecinos = k_vecinos
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

        lista_knn = self.encuentra_knn()
        lista_mknn = self.encuentra_mknn(lista_knn)
        self.conectar_minima_distancia(lista_mknn)
        grafo_dist_min = deepcopy(self.grafo)
        # Componente = subgrafo
        componentes = self.encontrar_componentes()   
        self.conectar_componentes(componentes, lista_knn)

        return lista_knn, lista_mknn, grafo_dist_min, self.grafo

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
            lista_knn[i] = [ind_v for _, ind_v in distancias[:self.k_vecinos]]
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

        for v in range(len(self.nodos)):
            componente_v = comp[v]
            if not comp_etiquetados[componente_v]:
                for vk in lista_knn[v]:
                    componente_vk = comp[vk]
                    if comp_etiquetados[componente_vk]:
                        if vk not in self.grafo[v]:
                            self.grafo[v].append(vk)
                        if v not in self.grafo[vk]:
                            self.grafo[vk].append(v)
