from matplotlib import pyplot as plt
import networkx as nx
from collections import deque, defaultdict
import numpy as np
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from copy import deepcopy
class Gbili:
    def __init__( self, datos_se, datos_e, etiquetas_e, list_colors, K):
        self.datos_sin_etiquetar = datos_se
        self.datos_etiquetados = datos_e
        self.K = K
        #Union de los nodos etiquetados y no etiquetados
        self.vertices = np.concatenate((self.datos_etiquetados, self.datos_sin_etiquetar), axis=0)
        print("Vertices: ", self.vertices)
        print()
        self.nodos_etiquetados = range(len(self.datos_etiquetados))
        self.matriz_distancias = distance_matrix(self.vertices, self.vertices)
        self.grafo = {}
        
        self.etiquetas_modificadas = list_colors
        #Inferencia
        self.etiquetas_etiquetados = etiquetas_e
        self.n_categorias = len(np.unique(self.etiquetas_etiquetados))
        print(np.unique(self.etiquetas_etiquetados))
        self.Y = self.inicializar_Y()


    def construir_grafo(self):
        """_summary_
        """
        # Configurar la figura y los ejes
        fig, axs = plt.subplots(2, 2, figsize=(12, 6))  # 2 fila, 2 columnas
        colores_map = np.array(self.etiquetas_modificadas)

        
        lista_knn = self.encuentra_knn()
        print("LISTA KNN: ", lista_knn)
        print()
        self.dibujar_grafo(lista_knn, colores_map, axs[0, 0], "K-NN")
        lista_mknn = self.encuentra_mknn(lista_knn)
        print("LISTA MKNN: ", lista_mknn)
        print()
        self.dibujar_grafo(lista_mknn, colores_map, axs[0, 1], "Mutual K-NN")
        self.conectar_minima_distancia(lista_mknn)
        #A esta altura debe haber varios subgrafos desconectados en el grafo
        print("GRAFO: ", self.grafo)
        print()
        if self.grafo == lista_mknn:
            print("El grafo es igual a lista_mknn")
        grafo_1 = deepcopy(self.grafo)
        # Componente = subgrafo
        componentes = self.encontrar_componentes()
        print("COMPONENTES: ", componentes)
        print()        
        self.conectar_componentes(componentes, lista_knn)
        print("GRAFO CON COMPONENTES CONECTADOS: ", self.grafo)
        print()        
        grafo_2 = deepcopy(self.grafo)
        print("Componentes despues de conectar: ", self.encontrar_componentes())
        
        self.dibujar_grafo(grafo_1, colores_map, axs[1, 0], "Grafo antes de conectar componentes")
        self.dibujar_grafo(grafo_2, colores_map, axs[1, 1], "Grafo después de conectar componentes")
        # Crear una leyenda
        custom_lines = [Line2D([0], [0], color='yellow', lw=3),
                        Line2D([0], [0], color='blue', lw=3),
                        Line2D([0], [0], color='green', lw=3),
                        Line2D([0], [0], color='grey', lw=3)]

        fig.legend(custom_lines, ['0', '1', '2', 'Desconocido'])
        plt.show()
        return self.grafo
        
    def encuentra_knn(self):
        """ Encuentra los k vecinos más cercanos de cada vértice.
        <<knn>>: k-nearest neighbors

        Returns:
            dict: contiene los k vecinos más cercanos de cada vértice.
        """
        n = len(self.vertices)
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
        Almacena esos enlaces en otro diccionario que representa el grafo.
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
                # Si el nodo no está en el grafo, se agrega
                if min_enlace[0] not in self.grafo:
                    self.grafo[min_enlace[0]] = []
                if min_enlace[1] not in self.grafo:
                    self.grafo[min_enlace[1]] = []
                # Se conectan los nodos
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
            vértices alcanzables desde 'v' en orden de 
            distancia. Utiliza una cola para mantener un seguimiento de los 
            vértices a visitar.

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
        
        for v in range(len(self.vertices)):
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
        for v in range(len(self.vertices)):
            componente_v = comp[v]
            # Verificar si la componente de v no tiene nodos etiquetados
            if not comp_etiquetados[componente_v]:
                for vk in lista_knn[v]:
                    componente_vk = comp[vk]
                    # Verificar si la componente de vk tiene nodos etiquetados
                    if comp_etiquetados[componente_vk]:
                            
                        if vk not in self.grafo[v]:
                            self.grafo[v].append(vk)
                            print("Conectando nodos: ", v, vk)
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
                
        colors = list(map(lambda x: 'grey' if x==-1 else 'yellow' if x==0 else 'blue' if x==1 else 'green', colores_map))        # Dibujar el grafo
        nx.draw(G, ax=ax, with_labels=True, node_color=colors, edge_color='gray', node_size=50, font_size=10, font_weight='bold')
            
        
        ax.set_title(titulo)
        
    def inicializar_Y(self):
        """ Inicializa la matriz de etiquetas Y (mascara). 
        Se inicializa con ceros y se asigna un 1 en la columna correspondiente a la etiqueta."""
        Y = np.zeros((len(self.vertices), self.n_categorias))
        
        for i, label in enumerate(self.etiquetas_etiquetados):
            Y[i, label] = 1
        return Y       
    
    def construir_matriz_afinidad(self, sigma=1):
        matriz_distancias_grafo = self.construir_matriz_distancias_grafo()
        # sigma es el parámetro de escala para la función exponencial
        W = np.exp(-matriz_distancias_grafo**2 / (2 * sigma**2))
        np.fill_diagonal(W, 0)  # Poner la diagonal a 0
        return W  
        
    def construir_matriz_distancias_grafo(self):
        # Crear una matriz de ceros del tamaño correcto
        matriz_distancias = np.zeros((len(self.grafo), len(self.grafo)))

        # Rellenar la matriz con 1s donde hay una arista en el grafo
        for nodo, vecinos in self.grafo.items():
            for vecino in vecinos:
                matriz_distancias[nodo][vecino] = 1
        np.savetxt("matriz_distancias.txt", matriz_distancias, fmt="%d")
        return matriz_distancias
        
    def normalizar_afinidad(self, W):
        D = np.diag(W.sum(axis=1))
        D_inversa = np.diag(1 / np.sqrt(D.diagonal()))
        S = D_inversa @ W @ D_inversa
        return S   
        
    def iterar_F(self, S, alpha=0.5, tol=1e-6):
        F = deepcopy(self.Y)
        while True: # TODO agregar condición de parada, ya que puede no converger o tardar mucho
            F_next = alpha * S @ F + (1 - alpha) * self.Y
            if np.linalg.norm(F_next - F) < tol:
                break
            F = F_next
        return F
        
    def predecir_etiquetas(self, F):
        return np.argmax(F, axis=1)

    def inferir_etiquetas(self):
        W = self.construir_matriz_afinidad()
        S = self.normalizar_afinidad(W)
        F_final = self.iterar_F(S)
        return self.predecir_etiquetas(F_final)
    
    
    
## Ejemplo de uso
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
iris = load_iris()

x = iris.data
y = iris.target
K = 7

L, U, L_, U_ = train_test_split(x, y, test_size=0.5, stratify=y, random_state=42)
U_labels = deepcopy(U_)
U_labels[:] = -1
all_labels = np.concatenate((L_, U_labels))
# print("All labels: ", all_labels)
# print("L: ", L)
# print("U: ", U)
# print("L_: ", L_)
# print("U_: ", U_)
print("X: ", x)
print("L: ", L)
print("U: ", U)

def encontrar_indices_multiples(vectores, subconjuntos):
    # Inicializar una máscara de False con longitud igual al número de filas en vectores
    mask_total = np.zeros(len(vectores), dtype=bool)
    # Iterar sobre cada subconjunto
    for subconjunto in subconjuntos:
        # Crear una máscara para cada subconjunto
        mask = np.all(np.equal(vectores, subconjunto), axis=1)
        # Combinar la máscara con la máscara total usando OR lógico
        mask_total = np.logical_or(mask_total, mask)
    # Obtener los índices donde hay coincidencias
    indices = np.where(mask_total)[0]
    return indices

# Llamar a la función y obtener el resultado
indices = encontrar_indices_multiples(x, L)
print("Índices de los subconjuntos:", indices)

solver = Gbili(U, L, L_,all_labels, K)
grafo = solver.construir_grafo()
# predicciones = solver.inferir_etiquetas()


# # TODO Para comparar habria que quitar los que ya conoce
# predicciones = predicciones[len(L):]
# print("Predicciones: ", predicciones)
# #e = np.concatenate((L_, U_))
# print("Etiquetas reales: ", U_)
# print("Accuracy: ", np.mean(predicciones == U_))


