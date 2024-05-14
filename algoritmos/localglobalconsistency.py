"""Este módulo contiene la implementación del algoritmo Local Global Consistency (LGC)
para la clasificación de nodos en grafos.

@Autor:     Mario Sanz Pérez
@Fecha:     08/05/2024
@Versión:   1.0
@Nombre:    localglobalconsistency.py
"""""

from copy import deepcopy
import numpy as np

class LGC:
    """ Algoritmo de inferencia de etiquetas en grafos basado en el algoritmo Local Global Consistency (LGC).
    Articulo original: Zhou, D., Bousquet, O., Lal, T. N., Weston, J., & Schölkopf, B. (2004). 
    Learning with local and global consistency.
    """
    def __init__(self, grafo, nodos, etiquetas_etiquetados, sigma=1, alpha=0.5, tol=1e-6, max_iter=1000):
            """Inicializa una instancia de la clase LocalGlobalConsistency.

            Args:
                grafo (tipo): El grafo utilizado para el algoritmo.
                nodos (tipo): Los nodos del grafo.
                etiquetas_etiquetados (tipo): Las etiquetas de los nodos etiquetados.
                sigma (float): El parámetro de escala para la función exponencial.
                alpha (float): El parámetro de suavizado entre la matriz de afinidad y las etiquetas.
                tol (float): La tolerancia para la convergencia del algoritmo.
                max_iter (int): El número máximo de iteraciones.
            """
            self.grafo = grafo
            self.nodos = nodos
            self.etiquetas_etiquetados = etiquetas_etiquetados
            self.n_categorias = len(np.unique(self.etiquetas_etiquetados))
            self.Y = self.inicializar_Y()
            self.matriz_afinidad = self.construir_matriz_afinidad()
            self.Y = self.inicializar_Y()
            self.sigma = sigma
            self.alpha = alpha
            self.tol = tol
            self.max_iter = max_iter

    def inicializar_Y(self):
        """ Inicializa la matriz de etiquetas Y (mascara). 
        Se inicializa con ceros y se asigna un 1 en la columna correspondiente a la etiqueta.
        Los nodos no etiquetados tienen todas las columnas a cero."""
        
        Y = np.zeros((len(self.nodos), self.n_categorias))
        for i, label in enumerate(self.etiquetas_etiquetados):
            Y[i, label] = 1
        return Y    
    
    def construir_matriz_afinidad(self):
        """ Construye una matriz de distancias simplificada del grafo.
        Se asume que la distancia entre nodos vecinos es 1 y el resto de distancias es 0.
        Returns:
            np.array: La matriz de distancias del grafo.
        """
        W = np.zeros((len(self.grafo), len(self.grafo)))
        for nodo, vecinos in self.grafo.items():
            for vecino in vecinos:
                W[nodo][vecino] = 1
        return W

    def inferir_etiquetas(self):
        """ Proceso de inferencia de etiquetas en el grafo.
        1. Construir la matriz de afinidad W (ponderada a 1 y 0).
        2. Normalizar la matriz de afinidad S.
        3. Iterar F hasta convergencia.
        4. Predecir las etiquetas de los nodos no etiquetados.
        
        Returns:
            np.array: Las etiquetas inferidas de los nodos no etiquetados.
        """

        # W = self.construir_matriz_afinidad(sigma=self.sigma)
        S = self.normalizar_afinidad(self.matriz_afinidad)
        F_final = self.iterar_F(S, alpha=self.alpha, tol=self.tol, max_iter=self.max_iter)
        return self.predecir_etiquetas(F_final)
    
    # def construir_matriz_afinidad(self, sigma=1):
    #     """ Construye la matriz de afinidad W a partir de la matriz de distancias ponderada.
    #     Args:
    #         sigma (float): El parámetro de escala para la función exponencial.

    #     Returns:
    #         np.array: La matriz de afinidad W.
    #     """
    #     W = np.exp(-self.matriz_afinidad**2 / (2 * sigma**2))
    #     np.fill_diagonal(W, 0)
    #     return W  

    def normalizar_afinidad(self, W):
        """ Normaliza la matriz de afinidad W.
            Args:
            W (np.array): La matriz de afinidad W.

            Returns:
            np.array: La matriz de afinidad normalizada S.
        """
        D = np.diag(W.sum(axis=1))
        D_inversa = np.diag(1 / np.sqrt(D.diagonal()))
        S = D_inversa @ W @ D_inversa
        return S

    def iterar_F(self, S, alpha=0.5, tol=1e-6, max_iter=1000):
        """ Itera la matriz de etiquetas F hasta convergencia.
        Args:
            S (np.array): La matriz de afinidad normalizada.
            alpha (float): El parámetro de suavizado entre la matriz de afinidad y las etiquetas.
            tol (float): La tolerancia para la convergencia del algoritmo.
            max_iter (int): El número máximo de iteraciones.
        Returns:
            np.array: La matriz de etiquetas F.
        """
        F = deepcopy(self.Y)
        for _ in range(max_iter):
            F_next = alpha * S @ F + (1 - alpha) * self.Y
            if np.linalg.norm(F_next - F) < tol:
                break
            F = F_next
        return F

    def predecir_etiquetas(self, F):
        """ Predice las etiquetas de los nodos no etiquetados.
        
        Args:
            F (np.array): La matriz de etiquetas F.
            
        Returns:
            np.array: Las etiquetas inferidas de los nodos no etiquetados.
        """
        return np.argmax(F, axis=1)
