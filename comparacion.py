from algPatri.PatriCoForest import coforest
from algoritmos.coforest import CoForest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from algPatri.sslEnsemble import SSLEnsemble

#separamos los datos en X y Y
x = load_iris().data
y = load_iris().target

scoresMario = []
scoresPatri = []

#porcentajes_test = [0.1, 0.2, 0.3, 0.4, 0.5]
porcentajes_U = [0.5, 0.6, 0.7, 0.8, 0.9]
#separamos los datos en train y test
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, stratify=y)
#X_test e y_test se queda como esta para hacer el score despues (datos desconocidos)
for i in range(50,100,5):
    
    #La siguiente distincion es para que el algoritmo coja gran parte de datos y no los etiquete (U)
    L, U, y_l, y_u = train_test_split(X_train, y_train, test_size=i, stratify=y_train)

    algMario = CoForest(n=10, theta=0.75)
    algPatri = coforest(n=10, theta=0.75)
    
    algMario.fit(L, y_l , U)
    algPatri.fit(L, y_l, U)
    
    scoresMario.append(algMario.score(X_test, y_test))
    scoresPatri.append(algPatri.score(X_test, y_test))
    
std_patri = np.std(scoresPatri)
plt.plot(range(50,100,5), scoresMario, label='Mario')
plt.errorbar(range(50,100,5), scoresPatri, yerr=std_patri, label='Patri', fmt='-o')
#ajustar eje y
plt.ylim(0.75,1)
plt.legend()
plt.xlabel('Porcentaje de datos no etiquetados')
plt.ylabel('Score')
plt.title("Comparacion CoForest con 20% de datos para test")
plt.show()
