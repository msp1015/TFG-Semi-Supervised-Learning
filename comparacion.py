import time
from algPatri.PatriCoForest import coforest
from algoritmos.coforest import CoForest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, load_digits
from algPatri.sslEnsemble import SSLEnsemble
from algPatri.PatriCoForest import coforest

#separamos los datos en X y Y
# x = load_iris().data
# y = load_iris().target
x = load_digits().data
y = load_digits().target

scoresMario = []
scoresPatri = []

#porcentajes_test = [0.1, 0.2, 0.3, 0.4, 0.5]
porcentajes_U = [0.5, 0.6, 0.7, 0.8, 0.9]
#separamos los datos en train y test
#X_train, X_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, stratify=y)
#X_test e y_test se queda como esta para hacer el score despues (datos desconocidos)
# for i in range(50,100,5):
    
#     #La siguiente distincion es para que el algoritmo coja gran parte de datos y no los etiquete (U)
#     L, U, y_l, y_u = train_test_split(X_train, y_train, test_size=i, stratify=y_train)

#     algMario = CoForest(n=10, theta=0.75)
#     algPatri = coforest(n=10, theta=0.75)
    
#     algMario.fit(L, y_l , U)
#     algPatri.fit(L, y_l, U)
    
#     scoresMario.append(algMario.score(X_test, y_test))
#     scoresPatri.append(algPatri.score(X_test, y_test))
    
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, stratify=y)
L, U, y_l, y_u = train_test_split(X_train, y_train, test_size=0.75, stratify=y_train)

#std_patri = np.std(scoresPatri)
#Iniciar contador de tiempo para algoritmo de Mario
t_inicio_mario = time.time()
alg = CoForest(n=20, theta=0.75)
alg.fit(L, y_l, U, X_test, y_test)
t_final_mario = time.time()

#Iniciar contador de tiempo para algoritmo de Patri
t_inicio_patri = time.time()
algPatri = coforest(n=20, theta=0.75)
algPatri.fit(L, y_l, U, X_test, y_test)
t_final_patri = time.time()

print("Tiempo Mario: ", t_final_mario - t_inicio_mario)
print("Tiempo Patri: ", t_final_patri - t_inicio_patri)


marioAcc = alg.get_accuracy_por_iteracion()
patriAcc = algPatri.get_accuracy_por_iteracion()

std_patri = np.std(patriAcc)
print("Accuracy Mario: ", marioAcc)
print("Accuracy Patri: ", patriAcc)


plt.plot(range(len(marioAcc)), marioAcc, label="Mario", marker='o')
plt.errorbar(range(len(patriAcc)), patriAcc, yerr=std_patri, label='Patri', fmt='-o')
plt.xlabel('Iteraciones')
plt.xticks(range(max(len(marioAcc), len(patriAcc))))
plt.ylabel('Accuracy')
plt.ylim(0.80,1)
plt.title('Accuracy por iteracion CoForest')
plt.legend()
plt.show()

