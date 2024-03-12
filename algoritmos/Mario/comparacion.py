import time
from algPatri.PatriCoForest import coforest
from coforest import CoForest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, load_digits, load_wine, load_breast_cancer
from algPatri.sslEnsemble import SSLEnsemble
from algPatri.PatriCoForest import coforest


conjuntos_de_datos = {"Iris": load_iris, "Digits": load_digits, "Wine": load_wine, 
                      "Breast Cancer": load_breast_cancer}

fig, axes = plt.subplots(2, 2, figsize=(10, 10))

tiemposMario = []
tiemposPatri = []

resultadosMario = []
resultadosPatri = []

varianzasMario = []
varianzasPatri = []

for nombre, dataset in conjuntos_de_datos.items():
    
    x = dataset().data
    y = dataset().target
     
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, stratify=y)
    L, U, y_l, y_u = train_test_split(X_train, y_train, test_size=0.75, stratify=y_train)

    #std_patri = np.std(scoresPatri)
    #Iniciar contador de tiempo para algoritmo de Mario
    t_inicio_mario = time.time()
    random_state = 42
    alg = CoForest(n=20, theta=0.75, random_state=random_state)
    alg.fit(L, y_l, U, X_test, y_test)
    t_final_mario = time.time()

    #Iniciar contador de tiempo para algoritmo de Patri
    t_inicio_patri = time.time()
    algPatri = coforest(n=20, theta=0.75, random_state=random_state)
    algPatri.fit(L, y_l, U, X_test, y_test)
    t_final_patri = time.time()

    print("Tiempo Mario para " + nombre + ": ", t_final_mario - t_inicio_mario)
    print("Tiempo Patri para " + nombre + ": ", t_final_patri - t_inicio_patri)

    tiemposMario.append(t_final_mario - t_inicio_mario)
    tiemposPatri.append(t_final_patri - t_inicio_patri)

    marioAcc = alg.get_accuracy_por_iteracion()
    patriAcc = algPatri.get_accuracy_por_iteracion()
    resultadosMario.append(marioAcc)
    resultadosPatri.append(patriAcc)
    
    std_mario = np.std(marioAcc)
    std_patri = np.std(patriAcc)
    varianzasMario.append(std_mario)
    varianzasPatri.append(std_patri)
    
    
    print("Accuracy Mario para " + nombre + ": ", marioAcc)
    print("Accuracy Patri para " + nombre + ": ", patriAcc)
    

    
for i in range(2):
    for j in range(2):
        '''
        El siguiente código es para graficar las curvas de aprendizaje de los algoritmos.
        Implementado con un error bar para mostrar la varianza de los resultados.
        '''
        axes[i, j].errorbar(range(1, len(resultadosMario[i*2+j])+1), resultadosMario[i*2+j], yerr=varianzasMario[i*2+j], label="Mario", marker='o')
        axes[i, j].errorbar(range(1, len(resultadosPatri[i*2+j])+1), resultadosPatri[i*2+j], yerr=varianzasPatri[i*2+j], label="Patri", marker = 'o')
        axes[i, j].set_title(list(conjuntos_de_datos.keys())[i*2+j])
        axes[i, j].set_xlabel("Iteraciones")
        axes[i, j].set_ylabel("Accuracy")
        axes[i, j].legend()
        
plt.show()
    


