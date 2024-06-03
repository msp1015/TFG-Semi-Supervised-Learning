import json
import os
from datetime import datetime

from flask import request, session, Blueprint, current_app, jsonify
from flask_login import current_user
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sqlalchemy.exc import SQLAlchemyError

from algoritmos import SelfTraining, CoTraining, DemocraticCoLearning, TriTraining, CoForest, Gbili, LGC
from algoritmos.utilidades.datasetloader import DatasetLoader
from algoritmos.utilidades.datasplitter import data_split
from algoritmos.utilidades.dimreduction import log_pca_reduction, log_cxcy_reduction

from . import db
from .models import Run

data_bp = Blueprint('data_bp', __name__)


@data_bp.route('/selftraining', methods=['POST'])
def datosselftraining():
    """
    Obtiene los datos de la ejecución de Self-Training

    :return: json con la información de ejecución.
    """

    clasificador = request.form['clasificador1']

    n = int(request.form['n'])
    th = int(request.form['th'])

    try:
        st = SelfTraining(
            clf=obtener_clasificador(clasificador, obtener_parametros_clasificador(
                "Inductive", clasificador, "clasificador1")),
            n=n if n != -1 else None,
            th=th/100 if th != -1 else None,
            n_iter=int(request.form['n_iter']))

        info = obtener_info_inductivo(st)
    except ValueError as e:
        return jsonify({
            "status": "warning",
            "error": str(e)
        }), 500
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

    return json.dumps(info)


@data_bp.route('/coforest', methods=['POST'])
def datoscoforest():
    """
    Obtiene los datos de la ejecución de Co-Forest

    :return: json con la información de ejecución.
    """

    # Este form se construye desde inicializar.js con FormData
    n_arboles = int(request.form['n_arboles'])
    theta = int(request.form['theta'])
    try:
        params_arbol_decision = obtener_parametros_clasificador(
            "Inductive", "DecisionTreeClassifier", "clasificador1")
        st = CoForest(n_arboles, theta/100, params_arbol_decision)

        info = obtener_info_inductivo(st)
    except ValueError as e:
        return jsonify({
            "status": "warning",
            "error": str(e)
        }), 500
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

    return json.dumps(info)


@data_bp.route('/cotraining', methods=['POST'])
def datoscotraining():
    """
    Obtiene los datos de la ejecución de Co-Training

    :return: json con la información de ejecución.
    """

    clasificador1 = request.form['clasificador1']
    clasificador2 = request.form['clasificador2']

    try:
        ct = CoTraining(
            clf1=obtener_clasificador(clasificador1, obtener_parametros_clasificador(
                "Inductive", clasificador1, "clasificador1")),
            clf2=obtener_clasificador(clasificador2, obtener_parametros_clasificador(
                "Inductive", clasificador2, "clasificador2")),
            p=int(request.form['p']),
            n=int(request.form['n']),
            u=int(request.form['u']),
            n_iter=int(request.form['n_iter']))

        info = obtener_info_inductivo(ct)
    except ValueError as e:
        return jsonify({
            "status": "warning",
            "error": str(e)
        }), 500
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

    return json.dumps(info)


def datossingleview(is_democratic):
    """
    Obtiene los datos de la ejecución de Democratic Co-Learning
    o de Tri-Training (existen muchos pasos comunes).

    :return: json con la información de ejecución.
    """

    clasificador1 = request.form['clasificador1']
    clasificador2 = request.form['clasificador2']
    clasificador3 = request.form['clasificador3']

    clf1 = obtener_clasificador(
        clasificador1, obtener_parametros_clasificador("Inductive", clasificador1, "clasificador1"))
    clf2 = obtener_clasificador(
        clasificador2, obtener_parametros_clasificador("Inductive", clasificador2, "clasificador2"))
    clf3 = obtener_clasificador(
        clasificador3, obtener_parametros_clasificador("Inductive", clasificador3, "clasificador3"))
    try:
        if is_democratic:
            svclf = DemocraticCoLearning([clf1, clf2, clf3])
        else:
            svclf = TriTraining([clf1, clf2, clf3])

        info = obtener_info_inductivo(svclf)
    except ValueError as e:
        return jsonify({
            "status": "warning",
            "error": str(e)
        }), 500
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

    return json.dumps(info)


@data_bp.route('/democraticcolearning', methods=['POST'])
def datosdemocraticcolearning():
    """
    Obtiene los datos de la ejecución de Democratic Co-Learning

    :return: json con la información de ejecución.
    """
    return datossingleview(True)


@data_bp.route('/tritraining', methods=['POST'])
def datostritraining():
    """
    Obtiene los datos de la ejecución de Tri-Training

    :return: json con la información de ejecución.
    """

    return datossingleview(False)

@data_bp.route('/graphs', methods=['POST'])
def datosgraphs():
    """
    Obtiene los datos de la ejecución de algoritmos basados en grafos
    
    : return json con la información de ejecución
    """
    constructor = request.form['constructor']
    inferencia = request.form['inferencia']
    print(inferencia)
    datasetloader = DatasetLoader(session['FICHERO'])
    datasetloader.set_target(request.form['target'])
    p_unlabelled=int(request.form['p_unlabelled']) / 100

    x, y, mapa, is_unlabelled = datasetloader.get_x_y()
    L, U, L_y, U_y = train_test_split(x, y, test_size=p_unlabelled, stratify=y, random_state=42)

    todas_etiquetas = np.concatenate((L_y, U_y))
    params_constructor = obtener_parametros_clasificador("Graphs", constructor, "constructor")
    params_inferancia = obtener_parametros_clasificador("Inference", inferencia, "inferencia")
    steps = []
    if constructor == "Gbili":

        solver = Gbili(U, L,todas_etiquetas, **params_constructor)
        list_knn, list_mknn, distmin, grafoFinal = solver.construir_grafo()
        steps = [list_knn, list_mknn, distmin, grafoFinal]
    elif constructor == "Rgcli":
        pass
    
    if inferencia == "LocalAndGlobalConsistency":
        propagacion = LGC(grafoFinal, solver.nodos, solver.etiquetas_etiquetados, **params_inferancia)
        predicciones = propagacion.inferir_etiquetas()
    else:
        pass
        
    
    nodos_iniciales = build_nodos_json(grafoFinal, solver.etiquetas_etiquetados) 
    for i in range(len(steps)):
        steps[i] = build_enlaces_json(steps[i], solver.matriz_distancias)
    predicciones = predicciones[len(L)].tolist()
    predicciones_json = {}
    for i, prediccion in enumerate(predicciones):
        predicciones_json[str(i + len(L))] = prediccion
    info_grafos = {"nodos": nodos_iniciales, "enlaces": steps, "predicciones": predicciones}
    print(info_grafos)
    if current_user.is_authenticated:
        date = int(datetime.now().timestamp())

        with open(os.path.join(current_app.config['CARPETA_RUNS'], f'run-{current_user.id}-{date}.json'), 'w') as f:
            json.dump(info_grafos, f)
        
        run = Run()
        run.algorithm = session['ALGORITMO']
        run.json_parameters = generar_json_parametros_grafos()
        run.filename = os.path.basename(session['FICHERO'])
        run.date = datetime.now()
        run.jsonfile = f'run-{current_user.id}-{date}.json'
        run.user_id = current_user.id
        
        try:
            db.session.add(run)
        except SQLAlchemyError:
            db.session.rollback()
            os.remove(os.path.join(
                current_app.config['CARPETA_RUNS'], f'run-{current_user.id}-{date}.json'))
        else:
            db.session.commit()
    return info_grafos

def obtener_info_inductivo(algoritmo):
    """Función auxiliar que evita el código duplicado de la obtención de toda la información
    de la ejecución de los algoritmos inductivos.

    Realiza la carga de datos, las particiones de datos, el entrenamiento del algoritmo,
    la conversión a un log (logger) en 2D y la conversión a JSON para las plantillas.

    :return: diccionario con la información de ejecución.
    """

    datasetloader = DatasetLoader(session['FICHERO'])
    datasetloader.set_target(request.form['target'])

    x, y, mapa, is_unlabelled = datasetloader.get_x_y()

    (x, y, x_test, y_test) = data_split(x,
                                        y,
                                        is_unlabelled,
                                        p_unlabelled=int(
                                            request.form['p_unlabelled']) / 100,
                                        p_test=int(request.form['p_test']) / 100)

    specific_stats = None
    if isinstance(algoritmo, SelfTraining):
        log, stats, iteration = algoritmo.fit(
            x, y, x_test, y_test, datasetloader.get_only_features())
    else:
        log, stats, specific_stats, iteration = algoritmo.fit(
            x, y, x_test, y_test, datasetloader.get_only_features())

    stand = True if request.form['stand'] == 'y' else False

    if request.form['pca'] == 'y':
        _2d = log_pca_reduction(log,
                                datasetloader.get_only_features(), standardize=stand).to_json()
    else:
        _2d = log_cxcy_reduction(log,
                                 request.form['cx'],
                                 request.form['cy'],
                                 datasetloader.get_only_features(), standardize=stand).to_json()

    info = {'iterations': iteration,
            'log': _2d,
            'stats': stats.to_json(),
            'mapa': json.dumps(mapa)}

    if not isinstance(algoritmo, SelfTraining):
        info = info | {'specific_stats': {
            key: specific_stats[key].to_json() for key in specific_stats}}

    if current_user.is_authenticated:
        date = int(datetime.now().timestamp())

        with open(os.path.join(current_app.config['CARPETA_RUNS'], f'run-{current_user.id}-{date}.json'), 'w') as f:
            json.dump(info, f)

        run = Run()
        run.algorithm = session['ALGORITMO']
        run.json_parameters = generar_json_parametros()
        run.filename = os.path.basename(session['FICHERO'])
        if request.form['pca'] == 'y':
            run.cx = 'C1'
            run.cy = 'C2'
        else:
            run.cx = request.form['cx']
            run.cy = request.form['cy']
        run.date = datetime.now()
        run.jsonfile = f'run-{current_user.id}-{date}.json'
        run.user_id = current_user.id

        try:
            db.session.add(run)
        except SQLAlchemyError:
            db.session.rollback()
            os.remove(os.path.join(
                current_app.config['CARPETA_RUNS'], f'run-{current_user.id}-{date}.json'))
        else:
            db.session.commit()
    return info


def obtener_parametros_clasificador(metodo, clasificador, nombre):
    """A la hora de instanciar un clasificador (sklearn), este tiene una serie de parámetros
    (NO CONFUNDIR CON LOS PARÁMETROS DE LOS ALGORITMOS SEMI-SUPERVISADOS).
    Aclaración: estos vienen codificados en parametros.json.

    Interpreta el formulario de la configuración para obtener estos valores y que puedan ser
    desempaquetados con facilidad (**).

    :return: diccionario con los parámetros del clasificador.
    """

    with open(os.path.join(os.path.dirname(__file__), os.path.normpath("static/json/parametros.json"))) as f:
        clasificadores = json.load(f)
    clasificadores = clasificadores[metodo]
    parametros_clasificador = {}
    for key in clasificadores[clasificador].keys():
        parametro = clasificadores[clasificador][key]
        if parametro["type"] == "number" and parametro["step"] == 0.1:
            p = float(request.form[nombre + "_" + key])
            parametros_clasificador[key] = p
        elif parametro["type"] == "number" and parametro["step"] == 1:
            p = int(request.form[nombre + "_" + key])
            parametros_clasificador[key] = p
        else:
            parametros_clasificador[key] = request.form[nombre + "_" + key]
    print(parametros_clasificador)
    return parametros_clasificador


def obtener_clasificador(nombre, params):
    """Instancia un clasificador (sklearn) a partir de su nombre y los parámetros
    introducidos (provenientes de "obtener_parametros_clasificador").

    :return: instancia del clasificador.
    """

    if nombre == "SVC":
        params = params | {"probability": True}
        return SVC(**params)
    elif nombre == "GaussianNB":
        return GaussianNB()
    elif nombre == "KNeighborsClassifier":
        return KNeighborsClassifier(**params)
    elif nombre == "DecisionTreeClassifier":
        return DecisionTreeClassifier(**params)


def generar_json_parametros():
    """
    Genera un JSON como cadena de texto. Recopila toda la configuración
    de ejecución que el usuario ha introducido para ser almacenada en base de datos

    :return: json con la información de todos los parámetros (tanto de los clasificadores como del algoritmo).
    """

    with open(os.path.join(os.path.dirname(__file__), os.path.normpath("static/json/parametros.json"))) as f:
        clasificadores = json.load(f)
    clasificadores = clasificadores["Inductive"]
    formulario = dict(request.form)

    claves_clasificadores = [
        k for k, _ in request.form.items() if 'clasificador' in k and "_" not in k]
    clasificadores_reales = [
        v for k, v in request.form.items() if 'clasificador' in k and "_" not in k]
    resto_de_parametros = [
        k for k, _ in request.form.items() if "clasificador" not in k]

    pre_json = dict()

    contador = {c: 1 for c in clasificadores_reales}

    for k, clasificador_real in zip(claves_clasificadores, clasificadores_reales):
        aux = clasificador_real
        if clasificadores_reales.count(clasificador_real) > 1:
            aux += str(contador[clasificador_real])
            pre_json[aux] = dict()
            contador[clasificador_real] += 1
        else:
            pre_json[clasificador_real] = dict()

        parametros_clasificador_real = clasificadores[clasificador_real].keys()

        for p in parametros_clasificador_real:
            pre_json[aux][p] = formulario[k + "_" + p]

    for r in resto_de_parametros:
        pre_json[r] = formulario[r]

    return json.dumps(pre_json)

def generar_json_parametros_grafos():
    """_summary_

    """
    # TODO: Implementar
    
    return json.dumps({})

def build_nodos_json(grafo, etiquetas_modificadas):
    """Construye y guarda los nodos en formato JSON.

    Args:
        grafo (dict): Grafo a guardar.

    Returns:
        list: Lista de nodos.
    """
    nodos = []
    for node in grafo:
        nodos.append({"id": str(node), "group": int(etiquetas_modificadas[node])})
    return nodos

def build_enlaces_json(grafo, matriz_distancias):
    """Construye y guarda los enlaces en formato JSON.

    Args:
        grafo (dict): Grafo a guardar.

    Returns:
        list: Lista de enlaces.
    """
    enlaces = []
    for node in grafo:
        for neighbor in grafo[node]:
            if {"source": str(node), "target": str(neighbor)} not in enlaces and {"source": str(neighbor), "target": str(node)} not in enlaces:
                enlaces.append({"source": str(node), "target": str(neighbor), "value": matriz_distancias[node][neighbor]})
    return enlaces