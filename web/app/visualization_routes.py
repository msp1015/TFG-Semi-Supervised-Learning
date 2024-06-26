import json
import os

from flask import flash, render_template, request, redirect, session, url_for, Blueprint, current_app, abort, jsonify
from flask_babel import gettext
from flask_login import login_required, current_user

from .models import Run

visualization_bp = Blueprint('visualization_bp', __name__)


@visualization_bp.route('/<algoritmo>', methods=['POST'])
def visualizar_algoritmo(algoritmo):
    """Centraliza la carga de la página de visualización.
    Es el paso siguiente después de la configuración

    :param algoritmo: nombre del algoritmo.
    :returns: - función de redirección si no se ha seleccionado ningún parámetro.
              - función que genera la página.
    """

    if 'target' not in request.form:
        flash(gettext("You must select the parameters of the algorithm"),
              category='error')
        return redirect(url_for('configuration_bp.configurar_algoritmo', algoritmo="None"))

    # En este punto se deben recoger todos los parámetros
    # que el usuario introdujo en el formulario de configuración
    params = []
    # Controlar si se trata de grafos o no
    es_grafo = False
    if session['ALGORITMO'] == "selftraining":
        params = parametros_selftraining()
    elif session['ALGORITMO'] == "cotraining":
        params = parametros_cotraining()
    elif session['ALGORITMO'] == "democraticcolearning":
        params = parametros_democraticcolearning_tritraining()
    elif session['ALGORITMO'] == "tritraining":
        params = parametros_democraticcolearning_tritraining()
    elif session['ALGORITMO'] == "coforest":
        params = parametros_coforest()
    elif session['ALGORITMO'] == "graphs":
        params = parametros_grafos()
        es_grafo = True
        nombreGrafo = request.form['constructor']
        nombreInferencia = request.form['inferencia']
    """En params se encontrarán todos los datos necesarios para ejecutar el algoritmo.
    Realmente no se le pasa la información ejecutada, se realiza una petición POST
    desde Javascript con estos parámetros al renderizar la plantilla del algoritmo."""
    if es_grafo:
        return render_template('visualizacion/graphs.html',
                               nombreGrafo=nombreGrafo,
                               nombreInferencia=nombreInferencia,
                               params=params,
                               ejecutar=True)
    
    return render_template('visualizacion/' + session['ALGORITMO'] + '.html',
                           params=params,
                           cx=request.form.get('cx', 'C1'),
                           cy=request.form.get('cy', 'C2'),
                           ejecutar=True)


@visualization_bp.route('/<algoritmo>/<run_id>', methods=['GET'])
@login_required
def visualizar_algoritmo_json(algoritmo, run_id):
    """
    Centraliza la carga de la página de visualización con una ejecución previa

    :param algoritmo: nombre del algoritmo.
    :param run_id: identificador de la ejecución previa.
    :return: función que genera la página.
    """

    run = Run.query.filter(Run.id == run_id).first()

    if not run:
        abort(404)

    if run.user_id != current_user.id:
        abort(401)

    session['ALGORITMO'] = algoritmo
    session['FICHERO'] = os.path.join(
        current_app.config['CARPETA_DATASETS_REGISTRADOS'], run.filename)

    with open(os.path.join(current_app.config['CARPETA_RUNS'], run.jsonfile)) as f:
        json_data = json.load(f)
        
    if algoritmo == "graphs":
        return render_template('visualizacion/graphs.html',
                               nombreGrafo=run.graph_constructor,
                               nombreInferencia=run.graph_inference,
                               ejecutar=False,
                               json_data=json_data)
    
    return render_template('visualizacion/' + algoritmo + '.html',
                           params=[],
                           cx=run.cx,
                           cy=run.cy,
                           ejecutar=False,
                           json_data=json_data)

def parametros_grafos():
    """
    Función auxiliar que obtiene todos los campos
    obtenidos del formulario de Grafos

    :return: lista de parámetros (en forma de diccionario).
    """

    constructor = request.form['constructor']
    inferencia = request.form['inferencia']
    # Estos son los parámetros concretos de Grafos
    params = [
        {"nombre": "constructor", "valor": request.form['constructor']},
        {"nombre": "inferencia", "valor": request.form['inferencia']},
        {"nombre": "target", "valor": request.form.get('target')},
        {"nombre": "p_unlabelled", "valor": request.form.get('p_unlabelled')}
    ]

    # Los parámetros anteriores no incluyen los propios parámetros de los clasificadores
    # (SVM, GaussianNB...), esta función lo incluye
    incorporar_clasificadores_grafos_params([constructor, inferencia], params)
    return params
def parametros_selftraining():
    """
    Función auxiliar que obtiene todos los campos
    obtenidos del formulario de Self-Training

    :return: lista de parámetros (en forma de diccionario).
    """

    clasificador = request.form['clasificador1']

    # Estos son los parámetros concretos de Self-Training
    params = [
        {"nombre": "clasificador1", "valor": request.form['clasificador1']},
        {"nombre": "n", "valor": request.form.get(
            'n', -1) if not request.form.get('n', -1) == "" else -1},
        {"nombre": "th", "valor": request.form.get('th', -1)},
        {"nombre": "n_iter", "valor": request.form.get('n_iter')},
        {"nombre": "target", "valor": request.form.get('target')},
        {"nombre": "cx", "valor": request.form.get('cx', 'C1')},
        {"nombre": "cy", "valor": request.form.get('cy', 'C2')},
        {"nombre": "pca", "valor": request.form.get('pca', 'n')},
        {"nombre": "stand", "valor": request.form.get('stand', 'n')},
        {"nombre": "p_unlabelled", "valor": request.form.get('p_unlabelled')},
        {"nombre": "p_test", "valor": request.form.get('p_test')},
    ]

    # Los parámetros anteriores no incluyen los propios parámetros de los clasificadores
    # (SVM, GaussianNB...), esta función lo incluye
    incorporar_clasificadores_inductivos_params([clasificador], params)

    return params


def parametros_cotraining():
    """
    Función auxiliar que obtiene todos los campos
    obtenidos del formulario de Co-Training

    :return: lista de parámetros (en forma de diccionario).
    """

    clasificador1 = request.form['clasificador1']
    clasificador2 = request.form['clasificador2']

    # Estos son los parámetros concretos de Co-Training
    params = [
        {"nombre": "clasificador1", "valor": request.form['clasificador1']},
        {"nombre": "clasificador2", "valor": request.form['clasificador2']},
        {"nombre": "p", "valor": request.form.get('p', -1)},
        {"nombre": "n", "valor": request.form.get('n', -1)},
        {"nombre": "u", "valor": request.form.get('u', -1)},
        {"nombre": "n_iter", "valor": request.form.get('n_iter')},
        {"nombre": "target", "valor": request.form.get('target')},
        {"nombre": "cx", "valor": request.form.get('cx', 'C1')},
        {"nombre": "cy", "valor": request.form.get('cy', 'C2')},
        {"nombre": "pca", "valor": request.form.get('pca', 'n')},
        {"nombre": "stand", "valor": request.form.get('stand', 'n')},
        {"nombre": "p_unlabelled", "valor": request.form.get('p_unlabelled')},
        {"nombre": "p_test", "valor": request.form.get('p_test')},
    ]

    # Los parámetros anteriores no incluyen los propios parámetros de los clasificadores
    # (SVM, GaussianNB...), esta función lo incluye
    incorporar_clasificadores_inductivos_params([clasificador1, clasificador2], params)

    return params


def parametros_democraticcolearning_tritraining():
    """
    Función auxiliar que obtiene todos los campos
    obtenidos del formulario de Democratic Co-Learning y Tri-Training

    :return: lista de parámetros (en forma de diccionario).
    """

    clasificador1 = request.form['clasificador1']
    clasificador2 = request.form['clasificador2']
    clasificador3 = request.form['clasificador3']

    # Estos son los parámetros concretos de Democratic Co-Learning
    params = [
        {"nombre": "clasificador1", "valor": request.form['clasificador1']},
        {"nombre": "clasificador2", "valor": request.form['clasificador2']},
        {"nombre": "clasificador3", "valor": request.form['clasificador3']},
        {"nombre": "target", "valor": request.form.get('target')},
        {"nombre": "cx", "valor": request.form.get('cx', 'C1')},
        {"nombre": "cy", "valor": request.form.get('cy', 'C2')},
        {"nombre": "pca", "valor": request.form.get('pca', 'n')},
        {"nombre": "stand", "valor": request.form.get('stand', 'n')},
        {"nombre": "p_unlabelled", "valor": request.form.get('p_unlabelled')},
        {"nombre": "p_test", "valor": request.form.get('p_test')},
    ]

    # Los parámetros anteriores no incluyen los propios parámetros de los clasificadores
    # (SVM, GaussianNB...), esta función lo incluye
    lista_clasificadores = [clasificador1, clasificador2, clasificador3]
    incorporar_clasificadores_inductivos_params(lista_clasificadores, params)

    if len(set(lista_clasificadores)) != len(lista_clasificadores) and session['ALGORITMO'] == "democraticcolearning":
        flash(gettext(
            "Classifiers must be different (diverse) to ensure proper execution"), category='warning')
    return params


def parametros_coforest():
    """
    Función auxiliar que obtiene todos los campos
    obtenidos del formulario de Co-Forest

    :return: lista de parámetros (en forma de diccionario).
    """
    # El Co-Forest unicamente usa arbóles de decisión por lo que no hay un div en el formulario
    clasificador = "DecisionTreeClassifier"

    # Estos son los parámetros concretos de Co-Forest
    params = [
        {"nombre": "n_arboles", "valor": request.form.get('n_arboles')},
        {"nombre": "W_inicial", "valor": request.form.get('W_inicial')},
        {"nombre": "theta", "valor": request.form.get('theta')},
        {"nombre": "target", "valor": request.form.get('target')},
        {"nombre": "cx", "valor": request.form.get('cx', 'C1')},
        {"nombre": "cy", "valor": request.form.get('cy', 'C2')},
        {"nombre": "pca", "valor": request.form.get('pca', 'n')},
        {"nombre": "stand", "valor": request.form.get('stand', 'n')},
        {"nombre": "p_unlabelled", "valor": request.form.get('p_unlabelled')},
        {"nombre": "p_test", "valor": request.form.get('p_test')},
    ]
    incorporar_clasificadores_inductivos_params([clasificador], params)
    return params


def incorporar_clasificadores_inductivos_params(nombre_clasificadores, params):
    """
    Incluye los parámetros de los propios clasificadores base
    a la lista de parámetros generales para luego realizar la petición
    e instanciar los clasificadores con dichos parámetros.
    """

    with open(os.path.join(os.path.dirname(__file__), os.path.normpath("static/json/parametros.json"))) as f:
        clasificadores = json.load(f)
    clasificadores = clasificadores["Inductive"]
    for i, clasificador in enumerate(nombre_clasificadores):
        for key in clasificadores[clasificador].keys():
            params.append({"nombre": f"clasificador{i + 1}_" + key,
                           "valor": request.form.get(f"clasificador{i + 1}_" + key, -1)})

def incorporar_clasificadores_grafos_params(nombre_clasificadores, params):
    """
    Incluye los parámetros de los propios clasificadores base
    a la lista de parámetros generales para luego realizar la petición
    e instanciar los clasificadores con dichos parámetros.
    """

    with open(os.path.join(os.path.dirname(__file__), os.path.normpath("static/json/parametros.json"))) as f:
        clasificadores = json.load(f)
    clasificadores_inferencia = clasificadores["Inference"]
    clasificadores_grafos = clasificadores["Graphs"]
    for i, clasificador in enumerate(nombre_clasificadores):
        if clasificador in clasificadores_inferencia.keys():
            for key in clasificadores_inferencia[clasificador].keys():
                params.append({"nombre": f"inferencia_" + key,
                               "valor": request.form.get(f"inferencia_" + key, -1)})
        elif clasificador in clasificadores_grafos.keys():
            for key in clasificadores_grafos[clasificador].keys():
                params.append({"nombre": f"constructor_" + key,
                               "valor": request.form.get(f"constructor_" + key, -1)})
