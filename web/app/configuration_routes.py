import json
import os
import sys

from flask import flash, render_template, redirect, session, url_for, Blueprint, current_app, abort
from flask_babel import gettext
from .forms import FormConfiguracionGrafos, FormConfiguracionSelfTraining, FormConfiguracionCoTraining, FormConfiguracionSingleView, FormConfiguracionCoForest

src_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(src_path)
from algoritmos.utilidades.datasetloader import DatasetLoader

configuration_bp = Blueprint('configuration_bp', __name__)


@configuration_bp.route('/<algoritmo>', methods=['GET'])
def configurar_algoritmo(algoritmo):
    """
    Renderiza la página de configuración de todos los algoritmos,
    para ello genera el formulario correspondiente dependiente de "algoritmo"

    :param algoritmo: nombre del algoritmo.
    :return: función que genera la página.
    """

    if 'FICHERO' not in session:
        flash(gettext("You must upload a file"), category='error')
        return redirect(url_for('main_bp.subida'))

    filename = session['FICHERO'].rsplit("-", 1)[0]
    if not filename.upper().endswith('.ARFF'):
        if not filename.upper().endswith('.CSV'):
            flash(gettext("File must be ARFF or CSV"), category='warning')
            session.pop('FICHERO', None)
            return redirect(url_for('main_bp.subida'))

    if algoritmo not in current_app.config['ALGORITMOS_SELECCIONABLES']:
        abort(404)

    session['ALGORITMO'] = algoritmo

    dl = DatasetLoader(session['FICHERO'])
    with open(os.path.join(os.path.dirname(__file__), os.path.normpath("static/json/parametros.json"))) as f:
        clasificadores = json.load(f)

    with open(os.path.join(os.path.dirname(__file__), os.path.normpath("static/json/grafos.json"))) as f:
        construccion_grafos = json.load(f)
    
    with open(os.path.join(os.path.dirname(__file__), os.path.normpath("static/json/inferencia.json"))) as f:
        inferencia_grafos = json.load(f)
        

    caracteristicas = list(dl.get_allfeatures())
    clasificadores_keys = list(clasificadores.keys())
    construccion_grafos_keys = list(construccion_grafos.keys())
    inferencia_grafos_keys = list(inferencia_grafos.keys())
    if session['ALGORITMO'] == "selftraining":
        form = FormConfiguracionSelfTraining()
        form.clasificador1.choices = clasificadores_keys
    elif session['ALGORITMO'] == "cotraining":
        form = FormConfiguracionCoTraining()
        form.clasificador1.choices = clasificadores_keys
        form.clasificador2.choices = clasificadores_keys
    elif session['ALGORITMO'] == "coforest":
        form = FormConfiguracionCoForest()
    elif session['ALGORITMO'] == "graphs":
        form = FormConfiguracionGrafos()
        form.constructor.choices = construccion_grafos_keys 
        form.inferencia.choices = inferencia_grafos_keys
    else:
        form = FormConfiguracionSingleView()
        form.clasificador1.choices = clasificadores_keys
        form.clasificador2.choices = clasificadores_keys
        form.clasificador3.choices = clasificadores_keys

    form.sel_target.choices = caracteristicas
    form.sel_target.default = caracteristicas[-1]
    form.cx.choices = caracteristicas
    form.cy.choices = caracteristicas
    
    # Aplica los valores por defecto que han sido definidos 
    # después de cargar el formulario
    form.process()
    return render_template('configuracion/' + algoritmo + 'config.html',
                           parametros=inferencia_grafos, 
                           form=form)
