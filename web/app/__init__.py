import os

from flask import Flask, request, render_template, session
from flask_babel import Babel, gettext, lazy_gettext
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_wtf import CSRFProtect
from werkzeug.middleware.proxy_fix import ProxyFix

db = SQLAlchemy()
DB_NAME = "db.db"


def create_app():
    app = Flask(__name__,
                static_url_path='',
                static_folder='./static',
                )
    app.wsgi_app = ProxyFix(
        app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1
    )
    app.config['SESSION_TYPE'] = 'filesystem'
    app.secret_key = "ae4c977b14e2ecf38d485869018ec8f924b312132ee3d11d1ce755cdff9bc0af"
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_NAME}'
    app.config.update(SESSION_COOKIE_SAMESITE='Strict')
    app.config['CARPETA_DATASETS_REGISTRADOS'] = os.path.join(os.path.basename(os.path.dirname(__file__)),
                                                              'datasets',
                                                              'registrados')
    app.config['CARPETA_DATASETS_ANONIMOS'] = os.path.join(os.path.basename(os.path.dirname(__file__)),
                                                           'datasets',
                                                           'anonimos')
    app.config['CARPETA_RUNS'] = os.path.join(
        os.path.basename(os.path.dirname(__file__)), 'runs')
    app.config['ALGORITMOS_SELECCIONABLES'] = ["selftraining", "cotraining", "democraticcolearning",
                                               "tritraining", "coforest", "graphs"]
    db.init_app(app)

    def get_locale():
        """
        Devuelve el idioma que haya seleccionado el usuario
        en la sesión. Si no ha seleccionado ninguno, se devuelve
        el mejor posible entre español e inglés

        :return: idioma
        """

        idioma = session.get('IDIOMA', None)
        if idioma is not None:  # Ha seleccionado un idioma manualmente
            return idioma

        return request.accept_languages.best_match(['es', 'en'])

    Babel(app, locale_selector=get_locale)
    CSRFProtect(app)

    @app.context_processor
    def variables_globales():
        """
        Define una serie de variables globales accesibles
        desde Jinja

        :return: diccionario con variables globales.
        """

        return {'titulos': {'selftraining': 'Self-Training',
                            'cotraining': 'Co-Training',
                            'democraticcolearning': 'Democratic Co-Learning',
                            'tritraining': 'Tri-Training',
                            'coforest': 'Co-Forest',
                            'graphs': gettext('Graphs')},
                'idiomas': {'en': gettext('English'),
                            'es': gettext('Spanish')},
                'idioma_actual': get_locale()}

    @app.before_request
    def before_request():
        """
        Actúa antes de cualquier petición comprobando
        el establecimiento de un idioma (?lang=).
        Lo guarda en la sesión del usuario
        """

        session.permanent = False
        if 'lang' in request.args:
            session['IDIOMA'] = request.args.get('lang')

    @app.template_filter()
    def nombredataset(text):
        """Obtiene solo el nombre del conjunto de datos
        eliminando la ruta completa"""

        return os.path.split(text.rsplit("-", 1)[0])[1]

    from .main_routes import main_bp
    from .configuration_routes import configuration_bp
    from .visualization_routes import visualization_bp
    from .data_routes import data_bp
    from .user_routes import users_bp

    app.register_blueprint(main_bp, url_prefix='/')
    app.register_blueprint(users_bp, url_prefix='/')
    app.register_blueprint(configuration_bp, url_prefix='/configuracion')
    app.register_blueprint(visualization_bp, url_prefix='/visualizacion')
    app.register_blueprint(data_bp, url_prefix='/datos')

    from .models import User, Dataset, Run

    with app.app_context():
        db.create_all()

        admin = User.query.filter_by(email='admin@admin.es').first()

        if not admin:
            admin = User()
            admin.name = 'Admin'
            admin.password = 'pbkdf2:sha256:600000$KZpGx9NV4Y8Qp7Ct$4e6f1333477946598d46f88b65717c28e843d25f9a67973dbe780395e5c39829'
            admin.email = 'admin@admin.es'
            admin.admin = True
            db.session.add(admin)
            db.session.commit()

    login_manager = LoginManager()
    login_manager.login_view = 'users_bp.login'
    login_manager.login_message = lazy_gettext(
        'Please log in to access this page.')
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

    template_error = 'error.html'

    @app.errorhandler(Exception)
    def critical_error(e):
        return render_template(template_error, error='Critical error', mensaje=str(e)), 500

    @app.errorhandler(500)
    def internal_server_error(e):
        return render_template(template_error, error=500, mensaje=gettext('Internal Server Error')), 500

    @app.errorhandler(404)
    def not_found(e):
        return render_template(template_error, error=404, mensaje=gettext('Not found')), 404

    @app.errorhandler(403)
    def forbidden(e):
        return render_template(template_error, error=403, mensaje=gettext('Forbidden')), 403

    @app.errorhandler(401)
    def unauthorized(e):
        return render_template(template_error, error=401, mensaje=gettext('Unauthorized')), 401

    @app.errorhandler(400)
    def bad_request(e):
        return render_template(template_error, error=400, mensaje=gettext('Bad request')), 400

    return app
