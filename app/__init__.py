from flask import Flask
import warnings
import matplotlib
matplotlib.use("Agg")
warnings.filterwarnings("ignore")

def create_app():
    app = Flask(__name__, template_folder="templates", static_folder="static")

    with app.app_context():
        from .routes import bp as main_bp
        app.register_blueprint(main_bp)

    return app
