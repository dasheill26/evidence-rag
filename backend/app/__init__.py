import os
from flask import Flask, send_from_directory
from flask_cors import CORS

FRONTEND_DIST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "static")
TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "templates")


def create_app():
    app = Flask(
        __name__,
        static_folder=os.path.abspath(FRONTEND_DIST),
        template_folder=os.path.abspath(TEMPLATES_DIR),
    )
    CORS(app)
    app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10MB upload cap

    from app.routes import bp
    app.register_blueprint(bp)

    return app
