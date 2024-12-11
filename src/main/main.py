from flask import Flask
from config import Config
from logger import setup_logging
from api import register_api
from database import init_db

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Setup logging
    setup_logging()

    # Initialize database
    init_db(app)

    # Register API routes
    register_api(app)

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host='0.0.0.0', port=5000)
