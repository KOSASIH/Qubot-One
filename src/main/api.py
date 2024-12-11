from flask import Blueprint, jsonify

api_bp = Blueprint('api', __name__)

@api_bp.route('/status', methods=['GET'])
def get_status():
    return jsonify({"status": "Qubot-One is running!"})

def register_api(app):
    app.register_blueprint(api_bp, url_prefix='/api')
