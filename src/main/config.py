import os

class Config:
    DEBUG = os.getenv('DEBUG', 'False') == 'True'
    DATABASE_URI = os.getenv('DATABASE_URI', 'sqlite:///qubot.db')
    SECRET_KEY = os.getenv('SECRET_KEY', 'your_secret_key')
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
