from models import Qubot
from database import session

def create_qubot(name):
    new_qubot = Qubot(name=name)
    session.add(new_qubot)
    session.commit()
    return new_qubot

def get_qubots():
    return session.query(Qubot).all()
