from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()
engine = create_engine('sqlite:///qubot.db')
Session = sessionmaker(bind=engine)
session = Session()

def init_db(app):
    Base.metadata.create_all(engine)
    app .teardown_appcontext(close_db)

def close_db(exception):
    session.close()
