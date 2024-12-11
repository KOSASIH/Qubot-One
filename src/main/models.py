from sqlalchemy import Column, Integer, String
from database import Base

class Qubot(Base):
    __tablename__ = 'qubots'

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    status = Column(String, default='idle')

    def __repr__(self):
        return f"<Qubot(name={self.name}, status={self.status})>"
