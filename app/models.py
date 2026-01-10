from sqlalchemy import Column, Integer, String, DateTime
from datetime import datetime
from .database import Base

class StudentQuery(Base):
    __tablename__ = "student_queries"

    id = Column(Integer, primary_key=True, index=True)
    question = Column(String)
    topic = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
