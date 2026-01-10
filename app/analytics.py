from sqlalchemy.orm import Session
from .models import StudentQuery
from .database import SessionLocal

def store_query(question, topic):
    db: Session = SessionLocal()
    db_query = StudentQuery(question=question, topic=topic)
    db.add(db_query)
    db.commit()
    db.close()

def analytics_summary():
    db = SessionLocal()
    queries = db.query(StudentQuery).all()
    db.close()

    summary = {}
    for q in queries:
        summary[q.topic] = summary.get(q.topic, 0) + 1

    return summary
