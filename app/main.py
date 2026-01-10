from fastapi import FastAPI, Response
from .database import engine
from .models import Base
from .query_analyzer import handle_query

Base.metadata.create_all(bind=engine)

app = FastAPI(title="College AI Assistant")


@app.get("/")
def read_root():
    return {"status": "ok", "message": "College AI Assistant — POST /ask"}


@app.get("/favicon.ico")
def favicon():
    # Return 204 No Content for favicon requests to avoid 404 in logs
    return Response(status_code=204)

@app.post("/ask")
def ask_question(question: str):
    return handle_query(question)
