"""
FastAPI bridge between the HTML/CSS/JS frontend (static/) and chatbot.py's
RAG + memory logic. Browser JS can't call Gemini/LangChain directly, so this
tiny server exposes one endpoint and reuses rag_chat() as-is.

Run:
    uv run uvicorn server:app --reload
Then open http://127.0.0.1:8000/
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from chatbot import rag_chat, conversation_history

app = FastAPI()


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    reply = rag_chat(request.message)
    return ChatResponse(reply=reply)


@app.post("/api/reset")
def reset():
    conversation_history.clear()
    return {"status": "ok"}


app.mount("/", StaticFiles(directory="static", html=True), name="static")
