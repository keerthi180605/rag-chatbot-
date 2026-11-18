from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq
import os

from rag import search_documents
load_dotenv(override=True)
app = FastAPI(title="RAG Chatbot")

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


class Question(BaseModel):
    question: str


@app.get("/")
def home():
    return {
        "message": "RAG Chatbot is running"
    }


@app.post("/chat")
def chat(data: Question):

    # Retrieve relevant information
    relevant_documents = search_documents(data.question)

    # Combine retrieved documents
    context = "\n\n".join(relevant_documents)

    # Create prompt
    prompt = f"""
You are a helpful AI assistant.

Answer the question using only the context below.

If the answer is not available in the context,
say that you don't have enough information.

Context:
{context}

Question:
{data.question}

Answer:
"""

    # Ask Groq
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2
    )

    return {
        "question": data.question,
        "answer": response.choices[0].message.content,
        "sources": relevant_documents
    }