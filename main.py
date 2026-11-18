from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader
import os

from rag import search_documents, model, collection


# Load environment variables
load_dotenv(override=True)


# Create FastAPI app
app = FastAPI(title="RAG Chatbot")


# Allow frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Groq client
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


# -----------------------------
# Home
# -----------------------------

@app.get("/")
def home():
    return {
        "message": "RAG Chatbot is running"
    }


# -----------------------------
# Chat request model
# -----------------------------

class Question(BaseModel):
    question: str


# -----------------------------
# Chat endpoint
# -----------------------------

@app.post("/chat")
def chat(data: Question):

    # Retrieve relevant documents
    relevant_documents = search_documents(data.question)

    # Combine retrieved documents
    context = "\n\n".join(relevant_documents)

    # Prompt for Groq
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

    # Call Groq
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


# -----------------------------
# Upload document
# -----------------------------

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):

    # Check file type
    filename = file.filename.lower()

    if not (filename.endswith(".txt") or filename.endswith(".pdf")):
        return {
            "error": "Only TXT and PDF files are supported"
        }

    # Read uploaded file
    content = await file.read()

    # Save uploaded file
    file_path = f"documents/{file.filename}"

    with open(file_path, "wb") as buffer:
        buffer.write(content)

    # -------------------------
    # Extract text
    # -------------------------

    if filename.endswith(".txt"):

        text = content.decode("utf-8")

    else:

        # Save PDF temporarily
        reader = PdfReader(file_path)

        text = ""

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"


    # Check empty document
    if not text.strip():
        return {
            "error": "The uploaded document contains no readable text"
        }


    # -------------------------
    # Split into chunks
    # -------------------------

    chunk_size = 300

    chunks = [
        text[i:i + chunk_size]
        for i in range(0, len(text), chunk_size)
    ]


    # -------------------------
    # Create embeddings
    # -------------------------

    embeddings = model.encode(chunks).tolist()


    # -------------------------
    # Store in ChromaDB
    # -------------------------

    collection.upsert(
        ids=[
            f"{file.filename}_{i}"
            for i in range(len(chunks))
        ],
        documents=chunks,
        embeddings=embeddings
    )


    return {
        "message": "Document uploaded and indexed successfully",
        "filename": file.filename,
        "chunks": len(chunks)
    }