import chromadb
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")

# Create collection
collection = client.get_or_create_collection(
    name="company_documents"
)


# Load document
def load_document():
    with open("documents/sample.txt", "r", encoding="utf-8") as file:
        return file.read()


# Split document into chunks
def create_chunks(text):
    chunk_size = 300

    return [
        text[i:i + chunk_size]
        for i in range(0, len(text), chunk_size)
    ]


# Add documents to ChromaDB
def add_documents():
    text = load_document()
    chunks = create_chunks(text)

    embeddings = model.encode(chunks).tolist()

    collection.upsert(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings
    )

    print("Documents added successfully!")
    print("Number of chunks:", len(chunks))


# Search documents
def search_documents(question):
    question_embedding = model.encode([question]).tolist()

    results = collection.query(
        query_embeddings=question_embedding,
        n_results=2
    )

    return results["documents"][0]


if __name__ == "__main__":
    add_documents()

    question = "What technologies does the company use?"

    results = search_documents(question)

    print("\nQuestion:")
    print(question)

    print("\nRelevant information:")

    for result in results:
        print("\n----------------")
        print(result)