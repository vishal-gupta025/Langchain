from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

# Sample documents
docs = [
    Document("Virat Kohli is one of the most successful batsmen in IPL history.", metadata={"team": "Royal Challengers Bangalore"}),
    Document("Rohit Sharma is the most successful captain in IPL history.", metadata={"team": "Mumbai Indians"}),
    Document("MS Dhoni has led Chennai Super Kings to multiple IPL titles.", metadata={"team": "Chennai Super Kings"}),
    Document("Jasprit Bumrah is one of the best fast bowlers in T20 cricket.", metadata={"team": "Mumbai Indians"}),
    Document("Ravindra Jadeja is a dynamic all-rounder for CSK.", metadata={"team": "Chennai Super Kings"})
]

# Initialize Chroma vector store
vector_store = Chroma(
    embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
    persist_directory="chroma_db",
    collection_name="sample"
)

# Add docs
vector_store.add_documents(docs)
vector_store.persist()

# Retrieve stored docs
print(vector_store.get(include=["embeddings", "documents", "metadatas"]))

# Similarity searches
print(vector_store.similarity_search("who among these are bowlers?", k=2))
print(vector_store.similarity_search_with_score("who among these are bowlers?", k=2))
print(vector_store.similarity_search_with_score("", filter={"team": "Chennai Super Kings"}))

# Updating a document
doc_id = "09a39dc6-3ba6-4ea7-927e-fdda591da5e4"
updated_doc1 = Document(
    page_content="Virat Kohli, former RCB captain, is the highest run-scorer in IPL history.",
    metadata={"team": "Royal Challengers Bangalore"}
)

vector_store.delete(ids=[doc_id])
vector_store.add_documents([updated_doc1], ids=[doc_id])
vector_store.persist()

# Verify
print(vector_store.get(include=["embeddings", "documents", "metadatas"]))
