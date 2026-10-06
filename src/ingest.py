import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

VECTOR_STORE_DIR = "./data/chroma_db"

def initialize_vector_store() -> Chroma:
    """Initialize the ChromaDB vector store with Google embeddings."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
    
    vectorstore = Chroma(
        collection_name="multimodal_rag",
        embedding_function=embeddings,
        persist_directory=VECTOR_STORE_DIR
    )
    return vectorstore

def ingest_document(file_path: str):
    """Ingest a PDF document, split it, and add to Vector Store."""
    print(f"Loading document: {file_path}")
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(docs)
    
    print(f"Adding {len(splits)} chunks to Vector Store...")
    vectorstore = initialize_vector_store()
    vectorstore.add_documents(documents=splits)
    print("Ingestion complete.")

def clear_database():
    """Clear the existing ChromaDB vector store."""
    import shutil
    if os.path.exists(VECTOR_STORE_DIR):
        shutil.rmtree(VECTOR_STORE_DIR)
        print("Vector database cleared.")

if __name__ == "__main__":
    # Example usage
    # ingest_document("./data/sample.pdf")
    pass
