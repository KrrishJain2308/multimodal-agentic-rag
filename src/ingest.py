import os
from typing import List
from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_STORE_DIR = os.path.join(BASE_DIR, "data", "chroma_db")

def initialize_vector_store() -> Chroma:
    """Initialize the ChromaDB vector store with HuggingFace embeddings."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vectorstore = Chroma(
        collection_name="multimodal_rag",
        embedding_function=embeddings,
        persist_directory=VECTOR_STORE_DIR
    )
    return vectorstore

def ingest_document(file_path: str):
    """Ingest a document based on its extension, split it, and add to Vector Store."""
    print(f"Loading document: {file_path}")
    
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.pdf':
        loader = PyPDFLoader(file_path)
    elif ext == '.csv':
        loader = CSVLoader(file_path)
    elif ext == '.docx':
        loader = UnstructuredWordDocumentLoader(file_path)
    elif ext == '.pptx':
        loader = UnstructuredPowerPointLoader(file_path)
    elif ext == '.xlsx':
        loader = UnstructuredExcelLoader(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
        
    docs = loader.load()
    
    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(docs)
    
    print(f"Adding {len(splits)} chunks to Vector Store in batches...")
    vectorstore = initialize_vector_store()
    
    # Chroma DB has a hard limit on batch sizes (usually 5461). We process in safe batches of 4000.
    BATCH_SIZE = 4000
    for i in range(0, len(splits), BATCH_SIZE):
        batch = splits[i:i + BATCH_SIZE]
        vectorstore.add_documents(documents=batch)
        print(f"Added batch {i//BATCH_SIZE + 1} ({len(batch)} chunks)")
        
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
