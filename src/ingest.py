import os
import shutil
import logging
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

from src.config import (
    VECTOR_STORE_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CHROMA_BATCH_SIZE
)

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def initialize_vector_store() -> Chroma:
    """Initialize the ChromaDB vector store with HuggingFace embeddings."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=VECTOR_STORE_DIR
    )
    return vectorstore


def ingest_document(file_path: str):
    """Ingest a document based on its extension, split it, and add to Vector Store."""
    logger.info(f"Loading document: {file_path}")
    
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
    
    logger.info(f"Splitting text into chunks (chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    splits = text_splitter.split_documents(docs)
    
    logger.info(f"Adding {len(splits)} chunks to Vector Store in batches of {CHROMA_BATCH_SIZE}...")
    vectorstore = initialize_vector_store()
    
    # Chroma DB has a hard limit on batch sizes (usually 5461). We process in safe configured batches.
    for i in range(0, len(splits), CHROMA_BATCH_SIZE):
        batch = splits[i:i + CHROMA_BATCH_SIZE]
        vectorstore.add_documents(documents=batch)
        logger.info(f"Added batch {i // CHROMA_BATCH_SIZE + 1} ({len(batch)} chunks)")
        
    logger.info(f"Ingestion complete for {file_path}")


def clear_database():
    """Clear the existing ChromaDB vector store."""
    if os.path.exists(VECTOR_STORE_DIR):
        shutil.rmtree(VECTOR_STORE_DIR)
        logger.info(f"Vector database cleared from {VECTOR_STORE_DIR}.")
