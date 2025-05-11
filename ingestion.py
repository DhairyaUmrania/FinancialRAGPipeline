##This code is the intellectual property of Dhairya Umrania, Naman Deep and Devaansh Kataria.

import os
from typing import List

from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

DATA_DIR        = os.environ.get("DATA_DIR", "documents")
CHUNK_SIZE      = int(os.environ.get("CHUNK_SIZE", 1000))
CHUNK_OVERLAP   = int(os.environ.get("CHUNK_OVERLAP", 200))


def ingest_documents() -> List[Document]:
    """
    Load all PDFs under DATA_DIR, split them into text chunks, and return them.
    """
    print(f"Ingesting PDF documents from {DATA_DIR}…")

    loader = DirectoryLoader(
        DATA_DIR,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    docs = loader.load()
    print(f"  • Loaded {len(docs)} raw PDF pages/documents")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(docs)
    print(f"  • Split into {len(chunks)} chunks")

    return chunks