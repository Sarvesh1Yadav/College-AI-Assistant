import os
import logging
import re
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import BaseRetriever
from langchain_groq import ChatGroq


def load_rag_components():

    # -------- 1. Load PDFs --------
    docs = []
    docs_path = "data/college_docs"

    if os.path.isdir(docs_path):
        for file in os.listdir(docs_path):
            if file.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(docs_path, file))
                loaded = loader.load()

                # Attach metadata for better filtering later
                for d in loaded:
                    d.metadata["source_file"] = file

                docs.extend(loaded)

    if not docs:
        logger.warning("No PDF documents found.")

    # -------- 2. Structured Splitting (Improved) --------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,           # larger chunk for legal text
        chunk_overlap=150,        # higher overlap
        separators=["\n\n", "\n", ".", " "]
    )

    chunks = splitter.split_documents(docs)

    logger.info(f"Total chunks created: {len(chunks)}")

    # -------- 3. Better Embedding Model --------
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # -------- 4. FAISS Vector DB --------
    vector_db = FAISS.from_documents(chunks, embeddings)

    retriever = vector_db.as_retriever(
        search_type="mmr",   # Maximal Marginal Relevance
        search_kwargs={
            "k": 6,          # Retrieve more
            "fetch_k": 10
        }
    )

    logger.info("Improved FAISS retriever created.")

    # -------- 5. LLM --------
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.0  # deterministic for extraction
    )

    return retriever, llm
