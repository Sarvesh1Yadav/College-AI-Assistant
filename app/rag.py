import os
import logging
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

load_dotenv()
logger = logging.getLogger(__name__)

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq

import google.generativeai as genai

# ---------- LOAD DOCUMENTS WITH TAGS ----------

def load_documents():
    docs = []
    docs_path = "data/college_docs"

    for file in os.listdir(docs_path):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(docs_path, file))
            loaded = loader.load()

            for d in loaded:
                if "UGC" in file:
                    d.metadata["type"] = "ugc"
                elif "Placement" in file:
                    d.metadata["type"] = "placement"
                else:
                    d.metadata["type"] = "msc"

            docs.extend(loaded)

    return docs


def load_rag_components():

    docs = load_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(docs)

    # ---------- VECTOR EMBEDDING ----------
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_db = FAISS.from_documents(chunks, embeddings)

    # ---------- BM25 ----------
    tokenized_corpus = [doc.page_content.split() for doc in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    # ---------- HYBRID RETRIEVER ----------
    def hybrid_retriever(query, doc_type=None):
        vector_docs = vector_db.similarity_search(query, k=6)

        if doc_type:
            vector_docs = [d for d in vector_docs if d.metadata["type"] == doc_type]

        bm25_scores = bm25.get_scores(query.split())
        top_bm25 = sorted(
            zip(bm25_scores, chunks),
            key=lambda x: x[0],
            reverse=True
        )[:6]

        bm25_docs = [doc for _, doc in top_bm25]

        combined = vector_docs + bm25_docs

        unique = {doc.page_content: doc for doc in combined}.values()

        return list(unique)

    # ---------- GROQ LLM ----------
    groq_llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.0
    )

    # ---------- GEMINI LLM ----------
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")

    return hybrid_retriever, groq_llm, gemini_model
