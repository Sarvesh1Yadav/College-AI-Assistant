import os
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------- LangChain / Community ----------
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except Exception:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain.schema import BaseRetriever
except Exception:
    from langchain_core.retrievers import BaseRetriever

# ---------- Local LLM (Ollama) ----------
from langchain_ollama import ChatOllama


def load_rag_components():
    """
    Loads documents, builds FAISS retriever with local embeddings,
    and initializes a local LLM (Ollama).
    Returns (retriever, llm).
    """

    # -------- 1. Load PDFs --------
    docs = []
    docs_path = "data/college_docs"

    if os.path.isdir(docs_path):
        for file in os.listdir(docs_path):
            if file.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(docs_path, file))
                docs.extend(loader.load())
    else:
        logger.warning("Documents path not found: %s", docs_path)

    if not docs:
        logger.warning("No PDF documents found in data/college_docs")

    # -------- 2. Split documents --------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)

    # -------- 3. Optional skip flag --------
    skip_flag = os.getenv("SKIP_EMBEDDINGS_ON_STARTUP", "false").lower()
    skip_embeddings = skip_flag in ("1", "true", "yes")

    class EmptyRetriever(BaseRetriever):
        def get_relevant_documents(self, query, **kwargs):
            return []

    # -------- 4. Build retriever (LOCAL embeddings) --------
    if skip_embeddings:
        logger.info("Skipping embeddings as per config.")
        retriever = EmptyRetriever()
    else:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vector_db = FAISS.from_documents(chunks, embeddings)
        retriever = vector_db.as_retriever(search_kwargs={"k": 3})

        logger.info("FAISS retriever created successfully.")

    # -------- 5. Local LLM (Ollama) --------
    llm = ChatOllama(
        model="phi",
        temperature=0
    )

    # Ensure the returned retriever exposes a synchronous
    # `get_relevant_documents(query)` method expected by the rest
    # of the code. Some retriever implementations only expose
    # `_get_relevant_documents` or async variants.
    def _wrap_retriever(r):
        if hasattr(r, "get_relevant_documents"):
            return r

        class RetrieverAdapter(BaseRetriever):
            inner: object

            class Config:
                arbitrary_types_allowed = True

            def __init__(self, inner):
                super().__init__(inner=inner)

            def get_relevant_documents(self, query, **kwargs):
                # Prefer direct method if available
                if hasattr(self.inner, "get_relevant_documents"):
                    return self.inner.get_relevant_documents(query, **kwargs)

                # Fall back to private sync implementation
                if hasattr(self.inner, "_get_relevant_documents"):
                    try:
                        return self.inner._get_relevant_documents(query, **kwargs)
                    except TypeError:
                        # Some implementations expect a run_manager kw-only arg
                        return self.inner._get_relevant_documents(query, run_manager=None, **kwargs)

                # Fall back to async method
                if hasattr(self.inner, "aget_relevant_documents"):
                    import asyncio
                    try:
                        return asyncio.run(self.inner.aget_relevant_documents(query, **kwargs))
                    except TypeError:
                        try:
                            return asyncio.run(self.inner.aget_relevant_documents(query, run_manager=None, **kwargs))
                        except RuntimeError:
                            loop = asyncio.get_event_loop()
                            return loop.run_until_complete(self.inner.aget_relevant_documents(query, run_manager=None, **kwargs))
                    except RuntimeError:
                        # If event loop is already running, run in it
                        loop = asyncio.get_event_loop()
                        return loop.run_until_complete(self.inner.aget_relevant_documents(query, **kwargs))

                raise AttributeError("inner retriever has no method to get relevant documents")

            # Provide the abstract/low-level methods expected by BaseRetriever
            def _get_relevant_documents(self, query, **kwargs):
                return self.get_relevant_documents(query, **kwargs)

            async def aget_relevant_documents(self, query, **kwargs):
                return self.get_relevant_documents(query, **kwargs)

        return RetrieverAdapter(r)

    retriever = _wrap_retriever(retriever)

    return retriever, llm
