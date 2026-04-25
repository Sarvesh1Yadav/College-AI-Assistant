%%writefile /content/College-AI-Assistant/server.py
import sys, os, re, threading
from concurrent.futures import ThreadPoolExecutor

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ.setdefault("GROQ_API_KEY", "")

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ── Import RAG handler (graceful if unavailable) ──
try:
    from app.query_analyzer import handle_query
    RAG_AVAILABLE = True
    print("[OK] RAG pipeline imported successfully")
except Exception as e:
    RAG_AVAILABLE = False
    handle_query = None
    print(f"[WARN] Could not import handle_query: {e}")

# ── General knowledge LLM (fast model, no documents) ──
general_llm = None
if os.environ.get("GROQ_API_KEY"):
    try:
        from langchain_groq import ChatGroq
        general_llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.3,
            api_key=os.environ["GROQ_API_KEY"]
        )
        print("[OK] General knowledge LLM ready")
    except Exception as e:
        print(f"[WARN] Could not init general LLM: {e}")

# ── Patterns that indicate RAG found nothing ──
_NOT_FOUND_PATTERNS = [
    r"(?i)don'?t have (any )?(information|data|details|record)",
    r"(?i)(not |isn'?t )?(mentioned|found|available|provided|specified|stated|discussed) (in |from )?(the )?(document|uploaded|provided|context|source)",
    r"(?i)no (information|data|details|mention|record|relevant) (found|available|provided)",
    r"(?i)i (can|could)not (find|determine|answer|provide|locate)",
    r"(?i)unable to (find|determine|answer|provide|locate)",
    r"(?i)(i'?m|i am) (not sure|unsure|unable|sorry)",
    r"(?i)the (document|context|provided|source|data) (does not|doesn'?t) (contain|have|mention|provide|say|include)",
    r"(?i)not (specifically|explicitly) (mentioned|stated|discussed|addressed|covered)",
    r"(?i)there is (no|not any) (information|data|mention|detail)",
    r"(?i)cannot (find|determine|answer|provide) (any|this|that) (information|answer|detail)",
    r"(?i)beyond (the )?(scope|range) of (the )?(document|provided|context)",
    r"(?i)please (refer|check|consult) (the|your) (official|college|department)",
]


def _is_from_documents(answer: str) -> bool:
    """Check if the RAG answer genuinely came from documents."""
    if not answer or len(answer.strip()) < 40:
        return False
    for pattern in _NOT_FOUND_PATTERNS:
        if re.search(pattern, answer):
            return False
    return True


def _get_general_answer(question: str) -> str:
    """Get answer from LLM general knowledge (no documents)."""
    if not general_llm:
        return "I don't have enough information to answer this question. Please try rephrasing or ask about a different topic."
    try:
        prompt = (
            f"A college student asked: \"{question}\"\n\n"
            f"The answer was NOT found in the college's official documents. "
            f"Provide a helpful answer based on common practices in Indian colleges/universities. "
            f"Keep it concise (3-6 sentences max). "
            f"If the question is too specific to one particular college, say so honestly."
        )
        resp = general_llm.invoke(prompt)
        return resp.content.strip()
    except Exception as e:
        return f"Could not generate general answer: {str(e)}"


def smart_query(question: str) -> dict:
    """
    Core query function:
    1. Try RAG (documents) first
    2. If answer looks like 'not found' → say so + give general knowledge answer
    """
    # RAG not available at all
    if not RAG_AVAILABLE:
        general = _get_general_answer(question)
        return {
            "answer": (
                "⚠️ **Document system is not available.** "
                "The RAG pipeline could not be loaded.\n\n"
                "Based on general knowledge:\n\n" + general
            ),
            "from_docs": False,
        }

    # Call RAG
    try:
        result = handle_query(question)
        rag_answer = result.get("answer", "").strip()

        # If the caller already provides from_docs info, respect it
        if "from_docs" in result:
            if not result["from_docs"]:
                general = _get_general_answer(question)
                return {
                    "answer": (
                        "⚠️ **Answer not found in the uploaded documents.**\n\n"
                        "Based on general knowledge:\n\n" + general
                    ),
                    "from_docs": False,
                }
            return {"answer": rag_answer, "from_docs": True}

        # Heuristic check
        if _is_from_documents(rag_answer):
            return {"answer": rag_answer, "from_docs": True}

        # Not found in docs → fallback
        general = _get_general_answer(question)
        return {
            "answer": (
                "⚠️ **Answer not found in the uploaded documents.**\n\n"
                "Based on general knowledge:\n\n" + general
            ),
            "from_docs": False,
        }

    except Exception as e:
        return {
            "answer": f"Error querying documents: {str(e)}",
            "from_docs": False,
        }


# ── FAQ questions (answered lazily from your documents) ──
FAQ_QUESTIONS = [
    {"q": "What is the admission process and eligibility criteria?", "cat": "academic"},
    {"q": "What is the fee structure for different courses?", "cat": "fees"},
    {"q": "What are the placement statistics and top recruiting companies?", "cat": "placement"},
    {"q": "What is the minimum attendance requirement for exams?", "cat": "academic"},
    {"q": "What scholarships are available and how to apply?", "cat": "fees"},
    {"q": "What are the hostel facilities and room allocation rules?", "cat": "general"},
    {"q": "What is the exam pattern, grading system, and credit structure?", "cat": "academic"},
    {"q": "What library resources and digital facilities are available?", "cat": "general"},
    {"q": "What is the fee refund and cancellation policy?", "cat": "fees"},
    {"q": "What documents are required during admission?", "cat": "academic"},
    {"q": "What sports and extracurricular activities are offered?", "cat": "general"},
    {"q": "How do I register for campus placements?", "cat": "placement"},
]

# Cache for loaded FAQ answers
_faq_cache: dict = {}
_faq_lock = threading.Lock()


# ── Routes ──
HTML_PATH = os.path.join(PROJECT_ROOT, "index.html")


@app.route("/")
def index():
    return send_file(HTML_PATH)


@app.route("/ask", methods=["POST"])
def ask():
    try:
        question = request.json.get("question", "").strip()
        if not question:
            return jsonify({"error": "Please enter a question"}), 400
        result = smart_query(question)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/faq-answer")
def faq_answer():
    """Return answer for a single FAQ question (lazy-loaded)."""
    question = request.args.get("q", "").strip()
    if not question:
        return jsonify({"error": "No question provided"}), 400

    # Check cache
    with _faq_lock:
        if question in _faq_cache:
            return jsonify(_faq_cache[question])

    # Query
    result = smart_query(question)

    # Cache
    with _faq_lock:
        _faq_cache[question] = result

    return jsonify(result)


@app.route("/faq-list")
def faq_list():
    """Return the list of FAQ questions (no answers, fast)."""
    return jsonify(FAQ_QUESTIONS)


if __name__ == "__main__":
    port = 7860
    print(f"\n{'='*50}")
    print(f"  College AI Assistant")
    print(f"  RAG Available: {RAG_AVAILABLE}")
    print(f"  General LLM: {'Yes' if general_llm else 'No'}")
    print(f"  FAQ Questions: {len(FAQ_QUESTIONS)}")
    print(f"{'='*50}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
