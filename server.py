
import sys, os, re, traceback

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ.setdefault("GROQ_API_KEY", "")

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ── Safe import of RAG ──
try:
    from app.query_analyzer import handle_query
    RAG_READY = True
    print("[OK] RAG pipeline loaded")
except Exception as e:
    RAG_READY = False
    handle_query = None
    print(f"[FAIL] RAG pipeline: {e}")
    traceback.print_exc()

# ── General knowledge LLM ──
general_llm = None
if os.environ.get("GROQ_API_KEY"):
    try:
        from langchain_groq import ChatGroq
        general_llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.3,
            api_key=os.environ["GROQ_API_KEY"]
        )
        print("[OK] General LLM ready")
    except Exception as e:
        print(f"[FAIL] General LLM: {e}")

# ══════════════════════════════════════════════════
# NOT-FOUND DETECTION — 30+ patterns
# ══════════════════════════════════════════════════
_NOT_FOUND = [
    r"(?i)don'?t have (any )?(information|data|details|record|knowledge)",
    r"(?i)(not |isn'?t |aren'?t )?(mentioned|found|available|provided|specified|stated|discussed|covered|included|given) (in|from|within)? ?(the )?(document|uploaded|provided|context|source|text|data|information|rag)",
    r"(?i)no (information|data|details|mention|record|relevant|specific|clear) (found|available|provided|given|mentioned)",
    r"(?i)i (can|could)not (find|determine|answer|provide|locate|see|identify)",
    r"(?i)unable to (find|determine|answer|provide|locate|access|retrieve)",
    r"(?i)(i'?m|i am) (not sure|unsure|unable|sorry|afraid)",
    r"(?i)the (document|context|provided|source|data|text|information|upload) (does not|doesn'?t) (contain|have|mention|provide|say|include|state|address|cover|specify)",
    r"(?i)not (specifically|explicitly|directly|clearly) (mentioned|stated|discussed|addressed|covered|provided|specified)",
    r"(?i)there is (no|not any|nothing) (information|data|mention|detail|content|relevant)",
    r"(?i)cannot (find|determine|answer|provide|locate) (any|this|that|the) (information|answer|detail|data)",
    r"(?i)beyond (the )?(scope|range) of (the )?(document|provided|context|text|source)",
    r"(?i)please (refer|check|consult|contact) (the|your) (official|college|department|administration|website|handbook)",
    r"(?i)does not (say|state|mention|indicate|specify|reveal) anything (about|regarding|related to)",
    r"(?i)no (such|relevant|specific) (information|detail|data|mention|content) (is |was )?(available|found|provided|given|present)",
    r"(?i)answer (is )?not (available|present|found|provided|mentioned|contained)",
    r"(?i)(context|document|text|source) (does|did) not (provide|contain|have|give|include)",
    r"(?i)cannot be (determined|answered|found|inferred) from (the )?(document|context|provided|text|source)",
    r"(?i)not (enough|sufficient|adequate) (information|data|context|detail)",
    r"(?i)i do not (have|know|possess) (this|that|any|the|enough) (information|data|detail)",
    r"(?i)there('s| is) no (way|method) to (determine|answer|know|find)",
    r"(?i)unclear (from|in) (the )?(document|context|provided|text)",
    r"(?i)the (provided|uploaded|given) (documents?|data|information|context) (do not|don'?t) (mention|contain|specify|state|have)",
]

# ══════════════════════════════════════════════════
# POSITIVE signals — answer likely FROM documents
# ══════════════════════════════════════════════════
_FROM_DOCS = [
    r"(?i)(according to|as per|based on|mentioned in|stated in|from the|the document says|the document states|the policy says|as stated|as mentioned)\b",
    r"(?i)(chapter|section|page|rule|clause|regulation|article|clause)\s*[\d\.]+",
    r"(?i)(the college|our college|this college|the institute|the university)\s+(has|offers|provides|requires|states|maintains|follows|mandates)",
    r"(?i)(students?( are| must| should| need| have| will| can)|the student)\b",
    r"(?i)(\d+)\s*(percent|%)",
    r"(?i)(fee|rs\.?|rupees|INR)\s*[\d,]+",
    r"(?i)(semester|academic year|academic session|term)\s*[\d\-/]+",
    r"(?i)(placement|training|recruitment)\s+(cell|department|process|procedure)",
]

def _is_from_docs(answer: str) -> bool:
    """Triple-check: negative patterns, positive patterns, length heuristic."""
    text = answer.strip()
    if not text or len(text) < 60:
        return False

    # 1. Strong negative → definitely NOT from docs
    for p in _NOT_FOUND:
        if re.search(p, text):
            return False

    # 2. Strong positive → likely FROM docs
    for p in _FROM_DOCS:
        if re.search(p, text):
            return True

    # 3. Length heuristic: long detailed answer without negatives → probably from docs
    if len(text) > 250:
        return True

    # 4. Short answer without any positive signal → uncertain, assume not
    return False


def _get_general_answer(question: str) -> str:
    if not general_llm:
        return "I don't have enough information about this. Please try rephrasing or contact the college administration directly."
    try:
        resp = general_llm.invoke(
            f"A college student asked: \"{question}\"\n\n"
            f"This was NOT found in the college's official documents. "
            f"Give a helpful answer based on common practices in Indian engineering colleges. "
            f"Be concise (3-5 sentences). "
            f"If too specific to one college, say so honestly."
        )
        return resp.content.strip()
    except Exception as e:
        return f"General knowledge lookup failed: {str(e)}"


def smart_query(question: str) -> dict:
    # RAG not available
    if not RAG_READY:
        g = _get_general_answer(question)
        return {
            "answer": "⚠️ **Document system unavailable.** RAG pipeline failed to load.\n\nBased on general knowledge:\n\n" + g,
            "from_docs": False,
        }

    # Call RAG — wrapped in broad try/except
    try:
        result = handle_query(question)

        # Handle different return formats
        if isinstance(result, dict):
            rag_answer = result.get("answer", "").strip()
            # Some implementations return from_docs directly
            if "from_docs" in result:
                if not result["from_docs"]:
                    g = _get_general_answer(question)
                    return {
                        "answer": "⚠️ **Answer not found in the uploaded documents.**\n\nBased on general knowledge:\n\n" + g,
                        "from_docs": False,
                    }
                return {"answer": rag_answer, "from_docs": True}
        elif isinstance(result, str):
            rag_answer = result.strip()
        else:
            rag_answer = str(result).strip()

        # Heuristic check
        if _is_from_docs(rag_answer):
            return {"answer": rag_answer, "from_docs": True}

        # Not from docs
        g = _get_general_answer(question)
        return {
            "answer": "⚠️ **Answer not found in the uploaded documents.**\n\nBased on general knowledge:\n\n" + g,
            "from_docs": False,
        }

    except Exception as e:
        err_msg = str(e)
        traceback.print_exc()
        # If RAG itself crashes, try general answer
        g = _get_general_answer(question)
        return {
            "answer": f"⚠️ **Document query failed** (`{err_msg[:80]}`).\n\nBased on general knowledge:\n\n" + g,
            "from_docs": False,
        }


# ── FAQ questions ──
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

_faq_cache = {}


# ══════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════
HTML_PATH = os.path.join(PROJECT_ROOT, "index.html")


@app.route("/")
def index():
    return send_file(HTML_PATH)


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "rag": RAG_READY,
        "llm": general_llm is not None,
        "faq_cached": len(_faq_cache),
    })


@app.route("/ask", methods=["POST"])
def ask():
    try:
        question = request.json.get("question", "").strip()
        if not question:
            return jsonify({"error": "Please enter a question"}), 400
        result = smart_query(question)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/faq-answer")
def faq_answer():
    try:
        question = request.args.get("q", "").strip()
        if not question:
            return jsonify({"error": "No question"}), 400
        if question in _faq_cache:
            return jsonify(_faq_cache[question])
        result = smart_query(question)
        _faq_cache[question] = result
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/faq-list")
def faq_list():
    return jsonify(FAQ_QUESTIONS)


if __name__ == "__main__":
    print(f"\n{'='*50}")
    print(f"  College AI Assistant")
    print(f"  RAG Ready : {RAG_READY}")
    print(f"  Gen LLM   : {'Yes' if general_llm else 'No'}")
    print(f"  FAQ Items : {len(FAQ_QUESTIONS)}")
    print(f"{'='*50}\n")
    app.run(host="0.0.0.0", port=7860, debug=False)
