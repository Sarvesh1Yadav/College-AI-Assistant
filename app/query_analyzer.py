from .rag import load_rag_components

retriever, groq_llm, gemini_model = load_rag_components()

conversation_memory = []


def classify_query(question):
    q = question.lower()
    if "ragging" in q or "ugc" in q:
        return "ugc"
    elif "placement" in q or "dream" in q:
        return "placement"
    elif "cgpa" in q or "semester" in q:
        return "msc"
    return None


def handle_query(question):

    doc_type = classify_query(question)

    docs = retriever(question, doc_type)

    context = "\n\n".join(doc.page_content for doc in docs[:6])

    if not context:
        return general_answer(question)

    # ---------- GROQ PRIMARY ----------
    prompt = f"""
Answer strictly using context.
If not found respond: NOT_FOUND

Context:
{context}

Question:
{question}
"""

    groq_response = groq_llm.invoke(prompt)
    answer = groq_response.content.strip()

    # ---------- GEMINI VALIDATION ----------
    gemini_check = gemini_model.generate_content(
        f"Is this answer correct based on context? Answer YES or NO.\n\n{answer}"
    )

    validation = gemini_check.text.strip().upper()

    if "NO" in validation or answer == "NOT_FOUND":
        return general_answer(question)

    confidence = min(95, 60 + len(docs)*5)

    conversation_memory.append({"q": question, "a": answer})

    return {
        "answer": answer,
        "confidence": f"{confidence}%",
        "memory_length": len(conversation_memory)
    }


def general_answer(question):
    fallback = groq_llm.invoke(question)
    conversation_memory.append({"q": question, "a": fallback.content})
    return {
        "answer": fallback.content,
        "confidence": "General Knowledge Mode",
        "memory_length": len(conversation_memory)
    }
