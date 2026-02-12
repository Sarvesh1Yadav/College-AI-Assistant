from .rag import load_rag_components

retriever, groq_llm = load_rag_components()

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

    prompt = f"""
Answer strictly using context.
If answer not found respond exactly: NOT_FOUND

Context:
{context}

Question:
{question}
"""

    response = groq_llm.invoke(prompt)
    answer = response.content.strip()

    if answer == "NOT_FOUND" or len(answer) < 15:
        return general_answer(question)

    confidence = min(95, 60 + len(docs)*5)

    conversation_memory.append({"q": question, "a": answer})

    return {
        "answer": answer,
        "doc_type": doc_type,
        "classification": doc_type
    }


def general_answer(question):
    fallback = groq_llm.invoke(question)
    conversation_memory.append({"q": question, "a": fallback.content})
    return {
        "answer": fallback.content,
        "doc_type": "General Knowledge",
        "classification": doc_type
    }
