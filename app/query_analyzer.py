from .rag import load_rag_components

retriever, llm = load_rag_components()


def handle_query(question: str):

    # -------- 1. Retrieve from documents --------
    docs = retriever.invoke(question) or []

    if docs:
        context = "\n\n".join(doc.page_content for doc in docs)

        prompt = f"""
You are a strict document-based assistant.

Rules:
- Answer ONLY using the context.
- Do NOT use outside knowledge.
- If answer is not explicitly in context, respond exactly:
  "NOT_FOUND"

Context:
{context}

Question:
{question}

Answer:
"""

        response = llm.invoke(prompt)
        answer = response.content.strip()

        if answer and answer != "NOT_FOUND":
            return {
                "question": question,
                "answer": answer
            }

    # -------- 2. Fallback to General Knowledge --------
    general_prompt = f"""
You are a knowledgeable academic assistant.

The question was not found in the official documents.

Provide a clear, helpful, and professional general answer.

Question:
{question}

Answer:
"""

    response = llm.invoke(general_prompt)

    return {
        "question": question,
        "answer": response.content.strip()
    }
