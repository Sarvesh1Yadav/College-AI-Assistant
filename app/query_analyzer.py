from .rag import load_rag_components

# Load retriever and local LLM once
retriever, llm = load_rag_components()


def handle_query(question: str):
    """
    Handles user queries by retrieving relevant document chunks
    and generating a strictly document-grounded answer.
    """

    # 1. Retrieve relevant document chunks
    docs = retriever.get_relevant_documents(question)

    if not docs:
        return {
            "question": question,
            "answer": "The information is not specified in the provided documents."
        }

    # 2. Use only top-k chunks to avoid noise
    context = " ".join(doc.page_content for doc in docs[:2])
    context = " ".join(context.split())

    # 3. General, domain-independent extractive prompt
    prompt = f"""
You are a document-based question answering assistant.

Instructions:
- Use ONLY the information present in the context.
- Do NOT add outside knowledge.
- Do NOT summarize or paraphrase.
- Extract and return the COMPLETE sentence(s) from the context
  that directly answer the question.
- If the answer is not explicitly present, respond exactly with:
  "The information is not specified in the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""

    # 4. Invoke the local LLM
    response = llm.invoke(prompt)
    answer = response.content.strip()

    # 5. Final safety validation
    if not answer or len(answer) < 10:
        answer = "The information is not specified in the provided documents."

    return {
        "question": question,
        "answer": answer
    }
