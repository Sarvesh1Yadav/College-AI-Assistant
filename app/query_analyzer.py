from .rag import load_rag_components

retriever, llm = load_rag_components()


def handle_query(question: str):

    # -------- 1. Retrieve --------
    docs = retriever.get_relevant_documents(question)

    if not docs:
        return {
            "question": question,
            "answer": "The information is not specified in the provided documents."
        }

    # -------- DEBUG: Print retrieved chunks --------
    print("\n--- Retrieved Chunks ---")
    for i, d in enumerate(docs):
        print(f"\nChunk {i+1}:\n{d.page_content[:400]}")

    # -------- 2. Use FULL retrieved context --------
    context = "\n\n".join(doc.page_content for doc in docs)
    context = " ".join(context.split())

    # -------- 3. Stronger Extraction Prompt --------
    prompt = f"""
You are a strict legal document extraction assistant.

Rules:
1. Answer ONLY using exact sentences from the context.
2. Do NOT paraphrase.
3. Do NOT explain.
4. If multiple sentences are relevant, return ALL of them.
5. If not explicitly found, respond exactly:
   "The information is not specified in the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)
    answer = response.content.strip()

    # -------- 4. Safety Validation --------
    if (
        not answer
        or answer.lower().startswith("the information is not specified")
        or len(answer) < 15
    ):
        answer = "The information is not specified in the provided documents."

    return {
        "question": question,
        "answer": answer
    }
