from .rag import load_rag_components

retriever, llm = load_rag_components()


def handle_query(question: str):
    # Step 1: Retrieve relevant chunks ONLY from uploaded PDFs
    docs = retriever.get_relevant_documents(question)

    if not docs:
        return {
            "question": question,
            "answer": "The information is not specified in the provided documents."
        }

    # Step 2: Combine retrieved document text
    context = " ".join(doc.page_content for doc in docs)
    context = " ".join(context.split())

    # Step 3: Strict, document-grounded prompt
    prompt = f"""
You are a document-grounded question answering assistant.

Rules:
- Answer ONLY using the information present in the context.
- Do NOT use outside knowledge or assumptions.
- Do NOT give examples, explanations, or scenarios.
- If the answer is not present in the context, reply exactly:
  "The information is not specified in the provided documents."
- Answer in at most 2 sentences.

Context (from uploaded documents only):
{context}

Question:
{question}

Answer:
"""

    # Step 4: Get LLM response
    response = llm.invoke(prompt)
    answer = response if isinstance(response, str) else response.content
    answer = answer.strip()

    # Step 5: Hard limit to 2 sentences (safety guard)
    sentences = answer.split(".")
    answer = ".".join(sentences[:2]).strip()
    if answer and not answer.endswith("."):
        answer += "."

    return {
        "question": question,
        "answer": answer
    }
