import csv
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings

DB_PATH = "chroma_db"
OUTPUT_FILE = "rag_evaluation_results.csv"

EVAL_QUESTIONS = [
    {
        "question": "How does SINDy help in low-data regimes?",
        "expected_keywords": ["sparse", "low-data", "training data", "overfitting", "governing equations"]
    },
    {
        "question": "What is Model Predictive Control?",
        "expected_keywords": ["optimization", "prediction", "constraints", "control", "horizon"]
    },
    {
        "question": "What are Physics-Informed Neural Networks?",
        "expected_keywords": ["physics", "neural network", "differential equations", "loss", "residual"]
    },
    {
        "question": "Compare PINNs and SINDy for nonlinear system identification.",
        "expected_keywords": ["PINN", "SINDy", "dynamics", "data", "physics"]
    },
    {
        "question": "What are the limitations of SINDy?",
        "expected_keywords": ["library", "noise", "high-dimensional", "derivatives", "sparse"]
    }
]

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embeddings
)

retriever = db.as_retriever(
    search_kwargs={"k": 10}
)

llm = OllamaLLM(
    model="llama3"
)

def build_context(docs):
    context = ""

    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "Unknown source")
        page = doc.metadata.get("page", "Unknown page")

        if isinstance(page, int):
            page = page + 1

        context += f"\n[Source {i}] {source}, page {page}\n"
        context += doc.page_content
        context += "\n"

    return context

def ask_rag(question):
    docs = retriever.invoke(question)
    context = build_context(docs)

    prompt = f"""
You are Scientific ML Copilot, an expert assistant in PINNs, SINDy, MPC,
dynamical systems, and scientific machine learning.

Use ONLY the retrieved context below.

Answer clearly and technically.
Every answer must include at least one source citation.
Use citations in this exact format: [Source 1], [Source 2].
Do not answer without citations.

Question:
{question}

Retrieved Context:
{context}

Answer:
"""

    answer = llm.invoke(prompt)

    return answer, docs

def keyword_score(answer, expected_keywords):
    answer_lower = answer.lower()
    hits = 0

    for keyword in expected_keywords:
        if keyword.lower() in answer_lower:
            hits += 1

    return hits / len(expected_keywords)

def citation_score(answer):
    if "[Source" in answer:
        return 1
    return 0

def retrieval_score(docs, expected_keywords):
    retrieved_text = " ".join([doc.page_content for doc in docs]).lower()
    hits = 0

    for keyword in expected_keywords:
        if keyword.lower() in retrieved_text:
            hits += 1

    return hits / len(expected_keywords)

def main():
    rows = []

    for item in EVAL_QUESTIONS:
        question = item["question"]
        expected_keywords = item["expected_keywords"]

        print(f"\nEvaluating: {question}")

        answer, docs = ask_rag(question)

        k_score = keyword_score(answer, expected_keywords)
        c_score = citation_score(answer)
        r_score = retrieval_score(docs, expected_keywords)

        rows.append({
            "question": question,
            "keyword_score": round(k_score, 2),
            "citation_score": c_score,
            "retrieval_score": round(r_score, 2),
            "answer": answer
        })

        print(f"Keyword Score: {k_score:.2f}")
        print(f"Citation Score: {c_score}")
        print(f"Retrieval Score: {r_score:.2f}")

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "question",
                "keyword_score",
                "citation_score",
                "retrieval_score",
                "answer"
            ]
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"\nEvaluation complete. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()