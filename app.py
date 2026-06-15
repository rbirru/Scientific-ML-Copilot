import streamlit as st

from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings

DB_PATH = "chroma_db"

st.set_page_config(
    page_title="Scientific ML Copilot Using RAG",
    layout="wide"
)

st.title("Scientific ML Copilot Using RAG")
st.write(
    "A local RAG assistant for PINNs, SINDy, MPC, dynamical systems, "
    "and scientific machine learning research."
)

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

if "messages" not in st.session_state:
    st.session_state.messages = []

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
You are Scientific ML Copilot, an expert assistant in:

- Physics-Informed Neural Networks (PINNs)
- Sparse Identification of Nonlinear Dynamics (SINDy)
- Model Predictive Control (MPC)
- Dynamical systems
- Control theory
- Scientific machine learning

Use ONLY the retrieved context below.

Your response should:
1. Answer the user's question clearly.
2. Synthesize information across sources when possible.
3. Explain technical concepts in an engineering-friendly way.
4. Cite sources using [Source 1], [Source 2], etc.
5. Only cite sources that directly support the specific claim.
6. Do not cite all retrieved sources.
7. Do not write broad citations like [Source 1] - [Source 10].
8. If the context is insufficient, say what is missing.
9. Every answer must include at least one source citation.
10. Use citations in this exact format: [Source 1], [Source 2].
11. Do not answer without citations.
12. Write answers in a concise technical style suitable for graduate students,
researchers, and engineering professionals.


User Question:
{question}

Retrieved Context:
{context}

Final Answer:
"""

    answer = llm.invoke(prompt)

    return answer, docs

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

question = st.chat_input("Ask about PINNs, SINDy, MPC, or Scientific ML...")

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        answer, docs = ask_rag(question)
        st.write(answer)

        with st.expander("Retrieved Sources"):
            for i, doc in enumerate(docs, start=1):
                source = doc.metadata.get("source", "Unknown source")
                page = doc.metadata.get("page", "Unknown page")
                if isinstance(page, int):
                    page = page + 1

                st.markdown(f"### Source {i}")
                st.write(f"**File:** {source}")
                st.write(f"**Page:** {page}")
                st.write(doc.page_content[:1200])

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )