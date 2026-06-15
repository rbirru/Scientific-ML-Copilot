import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

CSV_FILE = "rag_evaluation_results.csv"

st.set_page_config(
    page_title="RAG Evaluation Dashboard",
    layout="wide"
)

st.title("RAG Evaluation Dashboard")
st.write("Evaluation results for Scientific ML Copilot Using RAG")

df = pd.read_csv(CSV_FILE)

st.subheader("Evaluation Results Table")
st.dataframe(df)

avg_keyword = df["keyword_score"].mean()
avg_citation = df["citation_score"].mean()
avg_retrieval = df["retrieval_score"].mean()

st.subheader("Average Scores")

col1, col2, col3 = st.columns(3)

col1.metric("Keyword Score", f"{avg_keyword:.2f}")
col2.metric("Citation Score", f"{avg_citation:.2f}")
col3.metric("Retrieval Score", f"{avg_retrieval:.2f}")

st.subheader("Scores by Question")

score_cols = [
    "keyword_score",
    "citation_score",
    "retrieval_score"
]

for score in score_cols:
    st.markdown(f"### {score.replace('_', ' ').title()}")

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(
        df["question"],
        df[score]
    )

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_xlabel("Question")
    ax.set_title(score.replace("_", " ").title())

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    st.pyplot(fig)

st.subheader("Overall Average Score Comparison")

avg_df = pd.DataFrame({
    "Metric": [
        "Keyword Score",
        "Citation Score",
        "Retrieval Score"
    ],
    "Average Score": [
        avg_keyword,
        avg_citation,
        avg_retrieval
    ]
})

fig, ax = plt.subplots(figsize=(8, 5))

ax.bar(
    avg_df["Metric"],
    avg_df["Average Score"]
)

ax.set_ylim(0, 1.05)
ax.set_ylabel("Average Score")
ax.set_title("Average RAG Evaluation Metrics")

plt.tight_layout()

st.pyplot(fig)