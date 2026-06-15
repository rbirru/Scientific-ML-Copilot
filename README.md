# Scientific-ML-Copilot
Scientific ML Copilot Using RAG

A Retrieval-Augmented Generation (RAG) assistant for scientific machine learning,
physics-informed neural networks (PINNs), sparse identification of nonlinear
dynamics (SINDy), and model predictive control (MPC).

Built using Llama 3, Ollama, ChromaDB, LangChain, HuggingFace Embeddings,
and Streamlit.

---

## Features

- Local LLM inference using Llama 3 and Ollama
- Vector search using ChromaDB
- Scientific literature ingestion from PDF papers
- Cross-document question answering
- Source-grounded responses with citations
- Evaluation framework for retrieval and answer quality
- Streamlit web interface

---

## Technology Stack

| Component | Technology |
|------------|------------|
| LLM | Llama 3 |
| Local Inference | Ollama |
| Embeddings | HuggingFace (all-MiniLM-L6-v2) |
| Vector Database | ChromaDB |
| Framework | LangChain |
| UI | Streamlit |
| Evaluation | Custom RAG Metrics |

---

## Architecture

```text
Scientific Papers (PDFs)
           │
           ▼
      Document Loader
           │
           ▼
      Text Chunking
           │
           ▼
        Embeddings
           │
           ▼
        ChromaDB
           │
           ▼
        Retriever
           │
           ▼
        Llama 3
           │
           ▼
 Scientific ML Copilot
```

---

## Project Structure

```text
scientific-ml-copilot/

├── app.py
├── ingest.py
├── evaluate.py
├── evaluation_dashboard.py
├── requirements.txt
├── README.md
├── data/
├── chroma_db/
└── screenshots/
```
Note: Users should place their PDF documents in the data/ directory and run: python ingest.py
---

## Example Questions

### SINDy

```text
How does SINDy help in low-data regimes?
```

### PINNs

```text
What are Physics-Informed Neural Networks?
```

### MPC

```text
What are the advantages of Model Predictive Control?
```

### Literature Synthesis

```text
Compare PINNs and SINDy for nonlinear system identification.
```

---

## Evaluation Metrics

The system evaluates:

- Keyword Score
- Citation Score
- Retrieval Score

Example Results:

| Metric | Score |
|----------|----------|
| Keyword Score | 0.76 |
| Citation Score | 0.80 |
| Retrieval Score | 0.80 |

---

## Installation

### Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/scientific-ml-copilot.git

cd scientific-ml-copilot
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Install Ollama Models

```bash
ollama pull llama3
```

### Build Vector Database

```bash
python ingest.py
```

### Run Application

```bash
python -m streamlit run app.py
```

---

## Screenshots

### Scientific ML Copilot

```
### Evaluation Dashboard

Add screenshot here:

```text
screenshots/evaluation_dashboard.png
<img width="1460" height="902" alt="evaluation-dashboard" src="https://github.com/user-attachments/assets/54ab9d39-7682-4cbe-834f-e0d79431b702" />

```

---
------------------

## Future Work

- Multi-Agent Scientific ML Copilot using LangGraph
- RAGAS-based evaluation
- Scientific paper summarization
- Automated literature review generation
- PDF report generation

---

## Author

Raj Birru


