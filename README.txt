rmdir /s /q chroma_db

mkdir chroma_db

python ingest.py

python -m streamlit run app.py

python evaluate.py

python -m streamlit run evaluation_dashboard.py



How does SINDy help in low-data regimes?
Compare SINDy and PINNs.
What is the role of MPC in nonlinear control?

