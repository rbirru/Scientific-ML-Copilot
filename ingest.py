import os
import shutil

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

DATA_PATH = "data"
DB_PATH = "chroma_db"

def main():
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)

    loader = PyPDFDirectoryLoader(DATA_PATH)
    docs = loader.load()

    print(f"Loaded {len(docs)} pages")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(docs)

    print(f"Created {len(chunks)} chunks")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )

    print(f"Indexed {len(chunks)} chunks into ChromaDB.")

if __name__ == "__main__":
    main()