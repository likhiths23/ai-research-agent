# tools/pdf_tool.py
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv

load_dotenv()

def pdf_search_tool(query: str, k=5):
    emb = HuggingFaceEmbeddings(model_name=os.getenv("EMBEDDING_MODEL"))
    db = FAISS.load_local(
        os.getenv("FAISS_INDEX_DIR"), 
        emb,
        allow_dangerous_deserialization=True  # Added this parameter
    )
    docs = db.similarity_search(query, k=k)
    combined = "\n\n".join([d.page_content for d in docs])
    sources = "\n".join([f"Source: {d.metadata.get('source', 'Unknown')}" for d in docs])
    return f"Results from PDFs:\n{combined}\n\n{sources}"