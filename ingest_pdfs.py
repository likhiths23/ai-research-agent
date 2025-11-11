# ingest_pdfs.py
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

def ingest(pdfs):
    index_dir = os.getenv("FAISS_INDEX_DIR")
    os.makedirs(index_dir, exist_ok=True)
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    embeddings = HuggingFaceEmbeddings(model_name=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
    
    all_chunks = []
    for pdf in pdfs:
        print(f"Loading {pdf} ...")
        loader = PyPDFLoader(pdf)
        docs = loader.load()
        chunks = splitter.split_documents(docs)
        for c in chunks:
            c.metadata["source"] = pdf
        all_chunks.extend(chunks)
    
    print(f"Creating FAISS index with {len(all_chunks)} chunks...")
    db = FAISS.from_documents(all_chunks, embeddings)
    db.save_local(index_dir)
    print(f"✅ Ingestion complete! Index saved to {index_dir}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ingest_pdfs.py <pdf1> <pdf2> ...")
        sys.exit(1)
    ingest(sys.argv[1:])