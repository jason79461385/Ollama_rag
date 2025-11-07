# ingest.py
import os
from pathlib import Path
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

DOCS_DIR = r"./docs"
DB_DIR = os.environ.get("DB_PATH", r"./chroma.db")
EMBED_BASE_URL = os.environ.get("OLLAMA_HOST","http://127.0.0.1:11434")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "mxbai-embed-large") 

def build_docs():
    loaders = [
        DirectoryLoader(DOCS_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader),
        DirectoryLoader(DOCS_DIR, glob="**/*.txt", loader_cls=TextLoader),
        DirectoryLoader(DOCS_DIR, glob="**/*.md", loader_cls=TextLoader),
    ]
    docs = []
    for loader in loaders:
        try:
            docs.extend(loader.load())
        except Exception as e:
            print(f"[loader] {e}")
    return docs

if __name__ == "__main__":
    Path(DB_DIR).mkdir(parents=True, exist_ok=True)
    docs = build_docs()
    print(f"Loaded docs: {len(docs)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200,
        separators=["\n\n", "\n", "。", "，", " ", ""],
    )
    splits = splitter.split_documents(docs)
    print(f"Split into chunks: {len(splits)}")

    emb = OllamaEmbeddings(model=EMBED_MODEL, base_url=EMBED_BASE_URL)
    _ = emb.embed_query("ping")
    print("Embedding warmup OK")

    vs = Chroma.from_documents(
        documents=splits,
        embedding=emb,
        persist_directory=DB_DIR,
        collection_name="local_docs",
    )
    vs.persist()
    print("Chroma DB built at", DB_DIR)
