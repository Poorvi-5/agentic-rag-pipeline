# verify_setup.py

import sys

results = []

def check(label, fn):
    try:
        fn()
        results.append((True, label))
        print(f"  PASS  {label}")
    except Exception as e:
        results.append((False, label))
        print(f"  FAIL  {label}  →  {e}")

print("\n Agentic RAG — Setup Verification")
print("─" * 45)

check("Python 3.10+",
    lambda: __import__('sys') and sys.version_info >= (3,10))

check("LangChain",
    lambda: __import__('langchain'))

check("LangChain-Groq",
    lambda: __import__('langchain_groq'))

check("FAISS",
    lambda: __import__('faiss'))

check("ChromaDB",
    lambda: __import__('chromadb'))

check("sentence-transformers",
    lambda: __import__('sentence_transformers'))

check("Streamlit",
    lambda: __import__('streamlit'))

check("PyPDF",
    lambda: __import__('pypdf'))

check("python-dotenv",
    lambda: __import__('dotenv'))

check("DuckDuckGo search",
    lambda: __import__('duckduckgo_search'))

check("Config + .env loads", lambda: (
    __import__('src.config', fromlist=['validate_config'])
    .validate_config()
))

check("Groq API reachable", lambda: (
    __import__('langchain_groq', fromlist=['ChatGroq'])
    .__dict__['ChatGroq'](
        api_key=__import__('os').getenv('GROQ_API_KEY'),
        model="llama-3.1-70b-versatile"
    ).invoke("Say OK in one word only")
))

check("FAISS index works", lambda: (
    __import__('faiss').__dict__['IndexFlatL2'](128)
))

check("ChromaDB works", lambda: (
    __import__('chromadb').Client()
    .get_or_create_collection("smoke_test")
))

check("Embedding model loads", lambda: (
    __import__('sentence_transformers',
               fromlist=['SentenceTransformer'])
    .__dict__['SentenceTransformer']("all-MiniLM-L6-v2")
    .encode("hello world")
))

check("Folder structure OK", lambda: (
    __import__('src.config', fromlist=['RAW_DATA_DIR'])
    .__dict__['RAW_DATA_DIR'].exists()
    or (_ for _ in ()).throw(Exception("data/raw missing"))
))

print("─" * 45)
passed = sum(1 for ok, _ in results if ok)
print(f"\n{passed}/{len(results)} checks passed.")
if passed == len(results):
    print("Everything is GREEN. Ready for Step 2!\n")
else:
    print("Fix the FAIL lines before proceeding.\n")