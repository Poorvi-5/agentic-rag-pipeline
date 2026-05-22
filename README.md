# Agentic RAG Pipeline 🤖

An advanced Retrieval-Augmented Generation system where an AI agent
autonomously selects the best retrieval strategy for each query.

## Architecture
User Query → ReAct Agent → Tool Selection → Retrieval → Reflection → Answer

## Tools Available

| Tool | Purpose |
|---|---|
| Vector Search | Semantic search in local document knowledge base |
| SQL Query | Natural language to SQL on structured database |
| Web Search | Real-time DuckDuckGo search |
| External API | Live weather data via Open-Meteo |

## Tech Stack

- **LLM**: Groq (Llama 3.3 70B) — Free
- **Embeddings**: sentence-transformers (local) — Free
- **Vector DB**: FAISS + ChromaDB — Free
- **Framework**: LangChain — Open source
- **UI**: Streamlit — Open source

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/agentic-rag-pipeline.git
cd agentic-rag-pipeline
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
pip install -e .
```

Add your Groq API key to env`:

GROQ_API_KEY=your_groq_api_key_here

## Run

```bash
# Step 1: Ingest documents
python ingest.py

# Step 2: Launch UI
streamlit run app/streamlit_app.py
```

## Project Structure
agentic-rag-pipeline/
├── data/raw/          ← Drop your PDFs/TXT files here
├── src/
│   ├── ingestion/     ← Document loading, chunking, embedding
│   ├── tools/         ← 4 retrieval tools
│   ├── agent/         ← ReAct agent + prompts
│   └── reflection/    ← Evaluator + self-correction
├── app/               ← Streamlit chat UI
└── tests/             ← All test files

## Steps Built

- ✅ Step 1: Environment setup & project structure  
- ✅ Step 2: Document ingestion & embedding pipeline  
- ✅ Step 3: Multi-tool retrieval system  
- ✅ Step 4: ReAct agent with intent analysis  
- ✅ Step 5: Reflection & self-correction layer  
- ✅ Step 6: Streamlit chat UI & full integration
