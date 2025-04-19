# 🤖 AI IT Assistant — Hybrid RAG + Multi-Agent System with CrewAI

This project is a powerful AI-powered assistant designed to help IT teams answer technical queries using uploaded documentation (SOPs, Runbooks, Playbooks) — with fallback to Wikipedia for unknowns. It uses hybrid retrieval-augmented generation (RAG) and multi-agent collaboration with [CrewAI](https://github.com/joaomdmoura/crewai) for robust, context-aware answers.


*Main interface with PDF upload and query section*

---

## 🚀 Features

- 🔐 **Secure API Access**: Authenticate with your GROQ API key.
- 📁 **Multi-PDF Support**: Upload several documents in one go.
- 🧠 **Multi-Agent Crew**: 5 agents simulate a real IT team workflow.
- 🔎 **Hybrid Retrieval (Semantic + Keyword)** using FAISS + Hugging Face embeddings.
- 🌐 **Fallback to Wikipedia** if internal content lacks the answer.
- ✅ **Technical Answers + Suggestions + Test Steps** – all in one flow!

---

## 🎯 Objectives

- Automate IT question answering using internal documentation.
- Mimic IT team roles with LLM agents for ingest, retrieval, answering, suggestion, and validation.
- Seamlessly fetch fallback knowledge from Wikipedia when needed.
- Keep it flexible, secure, and user-friendly through Streamlit UI.

---

## 🧰 Tech Stack

| Tech                  | Description |
|-----------------------|-------------|
| **Streamlit**         | Web UI for uploading PDFs and asking queries |
| **CrewAI**            | Multi-agent LLM orchestration |
| **LangChain**         | LLM tooling, memory, and document parsing |
| **GEMMA2-9B-IT (via Groq)** | Main LLM for the agents |
| **Hugging Face Embeddings** | `all-MiniLM-L6-v2` for semantic vector search |
| **FAISS**             | Fast hybrid similarity search (keyword + vector) |
| **Wikipedia**         | Used for fallback answers via custom CrewAI tool |

---

## 🔄 Flow of the Application

```txt
1. User logs in with GROQ API Key via Streamlit.
2. Uploads multiple technical PDFs (runbooks/SOPs).
3. Enters a technical question.
4. CrewAI activates 5 agents:
   ├─ 📄 DocIngestorAgent: Loads and chunks PDFs.
   ├─ 🔍 RetrieverAgent: Retrieves top docs using hybrid search.
   ├─ 💬 ResponderAgent: Answers the query from retrieved chunks.
   ├─ 💡 SuggesterAgent: Suggests next steps or automation.
   └─ 🧪 TesterAgent: Recommends validation/tests/scripts.
5. If no relevant info is found:
   └─ 🌐 WikipediaAgent is triggered to get external info.
6. ✅ Final answer is displayed on Streamlit UI.
```

## 🧠 Agents Overview
Agent | Role
DocIngestorAgent | Parses and chunks the uploaded PDFs
RetrieverAgent | Uses semantic + keyword hybrid search to find relevant docs
ResponderAgent | Generates the answer to the user’s query
SuggesterAgent | Suggests improvements, automation ideas
TesterAgent | Recommends test cases or validation steps
WikipediaAgent | Fallback agent that fetches info from Wikipedia when needed

## 📦 Setup Instructions
```bash
# Clone the repo
git clone https://github.com/your-username/hybrid-rag-crewai-assistant.git
cd hybrid-rag-crewai-assistant

# (Optional) Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .\.venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## ✅ Sample Environment Variables
```bash
# .env
GROQ_API_KEY=your_groq_key_here
HF_TOKEN=your_huggingface_token
```

## 🖼️ Screenshots
### 🔐 GROQ API Key Authentication

### 📁 Uploading PDFs
Upload multiple technical documents.

### 🧠 Agent Response Display
Query results with suggestions and validations.

### 🖥️ Terminal Logs: Agents in Action
Behind the scenes — each agent logs its steps in terminal.

