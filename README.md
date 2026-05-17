# SQL + Advanced RAG Dual-Engine PM Data Assistant

An intelligent, production-ready data assistant designed for Product Managers (PMs). This system bridges the gap between colloquial business language and complex technical data by combining **Text-to-SQL** (for structured data analysis) and **Advanced RAG** (for unstructured market insights) via an **Asynchronous Parallel Agent Architecture**.

---

## 🗺️ System Architecture

```text
       ┌────────────────────────────────┐
       │     PM Question (Natural)      │
       └───────────────┬────────────────┘
                       │
                       ▼
       ┌────────────────────────────────┐
       │    Business Glossary Layer     │ <--- Translates business terms to logic
       └───────────────┬────────────────┘
                       │
                       ▼
       ┌────────────────────────────────┐
       │   Intelligent Router Agent     │ <--- OpenAI Tools / Function Calling
       └───────┬────────────────┬───────┘
               │ (Async)        │ (Async)
               ▼                ▼
       ┌───────────────┐┌───────────────┐
       │  Text-to-SQL  ││ Advanced RAG  │
       │  (Company DB) ││(Market Repts) │
       └───────┬───────┘└───────┬───────┘
               │                │
               └───────┬────────┘
                       ▼
       ┌────────────────────────────────┐
       │     Merge & Insights Layer     │ <--- Dual-source comparative analysis
       └───────────────┬────────────────┘
                       │
                       ▼
       ┌────────────────────────────────┐
       │  Structured PM Dashboard (UI)  │ <--- JSON Rendered Tables & Charts
       └────────────────────────────────┘
```

---

## 📂 Repository Structure

The project repository is structured logically to separate data assets, agent cores, evaluation pipelines, and frontend components:

```text
pm-data-assistant/
│
├── .github/workflows/       # CI/CD pipelines (Automation)
├── config/                  # Configuration files
│   ├── settings.yaml        # LLM parameters, database configurations, etc.
│   └── glossary.json        # PM Business Glossary mapping (calculations & logic)
│
├── data/                    # Data Storage Layer
│   ├── database/            # Structured Data (SQLite/PostgreSQL schemas & mocks)
│   │   └── mock_pm_data.db
│   ├── documents/           # Unstructured Data (Industry PDFs, Competitor Benchmarks)
│   │   ├── market_share_2025.pdf
│   │   └── industry_churn_benchmarks.pdf
│   └── vectorstore/         # Vector Database index files (Chroma / FAISS)
│
├── src/                     # Core Source Code
│   ├── __init__.py
│   ├── router.py            # Dynamic Routing Agent (Tool Calling logic)
│   ├── text_to_sql.py       # SQL Agent with Few-Shot prompts & execution sandbox
│   ├── advanced_rag.py      # ParentDocumentRetriever & Hybrid Search pipeline
│   ├── merge_layer.py       # Asynchronous gathering (asyncio) & comparative insights
│   └── utils.py             # Helper modules (parsers, token tools, sanitization)
│
├── app/                     # Frontend Application
│   └── main.py              # Streamlit Dashboard & layout configurations
│
├── evaluation/              # Testing & Quality Assurance
│   ├── test_dataset.json    # 20+ Ground Truth question-answer pairs
│   └── eval_ragas.py        # RAG Triad assessment using Ragas Framework
│
├── notebooks/               # R&D and Sandboxing
│   ├── 01_rag_experiment.ipynb
│   └── 02_sql_sandbox.ipynb
│
├── .gitignore               # Excluded files (vritual environments, keys, DBs)
├── README.md                # Comprehensive project manual (This file)
└── requirements.txt         # Package dependencies
```

---

## 🛠️ Key Technical Features

1. **Business Glossary Translation:** Integrates business definitions directly into the prompt stream, translating words like `"last month's churn rate"` into structured mathematical logic before querying metadata.
2. **Asynchronous Query Gathering:** Employs Python's `asyncio` to query database records and extract vector documents concurrently, minimizing user-facing latency to $\max(	ext{SQL}, 	ext{RAG})$ instead of their sum.
3. **Structured Response Enforcer:** Uses strict JSON schema bindings (`Pydantic` validation) to systematically convert raw generative LLM texts into fully interactive Streamlit components (`st.dataframe` and `st.plotly_chart`).
4. **Production Security Guard:** Mitigates SQL Injection risks through query inspection via pre-execution regex checks alongside read-only database connections.
5. **RAG Triad Evaluation:** Automatically monitors Context Relevance, Groundedness, and Answer Relevance utilizing LLM-as-a-judge patterns.

---

## 🚀 Getting Started

### 1. Clone the Repository & Environment Setup
```bash
git clone https://github.com/yourusername/pm-data-assistant.git
cd pm-data-assistant

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install required packages
pip install -r requirements.txt
```

### 2. Environment Variables Configuration
Create a `.env` file in the root directory:
```env
OPENAI_API_KEY=your_openai_api_key_here
DATABASE_URL=sqlite:///data/database/mock_pm_data.db
VECTORSTORE_DIR=data/vectorstore
```

### 3. Initialize the Database and Embeddings
```bash
# Run parsing and chunking script to populate the vector database
python src/advanced_rag.py --initialize

# Seed mock database values 
python src/text_to_sql.py --seed
```

### 4. Launch the Web Application
```bash
streamlit run app/main.py
```

---

## 📊 Evaluation Results
Our pipeline is evaluated against 20 production baseline questions across the RAG Triad framework:
* **Context Precision:** 92%
* **Faithfulness (Groundedness):** 95% (No hallucinations detected in core metrics)
* **Answer Relevance:** 89%

---

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.
