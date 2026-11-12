# Enterprise Finance & Risk Intelligence Copilot

An AI-powered financial analysis and document intelligence platform that combines deterministic financial analytics, machine-learning-based anomaly detection, Retrieval-Augmented Generation (RAG), and multi-step AI orchestration.

The system helps finance teams explore transaction data, investigate unusual spending, understand budget variances, and ask questions about financial policies through a unified copilot.

## Key Features

### 1. Financial Intelligence
- Upload and process financial transaction CSV files.
- Validate transaction data before analysis.
- Calculate financial KPIs and spending summaries.
- Analyze budget variances.
- Break down spending by department and other available dimensions.

### 2. Financial Anomaly Detection
- Detect suspicious transactions using Isolation Forest.
- Support baseline and engineered feature configurations.
- Identify patterns such as unusual transaction amounts, weekend payments, and rare combinations.
- Rank suspicious transactions using anomaly scores.
- Evaluate detection performance against synthetic ground-truth anomalies.

### 3. Document Intelligence (RAG)
- Ingest financial policy PDF documents.
- Extract and split document text into page-aware chunks.
- Generate vector embeddings using Sentence Transformers.
- Store and search document embeddings using Qdrant.
- Retrieve relevant policy evidence to support financial questions.
- Generate responses using a locally hosted Ollama model.

### 4. LangGraph-Powered Copilot
The copilot coordinates specialized capabilities through a LangGraph workflow.

Depending on the question, it can route work to:
- KPI analysis.
- Budget variance analysis.
- Financial anomaly detection.
- Policy document retrieval.

The results are combined into a unified response. Financial calculations remain the responsibility of deterministic Python services, while the LLM is used for language generation and evidence-based explanations.

### 5. API and User Interface
- FastAPI REST API.
- API-key protection for protected endpoints.
- Interactive API documentation through Swagger UI.
- Web interface for interacting with the copilot.
- Docker Compose deployment.

---

## Architecture

```mermaid
flowchart TD
    U[Finance User] --> UI[Web UI]
    U --> API[FastAPI REST API]
    UI --> API

    API --> G[LangGraph Copilot]

    G --> K[KPI Analysis]
    G --> V[Variance Analysis]
    G --> A[Anomaly Detection]
    G --> R[RAG Retrieval]

    K --> PG[(PostgreSQL)]
    V --> PG
    A --> PG

    R --> E[Sentence Transformers]
    E --> Q[(Qdrant)]
    Q --> R

    K --> S[Synthesis]
    V --> S
    A --> S
    R --> S

    S --> L[Ollama LLM]
    L --> API
    API --> U
```

### Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python 3.11 |
| API framework | FastAPI |
| AI orchestration | LangGraph |
| LLM integration | Ollama |
| Local language model | Llama 3.2 1B |
| Embedding model | Sentence Transformers — all-MiniLM-L6-v2 |
| Vector database | Qdrant |
| Relational database | PostgreSQL 16 |
| Data processing | Pandas |
| Anomaly detection | Scikit-learn — Isolation Forest |
| PDF processing | Python PDF processing libraries |
| Validation | Pydantic |
| Testing | Pytest |
| Containerization | Docker and Docker Compose |

---

## Project Structure

```text
Enterprise_finance_risk_copliot/
│
├── data/
│   ├── budget_variance_policy.pdf
│   ├── sample_transactions.csv
│   ├── synthetic_ground_truth.csv
│   └── synthetic_transactions.csv
│
├── scripts/
│   ├── evaluate_anomalies.py
│   ├── evaluate_answerability.py
│   ├── evaluate_retrieval.py
│   ├── generate_policy_pdf.py
│   └── generate_synthetic_data.py
│
├── src/
│   └── app/
│       ├── api/
│       │   ├── copilot.py
│       │   ├── documents.py
│       │   └── financial.py
│       │
│       ├── core/
│       │   └── security.py
│       │
│       ├── models/
│       │   └── database_models.py
│       │
│       ├── schemas/
│       │
│       ├── services/
│       │   ├── anomaly_service.py
│       │   ├── answerability.py
│       │   ├── copilot_graph.py
│       │   ├── csv_processor.py
│       │   ├── document_ingestion_service.py
│       │   ├── document_processor.py
│       │   ├── embedding_service.py
│       │   ├── ingestion_service.py
│       │   ├── kpi_service.py
│       │   ├── llm_service.py
│       │   ├── rag_service.py
│       │   ├── retrieval_service.py
│       │   ├── variance_service.py
│       │   └── vector_store.py
│       │
│       ├── static/
│       │   └── index.html
│       │
│       ├── config.py
│       ├── database.py
│       └── main.py
│
├── tests/
│   ├── test_anomaly_features.py
│   ├── test_answerability.py
│   ├── test_chunking.py
│   ├── test_copilot_logic.py
│   ├── test_csv_processor_rules.py
│   └── test_graph_wiring.py
│
├── .dockerignore
├── .env.example
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── pytest.ini
├── requirements.txt
└── requirements-dev.txt
```

---

## Getting Started

### Prerequisites

Install the following:

- Python 3.11
- Docker Desktop
- Git
- Ollama

The application uses a locally hosted language model, so a paid external LLM API is not required.

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Enterprise_finance_risk_copliot
```

### 2. Create the Python environment

Using Conda:

```bash
conda create -n finance_copilot python=3.11 -y
conda activate finance_copilot
```

Install dependencies:

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 3. Configure environment variables

Create a `.env` file in the project root using `.env.example` as a template.

Configure the following variables:

```dotenv
APP_NAME=Enterprise Finance & Risk Intelligence Copilot
ENVIRONMENT=development

API_KEY=replace_with_a_strong_random_secret

POSTGRES_USER=copilot
POSTGRES_PASSWORD=replace_with_your_database_password
POSTGRES_DB=copilot

# Python scripts running directly on Windows
DATABASE_URL=postgresql+psycopg://copilot:YOUR_PASSWORD@localhost:5432/copilot

# FastAPI running inside Docker
DATABASE_URL_DOCKER=postgresql+psycopg://copilot:YOUR_PASSWORD@postgres:5432/copilot

QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=financial_documents

OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:1b

EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
MODEL_CACHE_DIR=./models
```

Replace the password placeholders with the same database password. If your password contains URL-reserved characters, URL-encode it in the connection URLs.

**Important:** Never commit `.env` to GitHub. It contains secrets and is excluded through `.gitignore`.

### 4. Download the local LLM

Install and start Ollama, then pull the model:

```bash
ollama pull llama3.2:1b
```

Verify the model:

```bash
ollama list
```

Ollama runs on the host machine, not inside the Docker Compose application containers.

### 5. Start the infrastructure

Start PostgreSQL, Qdrant, and the API:

```bash
docker compose up --build -d
```

Check the services:

```bash
docker compose ps
```

View application logs:

```bash
docker compose logs -f api
```

The API container connects to:
- PostgreSQL using the Docker Compose service hostname `postgres`.
- Qdrant using `http://qdrant:6333`.
- Ollama through `http://host.docker.internal:11434`.

### 6. Access the application

| Resource | URL |
|---|---|
| Web interface | http://localhost:8000/ui/ |
| API documentation | http://localhost:8000/docs |
| Health check | http://localhost:8000/health |
| Qdrant dashboard/API | http://localhost:6333 |

---

## Running Tests

Run the automated test suite from the project root:

```bash
python -m pytest -v
```

The current test suite contains 37 passing tests covering:

- Anomaly feature engineering.
- Answerability logic.
- Document chunking.
- Copilot orchestration logic.
- CSV processing rules.
- LangGraph wiring.

**Latest verified result:**

```text
37 passed
```

---

## Evaluation

The project includes dedicated evaluation scripts for anomaly detection, document retrieval, and answerability.

### 1. Anomaly Detection Evaluation

The anomaly evaluation uses synthetic financial transactions with known injected anomalies.

Run:

```bash
python scripts/evaluate_anomalies.py --help
```

Then run the evaluation using the ID of the uploaded synthetic dataset:

```bash
python scripts/evaluate_anomalies.py --dataset-id YOUR_DATASET_ID
```

The script compares baseline features against engineered features across multiple contamination settings.

Evaluation metrics include:

- Precision.
- Recall.
- F1-score.
- Precision@12.
- Ranking of injected anomalies.

#### Latest verified anomaly evaluation

| Metric | Baseline | Engineered features |
|---|---:|---:|
| Precision@12 | 58.3% | **83.3%** |
| Best precision | 100% | **100%** |
| Best recall | 83.3% | **100%** |
| Best F1-score | 0.67 | **1.00** |

The engineered configuration with automatic contamination achieved:

| Metric | Result |
|---|---:|
| Transactions flagged | 12 |
| Injected anomalies detected | 12 |
| Precision | 100% |
| Recall | 100% |
| F1-score | 1.00 |

These results demonstrate the benefit of engineered financial features on the controlled synthetic dataset. They should not be interpreted as a guarantee of equivalent performance on real-world financial transactions.

### 2. Retrieval Evaluation

Run:

```bash
python scripts/evaluate_retrieval.py --help
```

The retrieval evaluation measures how effectively the system retrieves relevant policy information.

### 3. Answerability Evaluation

Run:

```bash
python scripts/evaluate_answerability.py --help
```

The answerability evaluation checks whether retrieved evidence sufficiently supports answering a question.

The current answerability mechanism uses a keyword-coverage heuristic. It is not a semantic entailment model, and its results should be interpreted accordingly.

---

## Security and Responsible AI

The application incorporates several safeguards:

- API-key authentication for protected endpoints.
- Environment-based configuration for credentials.
- Separation of structured financial calculations from LLM-generated explanations.
- Evidence checks to reduce unsupported policy claims.
- Human review remains important for financial decisions.
- Synthetic data is used for controlled anomaly evaluation.

The copilot is an analytical support tool. It does not replace financial professionals, auditors, or authorized approval processes.

---

## Current Limitations and Future Improvements

Potential next improvements include:

1. **Retrieval optimization:** introduce hybrid search, reranking, and more comprehensive retrieval benchmarks.
2. **Answer quality evaluation:** add semantic relevance, faithfulness, and groundedness evaluation.
3. **Anomaly detection:** evaluate on larger and more representative real-world datasets, with human-reviewed labels.
4. **Observability:** add structured logging, tracing, and performance monitoring.
5. **Security:** introduce more granular authorization, rate limiting, and audit logging.
6. **Deployment:** add CI/CD automation and production deployment configuration.
7. **Scalability:** optimize ingestion, model loading, and database access for larger workloads.

---

## Author

**[Mohamed Bahaa]**

AI Engineer | Machine Learning | LLM Applications | RAG | AI Engineering

Built as an end-to-end AI engineering portfolio project demonstrating financial analytics, anomaly detection, document intelligence, LLM orchestration, API development, testing, and containerized deployment.

