# StreamQuant 📈

**Distributed Financial Intelligence & ML Inference Pipeline**

StreamQuant is a production-grade, real-time financial intelligence platform. It ingests market data, runs multiple machine learning models concurrently to detect anomalies, classify trends, and analyze sentiment, and serves actionable signals through an asynchronous API.

## 🚀 The Problem & Solution

Retail investors and trading desks are overwhelmed by the velocity of financial market data. Existing tools either dump raw data with no interpretation or produce black-box signals with no reasoning. 

Furthermore, deploying deep learning models for financial analysis presents a fundamental systems challenge: ML inference is computationally expensive. Naive synchronous deployment behind a REST API causes server blocking and request timeouts under concurrent load.

**StreamQuant** solves this by decoupling high-speed data ingestion from slow ML inference using a distributed, asynchronous architecture. 

## 🏗️ System Architecture

```text
[Live Market Data / News]
          │
          ▼
 ┌─────────────────────┐
 │ C++ Data Ingestion  │ (High-speed parsing & filtering)
 └────────┬────────────┘
          │
          ▼
 ┌─────────────────────┐
 │    Redis Streams    │ (Message Broker / Queue)
 └────────┬────────────┘
          │
          ▼
 ┌─────────────────────┐
 │   Celery Workers    │ (Async Processing Pool)
 │                     │
 │  ┌───────────────┐  │
 │  │ ML Inference  │  │
 │  │ - XGBoost     │  │ 
 │  │ - FinBERT     │  │
 │  │ - Z-Score     │  │
 │  └──────┬────────┘  │
 └─────────┼───────────┘
           │
           ▼
 ┌─────────────────────┐      ┌─────────────────────┐
 │  FastAPI Backend    │ ◄─── │ Streamlit Dashboard │
 │  (RESTful API)      │      │ (Live UI & Alerts)  │
 └─────────┬───────────┘      └─────────────────────┘
           │
           ▼
 ┌─────────────────────┐
 │ PostgreSQL Database │ (Audit Log & Historical Data)
 └─────────────────────┘
```

## 🛠️ Getting Started (Running Locally)

Follow these steps to run the complete distributed pipeline on your local machine.

### 1. Prerequisites
- [Docker & Docker Desktop](https://docs.docker.com/desktop/) (Required for background services)
- Python 3.9+ 

### 2. Setup Environment

Clone the repository and install the required dependencies:

```bash
git clone <your-repo-url>
cd StreamQuant

# Create a virtual environment (optional but recommended)
python -m venv venv
# On Windows: venv\Scripts\activate
# On Mac/Linux: source venv/bin/activate

# Install all required Python packages
pip install -r requirements.txt
```

*(Note: The `requirements.txt` file contains dependencies for the FastAPI backend, Streamlit frontend, and ML tasks).*

### 3. Configure Database
The project can connect to either a local Docker PostgreSQL database or a remote Supabase instance.
To securely configure your database connection, create a `.env` file in the root directory:

```env
# Example .env file
DATABASE_URL=postgresql://postgres:password123@localhost:5432/finsignal
```

### 4. Start Background Services (Docker)
The architecture relies on Redis as a message broker and Celery for asynchronous ML processing.
Start these services (along with a local PostgreSQL DB) using Docker Compose:

```bash
docker-compose up --build -d
```
*(The Celery worker container will automatically install its own dependencies and start listening for tasks).*

### 5. Run the FastAPI Backend
Open a new terminal, activate your virtual environment, and start the API server. We recommend using `python -m` to bypass Windows PATH issues:

```bash
python -m uvicorn backend.main:app --reload
```
The API will be available at `http://127.0.0.1:8000`.

### 6. Run the Streamlit Dashboard
Open another terminal, activate your virtual environment, and start the frontend:

```bash
python -m streamlit run frontend/app.py
```
The terminal will launch in your browser at `http://localhost:8501`. From there, you can search for a ticker (e.g., TSLA) to trigger the ML pipeline!
