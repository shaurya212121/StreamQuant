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
