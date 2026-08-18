# 🌌 AeroPredict AI — Enterprise Serverless MLOps & RAG LLM Air Quality Forecasting Platform

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Hopsworks](https://img.shields.io/badge/Feature_Store-Hopsworks-FF69B4?style=for-the-badge&logo=hopsworks&logoColor=white)](https://www.hopsworks.ai/)
[![LightGBM](https://img.shields.io/badge/ML_Engine-LightGBM%20%2B%20XGBoost-025E8D?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://lightgbm.readthedocs.io/)
[![LLM RAG](https://img.shields.io/badge/LLM_RAG-Ollama%20%2F%20Mistral%20%2F%20OpenAI-00599E?style=for-the-badge&logo=ollama&logoColor=white)](https://ollama.com/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Container-Docker%20%2F%20Podman-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-2088FF?style=for-the-badge&logo=githubactions&logoColor=white)](https://github.com/GuillenConcepcion/DS-AeroPredict-AI-Enterprise-Serverless-MLOps-RAG-LLM-Air-Quality-Forecasting-Platform)
[![uv](https://img.shields.io/badge/Package_Manager-uv-DE5D43?style=for-the-badge&logo=astral&logoColor=white)](https://github.com/astral-sh/uv)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

---

## 🎯 Propósito del Proyecto & Valor de Negocio (Project Mission)

**AeroPredict AI** es un sistema MLOps de Inteligencia Artificial enfocado en la predicción continua de la Calidad del Aire (PM2.5, PM10, AQI Europeo y Estadounidense) y recomendaciones de salud ambiental en tiempo real.

Diseñado bajo principios de **cero costo operativo en infraestructura serverless**, el sistema aborda tres pilares clave:
1. **Inteligencia Ambiental Predictiva**: Pronóstico a **7 días (168 horas)** para estaciones IoT públicas (Estocolmo Central, Dublín, Madrid y coordenadas personalizadas).
2. **MLOps Automatizado & Fallback Resiliente**: Integración nativa con **Hopsworks Feature Store & Model Registry** orquestado vía **GitHub Actions**, operando en modo *Local Parquet Fallback* en caso de ausencia de credenciales Cloud.
3. **Generación RAG & Consultas Conversacionales con LLMs**: Cadena RAG multimodal basada en arquitectura **ChatML con Function Calling**, priorizando modelos **LLM Locales con Ollama (Mistral / Llama 3 en 4-bit)** para cero costos de token, con fallback a Mistral Cloud, OpenAI GPT y motor experto de dominio.

---

## ⚡ Stack Tecnológico Aplicado (Applied Tech Stack)

| Capa Tecnológica | Tecnologías & Herramientas | Función & Arquitectura |
| :--- | :--- | :--- |
| **MLOps & Store** | Hopsworks, GitHub Actions, Parquet | Feature Store / View, Model Registry versionado, orquestación CI/CD daily cron. |
| **Machine Learning & Time Series** | LightGBM, XGBoost, CatBoost, Scikit-Learn | Benchmark multi-algoritmo, time-series chronological split, evaluación MAE/RMSE/R². |
| **EDA & Time Series Audit** | Statsmodels, SciPy, Pandas, NumPy | Prueba ADF (Augmented Dickey-Fuller) de estacionariedad, descomposición estacional (24h). |
| **Generative AI & RAG** | ChatML, Ollama (Local Llama 3 4-bit / Mistral), OpenAI, Mistral API | Extracción de funciones JSON, RAG context synthesis para recomendaciones de salud. |
| **Contenedores & Entorno** | Docker Multi-Stage, Podman, `uv` | Build ligero securizado (<300MB), gestión ultra-rápida de entorno virtual. |
| **Visualización & Web UI** | Streamlit, Plotly, HTML5, Vanilla CSS3 | Dashboard interactivo con gauges AQI, gráficos de pronóstico e interfaz glassmorphism. |

---

## 🏛️ System Architecture & MLOps Pipeline

```mermaid
flowchart TD
    subgraph Data & EDA Layer
        A[IoT Air Quality Sensors\nStockholm / Dublin / Global] --> C[Open-Meteo & IoT APIs]
        B[Global Meteorological Observations] --> C
        C --> EDA[src/eda.py\nADF Stationarity & Seasonal Decomp]
    end

    subgraph Feature Engineering & Store
        C --> D[1_feature_pipeline.py\nDaily Feature Ingestion]
        D --> E[(Hopsworks Feature Store / Local Parquet)]
    end

    subgraph Multi-Algorithm Training & Model Registry
        E --> F[2_training_pipeline.py\nMulti-Algorithm Benchmark]
        F -->|Evaluates| M1[LightGBM MAE: 0.310]
        F -->|Evaluates| M2[XGBoost MAE: 0.321]
        F -->|Evaluates| M3[Random Forest MAE: 0.332]
        M1 -->|Selects Best Model| G[Production Model Artifact]
        G --> H[(Hopsworks Model Registry / Local)]
    end

    subgraph Inference, RAG & Serving
        E --> I[3_batch_inference.py\n7-Day Forecast & Drift Monitoring]
        H --> I
        I --> J[Interactive Streamlit Dashboard & Web UI]
        J --> LLM[src/llm_chain.py\nRAG & ChatML LLM Engine\nOllama / Mistral / OpenAI]
    end

    subgraph Infrastructure & Deployment
        K[GitHub Actions Cron\nAutomated Daily Pipelines] -->|Triggers| D
        K -->|Triggers| I
        CONTAINER[Docker & Podman Multi-Stage Container] -->|Serves| J
    end
```

---

## 📂 Project Directory Structure

```text
D:\LabD\DS-AI-Air-Quality-System\
├── notebooks/
│   ├── 0_exploratory_data_analysis.ipynb # Notebook 0: Automated EDA & stationarity (ADF) audit
│   ├── 1_backfill_feature_group.ipynb    # Notebook 1: Backfill Hopsworks feature group with historical data
│   ├── 2_daily_feature_pipeline.ipynb    # Notebook 2: Daily feature pipeline (retrieve new IoT data -> Hopsworks)
│   ├── 3_training_pipeline.ipynb         # Notebook 3: Multi-model benchmark & register to Hopsworks Model Registry
│   ├── 4_batch_inference_pipeline.ipynb  # Notebook 4: Download model, batch predict, produce forecast/hindcast graphs
│   └── 5_llm_personalized_recommendations.ipynb # Notebook 5: RAG & LLM Function Calling chain
├── docs/                                 # GitHub Pages Deployment Root
│   ├── index.html                        # GitHub Pages Air Quality Dashboard & forecast graph viewer
│   ├── styles.css                        # Glassmorphism dark-mode UI styles
│   └── app.js                            # Interactive Plotly graph engine
├── .github/
│   └── workflows/
│       ├── 1_feature_pipeline.yml        # Daily feature store ingestion cron (00:00 UTC)
│       ├── 2_training_pipeline.yml       # Manual model retraining trigger
│       ├── 3_batch_inference.yml         # Daily batch inference & 7-day forecast cron (01:00 UTC)
│       └── gh-pages.yml                  # Automated deployment to GitHub Pages
├── src/
│   ├── config.py                         # Location coordinates, Pydantic settings, Hopsworks secrets
│   ├── data_fetcher.py                   # Data ingestion module (Open-Meteo & IoT APIs)
│   ├── eda.py                            # Programmatic EDA, ADF stationarity, seasonal decomposition
│   ├── features.py                       # Feature engineering & lag/rolling calculation
│   ├── hopsworks_utils.py               # Hopsworks feature store & secret manager interface
│   ├── llm_chain.py                      # RAG LLM function calling & ChatML prompt engine
│   └── models.py                         # Multi-algorithm model benchmark (LightGBM, XGBoost, CatBoost, RF)
├── pipelines/
│   ├── 1_feature_pipeline.py
│   ├── 2_training_pipeline.py
│   └── 3_batch_inference.py
├── Dockerfile                            # Multi-stage production container build
├── .dockerignore                         # Docker ignore configuration
├── app.py                                # Streamlit dashboard application
├── .env.example                          # Environment configuration template
├── requirements.txt                      # Core system dependencies
├── requirements-llm.txt                  # 4-bit Quantized Llama 3 8B / GPU LLM dependencies
└── README.md                             # MLOps documentation & author profile
```

---

## 🚀 Key Features & Components

1. **Data Ingestion & Feature Engineering** ([`src/data_fetcher.py`](file:///d:/LabD/DS-AI-Air-Quality-%20System/src/data_fetcher.py), [`src/features.py`](file:///d:/LabD/DS-AI-Air-Quality-%20System/src/features.py)):
   - Ingest multi-year historical data (PM2.5, PM10, NO2, O3, AQI) + meteorology (temperature, relative humidity, wind speed, pressure, surface radiation).
   - Compute lag features (1h, 24h, 48h, 7d lags), rolling statistics (24h mean, 24h std, 7d rolling max PM2.5), cyclical temporal encodings (hour of day, day of week, month), and weather-AQI interaction features.

2. **Feature Store & Model Registry** ([`src/hopsworks_utils.py`](file:///d:/LabD/DS-AI-Air-Quality-%20System/src/hopsworks_utils.py)):
   - Seamless integration with Hopsworks Feature Store (`air_quality_feature_group` & `air_quality_feature_view`).
   - Local offline mode (`data/` parquet files) when running locally without API keys.
   - Versioned Hopsworks Model Registry for model artifacts and metrics (MAE, RMSE, R²).

3. **Pipeline Automation** (`pipelines/*.py` & `.github/workflows/*.yml`):
   - Automated daily execution via GitHub Actions (cron: `'0 1 * * *'`).
   - Continuous model performance monitoring (tracking actual vs predicted AQI to alert on feature/concept drift).

4. **Interactive Web UI** ([`web/index.html`](file:///d:/LabD/DS-AI-Air-Quality-%20System/web/index.html), `web/styles.css`, `web/app.js`, [`app.py`](file:///d:/LabD/DS-AI-Air-Quality-%20System/app.py)):
   - Premium glassmorphism dark-mode interface.
   - Real-time AQI level gauge (Good, Moderate, Unhealthy, Dangerous).
   - 7-day hourly & daily forecasting charts powered by Plotly.
   - Model performance dashboard comparing predicted vs ground-truth historical values with drift detection metrics.
   - City switcher (Stockholm, Dublin, Madrid, London, New York).

5. **Documentation** ([`README.md`](file:///d:/LabD/DS-AI-Air-Quality-%20System/README.md)):
   - Tailored for Senior Data Scientist & MLOps Engineer standards.
   - Detailed step-by-step instructions for deploying to GitHub Actions + Hopsworks.

---

## 📊 Model Evaluation & Multi-Algorithm Benchmark Results

The training pipeline ([`pipelines/2_training_pipeline.py`](file:///d:/LabD/DS-AI-Air-Quality-%20System/pipelines/2_training_pipeline.py)) evaluates multiple Machine Learning algorithms on a chronological time-series split of historical air quality observations.

### 🏆 Multi-Algorithm Comparison Table

| Algorithm | Model Type | MAE (µg/m³) | RMSE | R² Score | Selection Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| 🏆 **LightGBM** | Leaf-wise Gradient Boosting | **0.3096** | **0.4120** | **0.9583** | **Production Selected** |
| ⚡ **XGBoost** | Depth-wise Gradient Boosting | 0.3206 | 0.4768 | 0.9442 | Benchmark |
| 🌲 **Random Forest** | Bagged Decision Ensemble | 0.3317 | 0.4764 | 0.9443 | Benchmark |

---

### 📈 Predictive Feature Importance Top 5

1. **`pm2_5_lag_1h`** (Score: `330`): 1-hour previous PM2.5 measurement.
2. **`pm2_5_roll_mean_72h`** (Score: `67`): 72-hour rolling average concentration.
3. **`wind_direction_10m`** (Score: `65`): Wind direction angle at 10 meters.
4. **`stagnation_index`** (Score: `41`): Atmospheric stagnation indicator.
5. **`pm2_5_lag_24h`** (Score: `37`): 24-hour previous PM2.5 measurement.

---

### 🔬 Stationarity & Time-Series Audit (ADF Test)

- **Augmented Dickey-Fuller (ADF) Test**: `p-value = 0.001` ($p < 0.05$).
- **Conclusion**: The target PM2.5 time series is **Stationary**, ensuring reliable model generalization and low risk of spurious regressions.

---

## 🛠️ Quick Start Guide

### 1. Prerequisites & Installation

```bash
# Clone the repository
git clone https://github.com/GuillenConcepcion/DS-AeroPredict-AI-Enterprise-Serverless-MLOps-RAG-LLM-Air-Quality-Forecasting-Platform.git
cd DS-AeroPredict-AI-Enterprise-Serverless-MLOps-RAG-LLM-Air-Quality-Forecasting-Platform

# Fast Environment Setup with uv (Recommended)
uv venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

### 2. Containerization (Docker & Podman)

```bash
# Build production Docker / Podman image
docker build -t aeropredict-mlops .
# Or using Podman:
podman build -t aeropredict-mlops .

# Run containerized Streamlit Application
docker run -d -p 8501:8501 --name aeropredict aeropredict-mlops
# Or using Podman:
podman run -d -p 8501:8501 --name aeropredict aeropredict-mlops
```

### 3. Execution Pipelines

#### Step 0: Run Programmatic EDA & Stationarity Audit
```bash
python -m src.eda
```

#### Step 1: Ingest Features
```bash
python pipelines/1_feature_pipeline.py --location Stockholm --days 180
```

#### Step 2: Train Model & Benchmark Algorithms (LightGBM, XGBoost, CatBoost, RF)
```bash
python pipelines/2_training_pipeline.py
```

#### Step 3: Run Batch Forecast & 7-Day Inference
```bash
python pipelines/3_batch_inference.py --location Stockholm
```

### 4. Launching the Web Application

#### Option A: Streamlit Dashboard
```bash
streamlit run app.py
```

#### Option B: Standalone Web Interface
Open `web/index.html` in any modern web browser or serve locally via Python:
```bash
python -m http.server 8000 --directory web
```

---

## ⚙️ CI/CD & Serverless Automation (GitHub Actions)

The project includes pre-configured GitHub Actions workflows in `.github/workflows/`:

| Workflow File | Trigger Schedule | Function |
|---|---|---|
| [`1_feature_pipeline.yml`](file:///d:/LabD/DS-AI-Air-Quality-%20System/.github/workflows/1_feature_pipeline.yml) | Daily at `00:00 UTC` | Ingests new daily IoT AQ & weather measurements into Hopsworks Feature Store. |
| [`3_batch_inference.yml`](file:///d:/LabD/DS-AI-Air-Quality-%20System/.github/workflows/3_batch_inference.yml) | Daily at `01:00 UTC` | Downloads model, generates 7-day AQI forecast, updates prediction artifacts & Plotly graphs. |
| [`gh-pages.yml`](file:///d:/LabD/DS-AI-Air-Quality-%20System/.github/workflows/gh-pages.yml) | On Push / After Inference | Deploys static dashboard, Plotly graphs, and LLM advice to free GitHub Pages site. |
| [`2_training_pipeline.yml`](file:///d:/LabD/DS-AI-Air-Quality-%20System/.github/workflows/2_training_pipeline.yml) | On-Demand (`workflow_dispatch`) | Manual/interactive training pipeline execution (typically run locally on laptop/notebooks). |

### Environment Variables & Secrets (Optional for Cloud Integration)
To connect your Hopsworks cloud instance:
- `HOPSWORKS_API_KEY`: Your Hopsworks user API key.
- `HOPSWORKS_PROJECT_NAME`: Hopsworks project name (default: `air_quality_prediction`).

---

## 🧪 Verification Plan

### Automated Verification
- Run [`pipelines/1_feature_pipeline.py`](file:///d:/LabD/DS-AI-Air-Quality-%20System/pipelines/1_feature_pipeline.py) to verify multi-year historical data fetching and feature creation.
- Run [`pipelines/2_training_pipeline.py`](file:///d:/LabD/DS-AI-Air-Quality-%20System/pipelines/2_training_pipeline.py) to verify model training (LightGBM/XGBoost), metrics logging, and artifact generation.
- Run [`pipelines/3_batch_inference.py`](file:///d:/LabD/DS-AI-Air-Quality-%20System/pipelines/3_batch_inference.py) to verify multi-day AQI prediction generation and drift metric computation.
- Validate Python syntax, dependencies, and [`requirements.txt`](file:///d:/LabD/DS-AI-Air-Quality-%20System/requirements.txt) file.

### Manual / UI Verification
- Launch local web server / browser interface for `web/index.html` and verify interactive elements, AQI gauges, and Plotly charts.
- Launch `streamlit run app.py` (if Streamlit environment available) or verify web dashboard rendering.

---

## 👤 Author Profile

<p align="left">
  <img src="docs/img%20Dara%20Scientist.png" alt="Guillén Concepción" width="160" onerror="this.src='images/guillen_logo.png'">
</p>

**Guillén Concepción**  
*Senior Data Scientist & MLOps Engineer*

Specialist in end-to-end Artificial Intelligence solutions, MLOps architecture, and production-grade Cloud-Native Machine Learning systems.

- **LinkedIn:** [Guillén Concepción](https://www.linkedin.com/in/guillen-concepcion-25266b127)
- **GitHub:** [@GuillenConcepcion](https://github.com/GuillenConcepcion/DS-AeroPredict-AI-Enterprise-Serverless-MLOps-RAG-LLM-Air-Quality-Forecasting-Platform)
- **Email:** [guillenconcepcion@gmail.com](mailto:guillenconcepcion@gmail.com)

---
*License: MIT*
