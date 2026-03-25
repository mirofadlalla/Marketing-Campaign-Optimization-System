# Marketing Campaign Optimization System

> An end-to-end machine learning solution for predicting conversion rates and optimizing marketing campaign performance across multiple channels and audience segments.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Core Insights](#core-insights)
- [Technology Stack](#technology-stack)
- [API Documentation](#api-documentation)
- [ML Pipeline](#ml-pipeline)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

The Marketing Campaign Optimization System is a comprehensive data science solution designed to:

- **Analyze** 200,000+ marketing campaigns across 6 channels and 5 audience segments
- **Predict** conversion rates using machine learning models
- **Optimize** marketing spend and campaign performance
- **Deploy** production-ready models via REST API
- **Track** model performance using MLFlow

This system helps marketing teams understand campaign effectiveness and identify optimization opportunities through data-driven insights.

---

## Key Features

### 🎯 Core Capabilities

- **Conversion Rate Prediction**: ML model predicting conversion probability for new campaigns
- **Multi-Channel Analysis**: Support for Facebook, Google Ads, YouTube, Instagram, Email, and Website
- **Audience Segmentation**: Analysis across All Ages, Men 18-24, Men 25-34, Women 25-34, Women 35-44
- **Comprehensive Analytics**: CTR, engagement scores, ROI analysis, acquisition cost optimization
- **Production Deployment**: FastAPI-based REST API for real-time predictions
- **Model Versioning**: MLFlow integration for experiment tracking and model management

### 📊 Analysis Capabilities

- Click-Through Rate (CTR) Analysis
- Conversion Rate Optimization
- Channel Performance Comparison
- Audience Segment Performance
- ROI Distribution Analysis
- Acquisition Cost Benchmarking
- Engagement Score Analysis

---

## Project Structure

```
Marketing_Campaign_Optimization_System/
├── README.md                           # This file
├── Insights_README.md                  # Detailed business insights
├── data/
│   └── marketing_campaign_dataset.xlsx # Raw dataset (200K+ records)
│
├── data_analysis.ipynb                 # Exploratory Data Analysis
├── data_cleaning.ipynb                 # Data preprocessing and cleaning
│
├── data_modeling/
│   ├── run_pipeline.py                 # Entry point for training pipeline
│   ├── deploy.ipynb                    # Model deployment notebook
│   │
│   ├── piplines/                       # Training pipelines (ZenML)
│   │   └── train_pipeline.py          # Complete training workflow
│   │
│   ├── steps/                          # Individual pipeline steps
│   │   ├── load_df.py                 # Data loading
│   │   ├── data_preprossing.py        # Data cleaning & preprocessing
│   │   ├── feature_engineering.py     # Feature scaling & encoding
│   │   ├── apply_fetaure_eng_on_data_preprossing.py  # Apply transformations
│   │   ├── training.py                # Model training
│   │   └── __init__.py
│   │
│   └── deployment/                     # Inference pipelines
│       ├── README.md                   # Deployment guide
│       ├── DATA_PREPROCESSING_GUIDE.md # Preprocessing documentation
│       │
│       ├── pipelines/
│       │   ├── training_deployment.py # Train & deploy pipeline
│       │   └── inference.py           # Prediction pipeline
│       │
│       ├── steps/
│       │   ├── inference_data_loader.py   # Load new data for inference
│       │   ├── prediction_service_loader.py # Load model service
│       │   └── predictor.py               # Make predictions
│       │
│       └── utils/
│           └── deployment_trigger.py     # Model promotion logic
│
├── api/
│   ├── main.py                         # FastAPI application
│   ├── get_production_model.py         # Load production model from MLFlow
│   └── rate_limitizer.py               # Rate limiting utility
│
└── power bi/                           # Power BI dashboards (optional)
```

---

## Dataset

### Marketing Campaign Dataset

**File**: `data/marketing_campaign_dataset.xlsx`

**Size**: 200,000+ marketing campaign records

**Key Fields**:
- **Campaign Metadata**: Campaign_Type, Company, Location
- **Targeting**: Target_Audience, Customer_Segment
- **Performance**: Duration, Channel_Used, Date
- **Metrics**: Impressions, Clicks, Engagement_Score, Customer_Segment
- **Financial**: Acquisition_Cost, ROI
- **Target Variable**: Conversion_Rate (predicted variable)

**Audience Segments**:
- All Ages
- Men 18-24
- Men 25-34
- Women 25-34
- Women 35-44

**Marketing Channels**:
- Facebook
- Google Ads
- YouTube
- Instagram
- Email
- Website

---

## Architecture

### System Architecture

```
┌─────────────────────┐
│  Raw Data (.xlsx)   │
└──────────┬──────────┘
           │
           ▼
┌──────────────────────┐
│ Data Processing      │  (data_analysis.ipynb, data_cleaning.ipynb)
│ • Load               │
│ • Clean              │
│ • EDA                │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ ZenML Training       │  (data_modeling/)
│ Pipeline             │
│ • Preprocessing      │
│ • Feature Eng        │
│ • Model Training     │
│ • Evaluation         │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ MLFlow Registry      │  Experiment tracking &
│ (Production Model)   │  Model versioning
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ FastAPI Server       │  (api/main.py)
│ Production Inference │
└──────────────────────┘
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- MLFlow (with local or remote tracking server)
- FastAPI
- ZenML
- Pandas, Scikit-Learn, NumPy

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Marketing_Campaign_Optimization_System
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install fastapi uvicorn mlflow zenml pandas scikit-learn numpy pydantic
   ```

4. **Configure MLFlow** (if using remote tracking):
   ```bash
   mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts
   ```

---

## Usage

### 1. Training a New Model

#### Option A: Using Python Script
```bash
cd data_modeling
python run_pipeline.py
```

#### Option B: Using Jupyter Notebook
Open `data_modeling/deploy.ipynb` and run all cells.

**Pipeline Stages**:
1. Data loading from Excel file
2. Data preprocessing (drop duplicates, handle nulls)
3. Feature engineering (scaling, encoding, selection)
4. Model training (advanced regression techniques)
5. Automatic model evaluation and promotion

### 2. Running the API Server

Start the FastAPI server:
```bash
cd api
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Access API documentation: `http://localhost:8000/docs`

### 3. Making Predictions via API

#### Example Request:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d {
    "Company": "TechCorp",
    "Campaign_Type": "Product Launch",
    "Target_Audience": "Men 25-34",
    "Duration": "30 days",
    "Channel_Used": "Google Ads",
    "Acquisition_Cost": 5000,
    "ROI": 2.5,
    "Location": "USA",
    "Clicks": 1500,
    "Impressions": 10000,
    "Engagement_Score": 6,
    "Customer_Segment": "Premium"
  }
```

#### Expected Response:
```json
{
  "predicted_conversion_rate": 0.082,
  "confidence": "high",
  "timestamp": "2024-03-25T10:30:00"
}
```

### 4. Data Analysis

#### Exploratory Data Analysis (EDA)
Open and run `data_analysis.ipynb` to explore:
- Distribution of campaign metrics
- Correlation analysis
- Channel performance comparison
- Audience segment analysis

#### Data Cleaning
Open and run `data_cleaning.ipynb` to see:
- Data quality assessment
- Missing value handling
- Duplicate removal
- Data validation

---

## Core Insights

### Key Findings from Campaign Analysis

> For detailed insights, see [Insights_README.md](./Insights_README.md)

#### 🔵 Discovery 1: The "Curiosity Click" Problem
- **Average CTR**: 14% (0.14)
- **Average Conversion Rate**: Only 8% (0.08)
- **Issue**: Clicks don't lead to conversions; need landing page optimization

#### 🟡 Discovery 2: Engagement ≠ Conversion
- **Engagement Score**: Virtually identical across all audience segments (~5.5/10)
- **Conversions**: Nearly same across all segments (43-44 per segment)
- **Implication**: Demographic targeting isn't the key leverage point

#### 🟢 Discovery 3: All Channels Are Equally Mediocre
- **Conversion Rates**: Identical across all channels (~16.7%)
- **ROI Distribution**: Perfectly balanced across channels
- **Impact**: Channel selection not the primary optimization opportunity

---

## Technology Stack

### Data Processing
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Scikit-Learn**: Machine learning algorithms

### ML Orchestration
- **ZenML**: Pipeline orchestration and step management
- **MLFlow**: Experiment tracking, model registry, and versioning

### API & Deployment
- **FastAPI**: Modern, fast web framework for APIs
- **Uvicorn**: ASGI server
- **Pydantic**: Data validation using Python type hints

### Analytics & Visualization
- **Jupyter**: Interactive notebooks for analysis
- **Power BI**: Dashboard and business intelligence (support)

### Version Control
- **Git**: Source code management

---

## API Documentation

### Base URL
```
http://localhost:8000
```

### Endpoints

#### 1. Health Check
```http
GET /health
```

**Response**:
```json
{
  "status": "ok",
  "model_loaded": true
}
```

#### 2. Predict Conversion Rate
```http
POST /predict
Content-Type: application/json
```

**Request Body**:
```json
{
  "Company": "string",
  "Campaign_Type": "string",
  "Target_Audience": "string",
  "Duration": "string",
  "Channel_Used": "string",
  "Acquisition_Cost": "integer",
  "ROI": "float",
  "Location": "string",
  "Clicks": "integer",
  "Impressions": "integer",
  "Engagement_Score": "integer",
  "Customer_Segment": "string"
}
```

**Response**:
```json
{
  "predicted_conversion_rate": "float",
  "model_version": "string",
  "timestamp": "string"
}
```

#### 3. Batch Predictions
```http
POST /predict-batch
Content-Type: application/json
```

**Request Body**: Array of campaign records

**Response**: Array of predictions with conversion rates

#### 4. Model Information
```http
GET /model-info
```

**Response**:
```json
{
  "model_name": "Conversion_Rate_Prediction_Model",
  "version": "Production",
  "accuracy_metrics": {
    "rmse": "float",
    "r2_score": "float"
  },
  "last_trained": "string"
}
```

---

## ML Pipeline

### Training Pipeline (`data_modeling/piplines/train_pipeline.py`)

```
Input Data
    ↓
┌─────────────────────┐
│ Load Data Step      │  → Load Excel file
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ Preprocessing Step  │  → Drop unnecessary columns
└──────────┤          │  → Remove duplicates
           │          │  → Handle missing values
           │          │  → Train/test split
           └──────────┘
           ↓
┌─────────────────────┐
│ Feature Engineering │  → Scale numeric features
│ Step                │  → Encode categorical features
└──────────┬──────────┘  → Feature selection (K-Best)
           ↓
┌─────────────────────┐
│ Training Step       │  → Train multiple models
└──────────┬──────────┘  → Hyperparameter tuning
           │             → Cross-validation
           ↓
┌─────────────────────┐
│ Evaluation & Deploy │  → Compare with prod model
└──────────┬──────────┘  → Register best model
           ↓
    Production Model
```

### Data Preprocessing

**Steps**:
1. **Load**: Read Excel dataset
2. **Clean**: 
   - Drop columns: `Date`, `Campaign_ID`
   - Remove duplicates
   - Handle missing values (mean imputation)
3. **Split**: 80% train, 20% test
4. **Feature Engineering**:
   - Numeric: StandardScaler
   - Categorical: OneHotEncoder
   - Selection: SelectKBest (top 10 features)

### Model Training

**Algorithm**: Advanced regression techniques with hyperparameter optimization

**Metrics**:
- Root Mean Squared Error (RMSE)
- R² Score

**Model Registry**: MLFlow (with Production alias)

---

## Deployment Pipeline

### Inference Pipeline (`data_modeling/deployment/pipelines/inference.py`)

For real-time predictions:

1. Load new campaign data
2. Apply same preprocessing as training
3. Load production model from MLFlow
4. Generate predictions
5. Return confidence scores

See [Deployment Guide](./data_modeling/deployment/README.md) for detailed instructions.

---

## Configuration

### MLFlow Configuration

**Tracking URI** (in `api/get_production_model.py`):
```python
mlflow.set_tracking_uri("http://localhost:8000")
```

Update this if using a remote MLFlow server.

### Model Selection

**Model Name**: `Conversion_Rate_Prediction_Model`

**Production Alias**: `Production` (automatically promoted)

---

## Monitoring & Logging

### Logging
- All pipeline steps include logging
- Format: `%(asctime)s - %(levelname)s - %(message)s`
- Configuration: See individual step files

### MLFlow Tracking
- Experiment tracking: `mlflow.pyfunc`
- Model registry: Automatic promotion to Production
- Run history: Available in MLFlow UI

---

## Troubleshooting

### Issue: Model not found
**Solution**: Ensure MLFlow server is running and model has been trained at least once.

### Issue: API connection timeout
**Solution**: Check MLFlow tracking URI configuration in `api/get_production_model.py`

### Issue: Preprocessing errors
**Solution**: Verify input data matches schema. See [DATA_PREPROCESSING_GUIDE.md](./data_modeling/deployment/DATA_PREPROCESSING_GUIDE.md)

### Issue: Rate limiting errors
**Solution**: Configure rate limiter in `api/rate_limitizer.py` or disable for development

---

## Performance Benchmarks

### Model Performance
- **RMSE**: [See MLFlow for latest run]
- **R² Score**: [See MLFlow for latest run]

### API Performance
- **Latency**: < 100ms per prediction
- **Throughput**: ~10 requests/second (single instance)
- **Max Batch Size**: 1000 records

---

## Future Enhancements

- [ ] Advanced ensemble methods (Gradient Boosting, XGBoost)
- [ ] Deep learning models for non-linear patterns
- [ ] Real-time data streaming integration
- [ ] Automated retraining pipeline with performance monitoring
- [ ] A/B testing framework for campaign optimization
- [ ] Real-time alerting for anomalous campaign performance
- [ ] Dashboard with real-time metrics visualization
- [ ] Causal inference for better attribution modeling

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Commit changes: `git commit -m "Add your feature"`
3. Push to branch: `git push origin feature/your-feature`
4. Open a Pull Request

---

## Code Style Guidelines

- Use type hints in function signatures
- Add docstrings to all functions
- Keep functions focused and modular
- Use logging instead of print statements
- Follow PEP 8 naming conventions

---

## License

This project is licensed under the MIT License - see LICENSE file for details.

---

## Contact & Support

For questions or issues:
- Review [Insights_README.md](./Insights_README.md) for business context
- Check [Deployment Guide](./data_modeling/deployment/README.md) for technical details
- See [DATA_PREPROCESSING_GUIDE.md](./data_modeling/deployment/DATA_PREPROCESSING_GUIDE.md) for data specifications

---

## Project Statistics

- **Total Campaign Records**: 200,000+
- **Marketing Channels**: 6
- **Audience Segments**: 5
- **Companies**: Multiple
- **Training Data Path**: `data/marketing_campaign_dataset.xlsx`
- **Model Framework**: ZenML + MLFlow
- **API Framework**: FastAPI

---

**Last Updated**: March 2024  
**Status**: Production Ready  
**Version**: 1.0.0
