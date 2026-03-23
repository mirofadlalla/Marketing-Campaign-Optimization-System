# Deployment Pipelines - Quick Guide

This folder contains two main ML pipelines for training and inference using ZenML and MLFlow.

---

## 📁 Folder Structure

```
deployment/
├── pipelines/                     # Pipeline definitions
│   ├── training_deployment.py    # Pipeline 1: Train & Deploy
│   └── inference.py              # Pipeline 2: Make Predictions
├── steps/                        # Individual processing steps
│   ├── predictor.py              # Make predictions
│   ├── prediction_service_loader.py  # Load model service
│   └── inference_data_loader.py  # Preprocess new data
├── utils/
│   └── deployment_trigger.py     # Model promotion logic
├── README.md                     # This file
└── DATA_PREPROCESSING_GUIDE.md   # Detailed preprocessing info
```

---

## 🚀 The Two Pipelines

### Pipeline 1: Training & Deployment

**File**: `pipelines/training_deployment.py`

**Purpose**: Train a new model and deploy it if it's better than the current production model.

**Basic Usage**:
```python
from data_modeling.deployment.pipelines.training_deployment import training_deployment_pipeline

# Run the pipeline
training_deployment_pipeline()
```

**What it does** (step by step):
1. ✅ **Load Data** - Reads marketing campaign dataset
2. ✅ **Clean Data** - Removes duplicates, nulls, unnecessary columns
3. ✅ **Engineer Features** - Scales numbers, encodes categories, selects best 10 features
4. ✅ **Train Model** - Trains conversion rate prediction model
5. ✅ **Evaluate** - Checks if new model is better than current production
6. ✅ **Deploy** - Registers model in MLFlow if it's the best version

**Output**: 
- Trained model registered in MLFlow
- Model metrics (R² score, RMSE)
- Model deployed and ready for predictions

---

### Pipeline 2: Make Predictions (Inference)

**File**: `pipelines/inference.py`

**Purpose**: Load new data and make predictions using the deployed model.

**Basic Usage**:
```python
from data_modeling.deployment.pipelines.inference import inference_pipeline

# Run inference on new data
predictions = inference_pipeline(
    data_path="/path/to/your/data.xlsx"
)
```

**What it does** (step by step):
1. ✅ **Load New Data** - Reads the data file you provide
2. ✅ **Clean Data** - Same cleaning as training (consistency!)
3. ✅ **Engineer Features** - Uses the SAME feature transformations from training
4. ✅ **Load Model** - Gets the deployed model from MLFlow
5. ✅ **Predict** - Makes predictions on new data

**Output**: 
- Predictions array from the model
- Probability or value for each row

---

## 🔄 How Data Flows

### Training Pipeline Flow
```
📥 Raw Data File
    ↓
🧹 Clean: Drop columns, nulls, duplicates
    ↓
⚙️  Feature Engineering: Scale, encode, select
    ↓
🤖 Train Model
    ↓
📊 Evaluate: Compare with production
    ↓
✅ Deploy to MLFlow
```

### Inference Pipeline Flow
```
📥 New Data File
    ↓
🧹 Clean: Same as training
    ↓
⚙️  Feature Engineering: Same transformations as training
    ↓
🔍 Load Trained Model
    ↓
🎯 Make Predictions
    ↓
📤 Return Predictions
```

---

## 💡 Quick Examples

### Example 1: Train and Deploy a New Model

```python
from data_modeling.deployment.pipelines.training_deployment import training_deployment_pipeline

# Train and deploy
training_deployment_pipeline(
    min_accuracy=0.5,      # Minimum accuracy threshold
    workers=1,             # Number of workers
    timeout=300            # Timeout in seconds
)

print("✅ Model trained and deployed!")
```

### Example 2: Make Predictions on New Data

```python
from data_modeling.deployment.pipelines.inference import inference_pipeline
import numpy as np

# Get predictions
predictions = inference_pipeline(
    data_path="/home/omar/Python/Marketing_Campaign_Optimization_System/data/new_data.xlsx"
)

print(f"Predictions: {predictions}")
print(f"Average prediction: {np.mean(predictions)}")
```

### Example 3: Run from Command Line

```bash
# Train
cd /home/omar/Python/Marketing_Campaign_Optimization_System
python -m data_modeling.deployment.pipelines.training_deployment

# Inference
python -m data_modeling.deployment.pipelines.inference
```

---

## 🔧 How Data is Processed

Both pipelines apply the same data processing to ensure consistency:

### 1. Data Cleaning
- Drop columns: `Date`, `Campaign_ID`
- Remove duplicate rows
- Drop columns with >80% missing values
- Remove highly correlated features (>0.95)

### 2. Feature Engineering
**For numeric columns** (Age, Income, etc.):
- Fill missing values with mean
- Scale to standard normal (mean=0, std=1)

**For categorical columns** (City, Region, etc.):
- Fill missing values with most common value
- Convert to one-hot encoding (binary features)

**For all features**:
- Select top 10 features most correlated with target

### 3. Train/Test Split (Training only)
- 80% for training
- 20% for testing/validation

---

## 🎯 Key Points to Remember

✅ **Consistency**: Inference uses the SAME preprocessing as training
- This prevents model prediction errors
- Model always sees data in the same format

✅ **Automatic Comparison**: Training pipeline compares new model with current one
- Only deploys if new model is better
- Always keeps the best version in production

✅ **MLFlow Integration**: Everything is tracked
- Model versions
- Performance metrics
- Preprocessing parameters

✅ **Preprocessing is Learned, Not Hardcoded**
- Training learns how to scale and encode
- Inference applies same learning
- New categories handled gracefully

---

## 📊 Configuration

### Data Path
Default: `/home/omar/Python/Marketing_Campaign_Optimization_System/data/marketing_campaign_dataset.xlsx`

Change by passing to pipeline:
```python
inference_pipeline(
    data_path="/path/to/your/data.xlsx"
)
```

### MLFlow Server
- URL: `http://localhost:8000`
- Model Name: `Conversion_Rate_Prediction_Model`
- Production Alias: `Production`

### Model Parameters
- Target: `Conversion_Rate`
- Features: All numeric and categorical columns
- Algorithm: Scikit-learn model (check training step)
- Features selected: Top 10 by f_regression score

---

## 🐛 Troubleshooting

### No model found error
```
RuntimeError: No MLflow prediction service found...
```
**Solution**: Run training pipeline first
```python
training_deployment_pipeline()
```

### Data format error
```
ValueError: Expected DataFrame...
```
**Solution**: Make sure data file is .xlsx format with same columns as training data

### Missing values error
```
ValueError: Missing required columns...
```
**Solution**: Check that your data has these columns:
- Numeric: Age, Income, etc.
- Categorical: City, Region, etc.
- Target (for training): Conversion_Rate

---

## 📚 More Information

For detailed information about data preprocessing, see: `DATA_PREPROCESSING_GUIDE.md`

For ZenML documentation: https://docs.zenml.io/
For MLFlow documentation: https://mlflow.org/docs/

---

## ✨ Summary

| Task | Pipeline | Command |
|------|----------|---------|
| Train new model | Training & Deployment | `training_deployment_pipeline()` |
| Make predictions | Inference | `inference_pipeline(data_path="...")` |
| Check current model | MLFlow UI | Visit `http://localhost:8000` |
| View logs | ZenML | `zenml logs` |
