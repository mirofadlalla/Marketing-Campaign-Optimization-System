# Data Preprocessing Pipeline - Detailed Guide

## Overview

The data preprocessing pipeline is consistent between training and inference. This ensures that the model always sees data in the same format during training and prediction.

## Preprocessing Steps

### 1. **Data Loading**
```python
from data_modeling.steps.load_df import data_load

df = data_load(data_path="path/to/marketing_campaign_dataset.xlsx")
```
- Loads Excel file using pandas
- Converts 'Date' column to datetime format
- Returns raw DataFrame

**Output**: Raw DataFrame with all columns

---

### 2. **Data Cleaning** (`data_preprosser` step)

#### Step 2a: Drop Unnecessary Columns
```python
droped_cols = ['Date', 'Campaign_ID']
df = df.drop([col for col in droped_cols if col in df.columns], axis=1)
```
- Removes non-predictive columns
- Keeps numeric and categorical features plus target

#### Step 2b: Remove Duplicates
```python
df = df.drop_duplicates()
```
- Removes exact duplicate rows

#### Step 2c: Handle Missing Values
```python
null_features = df.columns[(df.isna().mean() > 0.8)]
df = df.drop(null_features, axis=1)
```
- Drops any column with more than 80% missing values
- These columns are too sparse to be useful

#### Step 2d: Train/Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```
- Splits data 80% training, 20% testing
- Keeps target variable (`Conversion_Rate`) separate

**Output**: Clean training and test sets

---

### 3. **Feature Engineering Pipeline**

#### Step 3a: Identify Column Types
```python
num_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
```

#### Step 3b: Numeric Pipeline
```python
num_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),      # Fill NaN with mean
    ('scaler', StandardScaler())                       # Normalize to mean=0, std=1
])
```
- **Imputer**: Fills remaining NaN values with column mean
- **Scaler**: Normalizes values to standard normal distribution

#### Step 3c: Categorical Pipeline  
```python
cat_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Fill NaN with mode
    ('encoder', OneHotEncoder(handle_unknown="ignore"))    # Convert to binary features
])
```
- **Imputer**: Fills NaN with most common category
- **Encoder**: Converts categories to binary (one-hot) features
  - Example: Color → [is_red=1, is_blue=0, is_green=0]

#### Step 3d: Combine Pipelines
```python
preprocessor = ColumnTransformer([
    ('nums', num_pipeline, num_cols),
    ('cats', cat_pipeline, cat_cols),
])
```
- Applies numeric pipeline to numeric columns
- Applies categorical pipeline to categorical columns
- Combines results

#### Step 3e: Feature Selection
```python
final_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('feature_selection', SelectKBest(score_func=f_regression, k=10))
])
```
- **SelectKBest**: Selects top 10 features with highest correlation to target
- Reduces dimensionality and improves model efficiency

**Output**: Processed features ready for modeling

---

### 4. **Training Data Processing**
```python
X_train_processed = final_pipeline.fit_transform(X_train, y_train)
X_test_processed = final_pipeline.transform(X_test)
```
- **fit_transform**: Learns transformations from training data
- **transform**: Applies same transformations to test data
- Both use same preprocessing parameters

**Output**: Clean, scaled, engineered features

---

### 5. **Inference Data Processing**

For new data coming in for predictions:

```python
X_inference = inference_data_preprocessor(
    df=raw_data,
    pipeline=preprocessing_pipeline
)
```

**Important**: Uses the **fitted preprocessing pipeline** from training
- No fitting - only transformation
- Ensures consistency with training

**Steps**:
1. Drop same columns as training
2. Remove duplicates
3. Drop high-nullity columns
4. Apply fitted preprocessing (impute, scale, encode)
5. Apply fitted feature selection
6. Return processed data ready for prediction

---

## Data Flow Diagram

```
Training:
├─ Load Data
├─ Clean (drop cols, nulls, duplicates)
├─ Split (train 80%, test 20%)
├─ Feature Engineering
│  ├─ Numeric: Impute → Scale
│  ├─ Categorical: Impute → One-Hot Encode
│  └─ Feature Selection (top 10)
└─ Train Model

Inference:
├─ Load New Data
├─ Clean (same as training)
├─ Apply Fitted Feature Engineering
│  ├─ Apply learned transformations
│  ├─ Apply learned scaling
│  └─ Apply learned feature selection
└─ Predict
```

---

## Why This Matters

### ✅ **Consistency**
- Model always sees data in same format
- Prevents training/inference mismatch

### ✅ **Prevention of Data Leakage**
- Test data uses fitted parameters from training
- No information from test set affects training

### ✅ **Prevention of Inference Leakage**
- New data never influences model
- Uses only training parameters

### ✅ **Handling New Categories**
- `OneHotEncoder(handle_unknown="ignore")` ignores unseen categories
- Returns all zeros for unknown categories

### ✅ **Robustness**
- Missing values handled consistently
- Outliers reduced by standardization
- Irrelevant features removed

---

## Example

### Raw Data
```
Age | Income | City       | Conversion_Rate
25  | 50000  | New York   | 1
NaN | 60000  | Los Angeles| 0
35  | 75000  | Chicago    | 1
```

### After Cleaning
```
Age | Income | Conversion_Rate
25  | 50000  | 1
35  | 75000  | 1
```
(Row with NaN removed)

### After Feature Engineering (numeric)
```
Age_scaled | Income_scaled | Conversion_Rate
-0.5       | -1.2          | 1
 0.5       |  1.2          | 1
```
(Values standardized)

### After Feature Engineering (with City)
```
Age_scaled | Income_scaled | City_LA | City_NY | Conversion_Rate
-0.5       | -1.2          | 0       | 1       | 1
 0.5       |  1.2          | 0       | 0       | 1
```
(City one-hot encoded)

### After Feature Selection
Keep top 10 features by f_regression score
(Excludes low-value features)

---

## Configuration

You can modify preprocessing in each step:

### In `data_preprossing.py`:
- Change columns to drop: `droped_cols = [...] `
- Change null threshold: `df.isna().mean() > 0.8`
- Change correlation threshold: `abs(corr) > 0.95`
- Change test size: `test_size=0.2`

### In `feature_engineering.py`:
- Change number of features: `k=10`
- Change numeric strategy: `strategy='mean'` → 'median', 'most_frequent'
- Change categorical strategy: `strategy='most_frequent'` → 'constant'
- Change encoding: `OneHotEncoder` → `OrdinalEncoder`

---

## Notes

- Preprocessing is **saved with the model** by MLFlow
- Inference automatically uses the **same preprocessing**
- No need to manually apply transformations
- All parameters are **version controlled** with the model

