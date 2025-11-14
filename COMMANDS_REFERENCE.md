# Hybrid Cloud Orchestrator - Commands Reference

Quick reference guide for all commands used to run, test, and validate this project.

**Table of Contents**:
- [Running the Application](#-running-the-application)
- [Data Preprocessing](#-data-preprocessing)
- [Data Validation & Quality Checks](#-data-validation--quality-checks)
- [Machine Learning Model Training](#-machine-learning-model-training)
- [Model Evaluation & Analysis](#-model-evaluation--analysis)
- [Testing](#-testing)
- [Installation & Setup](#-installation--setup)
- [API Testing](#-api-testing-curl)
- [Complete Workflow](#-complete-workflow)
- [Verification Commands](#-verification-commands)
- [Troubleshooting Commands](#-troubleshooting-commands)
- [Development Commands](#-development-commands)
- [Quick Command Cheat Sheet](#-quick-command-cheat-sheet)

---

## 🚀 Running the Application

### Start the Web Server (Main Application)

```bash
# From project root
cd backend
python app.py
```

**Or** (from project root):
```bash
python backend/app.py
```

**Access URLs**:
- Main page: http://127.0.0.1:5000
- Benchmark: http://127.0.0.1:5000/benchmark

**Stop the server**: Press `Ctrl+C` in the terminal

---

## 📊 Data Preprocessing

### Quick Run (Using Wrapper Script)

```bash
cd preprocessing
python run.py
```

### Manual Run (Full Control)

```bash
cd preprocessing
python preprocess.py --input ../data/Sample_Dataset.csv --output ../data/workload_dataset_sample.csv --cores 16 --mem_mb 65536
```

### Custom Configuration

```bash
cd preprocessing
python preprocess.py \
  --input ../data/Sample_Dataset.csv \
  --output ../data/workload_dataset_sample.csv \
  --cores 32 \
  --mem_mb 131072 \
  --seed 42
```

### Preprocessing Parameters

**Required Parameters**:
- `--input` / `-i`: Input CSV file path (required)

**Optional Parameters**:
- `--output` / `-o`: Output CSV file path (default: `workload_dataset_sample.csv`)
- `--cores`: Machine total CPU cores (default: 16)
- `--mem_mb`: Machine total memory in MB (default: 65536, which is 64 GB)
- `--seed`: Random seed for reproducible sampling (optional, for consistent results)

### Preprocessing Examples

**Basic preprocessing with defaults**:
```bash
cd preprocessing
python preprocess.py --input ../data/Sample_Dataset.csv
```

**High-memory machine configuration**:
```bash
cd preprocessing
python preprocess.py \
  --input ../data/Sample_Dataset.csv \
  --output ../data/workload_dataset_sample.csv \
  --cores 64 \
  --mem_mb 262144
```

**Reproducible preprocessing**:
```bash
cd preprocessing
python preprocess.py \
  --input ../data/Sample_Dataset.csv \
  --output ../data/workload_dataset_sample.csv \
  --seed 42
```

**Custom output location**:
```bash
cd preprocessing
python preprocess.py \
  --input ../data/Sample_Dataset.csv \
  --output ../data/my_processed_dataset.csv \
  --cores 16 \
  --mem_mb 65536
```

---

## 🔍 Data Validation & Quality Checks

### Validate Preprocessed Data

**Check data structure and basic statistics**:
```bash
# Python interactive session
python
```

```python
import pandas as pd
import numpy as np

# Load preprocessed data
df = pd.read_csv('data/workload_dataset_sample.csv')

# Check for missing values
print("Missing values:")
print(df.isnull().sum())

# Check data types
print("\nData types:")
print(df.dtypes)

# Basic statistics
print("\nBasic statistics:")
print(df.describe())

# Check value ranges
print("\nValue ranges:")
print(f"CPU cores: {df['cpu cores'].min():.2f} - {df['cpu cores'].max():.2f}")
print(f"Memory MB: {df['memory mb'].min():.2f} - {df['memory mb'].max():.2f}")
print(f"Execution time: {df['execution time'].min():.2f} - {df['execution time'].max():.2f}")

# Check target distribution
print("\nTarget platform distribution:")
print(df['target platform'].value_counts())
print(f"\nDistribution percentage:")
print(df['target platform'].value_counts(normalize=True) * 100)

# Check for infinite values
print("\nInfinite values:")
print(df.isin([np.inf, -np.inf]).sum())

# Check for negative values (should not exist)
print("\nNegative values:")
print((df.select_dtypes(include=[np.number]) < 0).sum())
```

### Quick Validation Script

Create a file `validate_data.py`:
```python
import pandas as pd
import numpy as np
import sys

def validate_data(csv_path):
    df = pd.read_csv(csv_path)
    
    print(f"Dataset: {csv_path}")
    print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
    print("\n" + "="*50)
    
    # Check missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print("⚠️  Missing values found:")
        print(missing[missing > 0])
    else:
        print("✅ No missing values")
    
    # Check infinite values
    inf_count = df.isin([np.inf, -np.inf]).sum().sum()
    if inf_count > 0:
        print(f"⚠️  {inf_count} infinite values found")
    else:
        print("✅ No infinite values")
    
    # Check target distribution
    if 'target platform' in df.columns:
        print("\nTarget distribution:")
        print(df['target platform'].value_counts())
        dist = df['target platform'].value_counts(normalize=True)
        if dist.min() < 0.2:
            print("⚠️  Imbalanced classes detected")
        else:
            print("✅ Balanced classes")
    
    # Check value ranges
    print("\nValue ranges:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != 'target platform':
            print(f"  {col}: [{df[col].min():.2f}, {df[col].max():.2f}]")
    
    print("\n" + "="*50)
    print("Validation complete!")

if __name__ == "__main__":
    validate_data(sys.argv[1] if len(sys.argv) > 1 else 'data/workload_dataset_sample.csv')
```

**Run validation**:
```bash
python validate_data.py data/workload_dataset_sample.csv
```

### Check Preprocessing Output

**View first few rows**:
```bash
# Linux/Mac
head -n 20 data/workload_dataset_sample.csv

# Windows (PowerShell)
Get-Content data/workload_dataset_sample.csv -Head 20
```

**Count rows and columns**:
```bash
# Linux/Mac
wc -l data/workload_dataset_sample.csv
head -n 1 data/workload_dataset_sample.csv | tr ',' '\n' | wc -l

# Windows (PowerShell)
(Get-Content data/workload_dataset_sample.csv | Measure-Object -Line).Lines
```

**Check file size**:
```bash
# Linux/Mac
ls -lh data/workload_dataset_sample.csv

# Windows
dir data\workload_dataset_sample.csv
```

---

## 🤖 Machine Learning Model Training

### Quick Run (Using Wrapper Script)

```bash
cd model
python run_model.py
```

### Manual Run (Full Control)

```bash
cd model
python train_model.py --input ../data/workload_dataset_sample.csv --output orchestrator_model.pkl
```

### Training Parameters

**Required Parameters**:
- `--input` / `-i`: Path to preprocessed dataset CSV (required)

**Optional Parameters**:
- `--output` / `-o`: Output model filename (default: `orchestrator_model.pkl`)

### Training Examples

**Basic training**:
```bash
cd model
python train_model.py --input ../data/workload_dataset_sample.csv
```

**Custom model output name**:
```bash
cd model
python train_model.py \
  --input ../data/workload_dataset_sample.csv \
  --output my_model_v1.pkl
```

**Training with different dataset**:
```bash
cd model
python train_model.py \
  --input ../data/custom_dataset.csv \
  --output custom_model.pkl
```

### Training Outputs

After training, you'll get:
- `orchestrator_model.pkl`: Trained RandomForest model (pickle format)
- `orchestrator_model_feature_importance.png`: Feature importance bar chart
- Console output with:
  - Accuracy score
  - Confusion matrix
  - Classification report (precision, recall, F1)
  - Top contributing features list

### Training Time Estimates

- **Small dataset (< 100 rows)**: 5-15 seconds
- **Medium dataset (100-1000 rows)**: 15-60 seconds
- **Large dataset (1000+ rows)**: 1-5 minutes

---

## 📈 Model Evaluation & Analysis

### View Model Performance Metrics

**After training, metrics are printed to console. To save them**:

```bash
cd model
python train_model.py --input ../data/workload_dataset_sample.csv --output orchestrator_model.pkl > training_output.txt 2>&1
```

**View saved output**:
```bash
# Linux/Mac
cat training_output.txt

# Windows
type training_output.txt
```

### Analyze Feature Importance

**View feature importance plot**:
```bash
# Linux/Mac
open model/orchestrator_model_feature_importance.png

# Windows
start model\orchestrator_model_feature_importance.png
```

**Extract feature importance programmatically**:
```python
import joblib
import pandas as pd

# Load model
model = joblib.load('model/orchestrator_model.pkl')

# Get feature importance
features = ['cpu cores', 'memory mb', 'latency sensitive', 
            'execution time', 'data size mb', 'cost ratio']
importance = pd.Series(model.feature_importances_, index=features)

# Sort by importance
print(importance.sort_values(ascending=False))
```

### Model Prediction Testing

**Test model with sample data**:
```python
import joblib
import pandas as pd

# Load model
model = joblib.load('model/orchestrator_model.pkl')

# Sample workload
sample = [[1.5, 1024, 0, 120, 300, 2.5]]  # [cpu, mem_mb, latency, time, data_size, cost_ratio]

# Predict
prediction = model.predict(sample)[0]
probability = model.predict_proba(sample)[0]

platform = "Serverless" if prediction == 0 else "Traditional"
print(f"Predicted platform: {platform}")
print(f"Confidence: Serverless={probability[0]:.2%}, Traditional={probability[1]:.2%}")
```

### Compare Model vs Rules

**Create comparison script** (`compare_predictions.py`):
```python
import joblib
import pandas as pd

# Load model
model = joblib.load('model/orchestrator_model.pkl')
df = pd.read_csv('data/workload_dataset_sample.csv')

# Rule-based prediction function
def rule_based_predict(row):
    if (row['execution time'] <= 300 and 
        row['cpu cores'] <= 2 and 
        row['memory mb'] <= 2048 and 
        row['latency sensitive'] == 0 and 
        row['cost ratio'] < 5):
        return 0  # Serverless
    return 1  # Traditional

# Compare predictions
features = ['cpu cores', 'memory mb', 'latency sensitive', 
            'execution time', 'data size mb', 'cost ratio']
X = df[features]
y_true = df['target platform'].map({'serverless': 0, 'traditional': 1})

# Model predictions
y_pred_model = model.predict(X)

# Rule-based predictions
y_pred_rules = df.apply(rule_based_predict, axis=1)

# Calculate agreement
agreement = (y_pred_model == y_pred_rules).sum() / len(df) * 100
print(f"Model and rules agree: {agreement:.2f}% of the time")
```

**Run comparison**:
```bash
python compare_predictions.py
```

### Model Versioning

**Save model with version**:
```bash
cd model
python train_model.py \
  --input ../data/workload_dataset_sample.csv \
  --output orchestrator_model_v1_20240101.pkl
```

**List all model versions**:
```bash
# Linux/Mac
ls -lh model/*.pkl

# Windows
dir model\*.pkl
```

---

## 🧪 Testing

### Run All Tests

```bash
# From project root
pytest tests/
```

### Run Specific Test File

```bash
pytest tests/test_deploy.py
```

### Verbose Output

```bash
pytest tests/ -v
```

### With Coverage Report

```bash
pytest tests/ --cov=backend --cov-report=html
```

### Run Specific Test Function

```bash
pytest tests/test_deploy.py::test_render_deployment_files_serverless -v
```

---

## 🔧 Installation & Setup

### Install Backend Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### Install Development Dependencies

```bash
# From project root
pip install -r requirements-dev.txt
```

### Install All Dependencies

```bash
pip install -r backend/requirements.txt
pip install -r requirements-dev.txt
```

---

## 🌐 API Testing (cURL)

### Test Prediction Endpoint

**Basic prediction request**:
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "cpu_cores": 1.5,
    "memory_mb": 1024,
    "latency_sensitive": 0,
    "execution_time": 120,
    "data_size_mb": 300
  }'
```

**Serverless-friendly workload**:
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "cpu_cores": 1.0,
    "memory_mb": 512,
    "latency_sensitive": 0,
    "execution_time": 60,
    "data_size_mb": 100
  }'
```

**Traditional-friendly workload**:
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "cpu_cores": 8,
    "memory_mb": 16384,
    "latency_sensitive": 1,
    "execution_time": 3600,
    "data_size_mb": 50000
  }'
```

**Save response to file**:
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "cpu_cores": 1.5,
    "memory_mb": 1024,
    "latency_sensitive": 0,
    "execution_time": 120,
    "data_size_mb": 300
  }' \
  -o prediction_response.json
```

**Pretty print JSON response** (requires `jq`):
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "cpu_cores": 1.5,
    "memory_mb": 1024,
    "latency_sensitive": 0,
    "execution_time": 120,
    "data_size_mb": 300
  }' | jq .
```

### Test Deployment Bundle Endpoint

**Generate deployment bundle**:
```bash
curl -X POST http://127.0.0.1:5000/deploy \
  -H "Content-Type: application/json" \
  -d '{
    "cpu_cores": 1.5,
    "memory_mb": 1024,
    "latency_sensitive": 0,
    "execution_time": 120,
    "data_size_mb": 300
  }' \
  --output deploy_bundle.zip
```

**Generate serverless bundle**:
```bash
curl -X POST http://127.0.0.1:5000/deploy \
  -H "Content-Type: application/json" \
  -d '{
    "cpu_cores": 1.0,
    "memory_mb": 512,
    "latency_sensitive": 0,
    "execution_time": 60,
    "data_size_mb": 100
  }' \
  --output serverless_bundle.zip
```

**Generate traditional bundle**:
```bash
curl -X POST http://127.0.0.1:5000/deploy \
  -H "Content-Type: application/json" \
  -d '{
    "cpu_cores": 8,
    "memory_mb": 16384,
    "latency_sensitive": 1,
    "execution_time": 3600,
    "data_size_mb": 50000
  }' \
  --output traditional_bundle.zip
```

**Verify bundle contents** (after download):
```bash
# Linux/Mac
unzip -l deploy_bundle.zip

# Windows (PowerShell)
Expand-Archive -Path deploy_bundle.zip -DestinationPath bundle_contents -Force
Get-ChildItem bundle_contents
```

### Test Benchmark Data Endpoint

**Get benchmark data**:
```bash
curl http://127.0.0.1:5000/benchmark-data
```

**Save benchmark data**:
```bash
curl http://127.0.0.1:5000/benchmark-data -o benchmark_data.json
```

**Pretty print** (requires `jq`):
```bash
curl http://127.0.0.1:5000/benchmark-data | jq .
```

### Test Home Page

```bash
curl http://127.0.0.1:5000/
```

### Test Error Handling

**Missing required field**:
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "cpu_cores": 1.5,
    "memory_mb": 1024
  }'
```

**Invalid data type**:
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "cpu_cores": "invalid",
    "memory_mb": 1024,
    "latency_sensitive": 0,
    "execution_time": 120,
    "data_size_mb": 300
  }'
```

**Negative values**:
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "cpu_cores": -1,
    "memory_mb": 1024,
    "latency_sensitive": 0,
    "execution_time": 120,
    "data_size_mb": 300
  }'
```

### API Testing with Python

**Create test script** (`test_api.py`):
```python
import requests
import json

BASE_URL = "http://127.0.0.1:5000"

# Test prediction
def test_predict():
    payload = {
        "cpu_cores": 1.5,
        "memory_mb": 1024,
        "latency_sensitive": 0,
        "execution_time": 120,
        "data_size_mb": 300
    }
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    print("Prediction Response:")
    print(json.dumps(response.json(), indent=2))
    return response.json()

# Test deploy
def test_deploy():
    payload = {
        "cpu_cores": 1.5,
        "memory_mb": 1024,
        "latency_sensitive": 0,
        "execution_time": 120,
        "data_size_mb": 300
    }
    response = requests.post(f"{BASE_URL}/deploy", json=payload)
    with open("deploy_bundle.zip", "wb") as f:
        f.write(response.content)
    print("Deployment bundle saved to deploy_bundle.zip")

# Test benchmark
def test_benchmark():
    response = requests.get(f"{BASE_URL}/benchmark-data")
    print("Benchmark Data:")
    print(json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    test_predict()
    test_deploy()
    test_benchmark()
```

**Run Python API tests**:
```bash
pip install requests
python test_api.py
```

---

## 📦 Complete Setup Workflow

### Phase 1: First-Time Setup

**1. Install Dependencies**:
```bash
# Install backend dependencies
cd backend
pip install -r requirements.txt
cd ..

# Install development dependencies
pip install -r requirements-dev.txt
```

**2. Verify Installation**:
```bash
python --version  # Should be 3.11+
pip list | grep -E "Flask|scikit-learn|pandas|pytest"
```

**3. Verify Data Files**:
```bash
# Check if input data exists
ls -lh data/Sample_Dataset.csv  # Linux/Mac
dir data\Sample_Dataset.csv     # Windows
```

### Phase 2: Data Preparation (Optional)

**4. Preprocess Data**:
```bash
cd preprocessing
python run.py
cd ..
```

**5. Validate Preprocessed Data**:
```bash
python validate_data.py data/workload_dataset_sample.csv
```

### Phase 3: Model Training (Optional)

**6. Train Model**:
```bash
cd model
python run_model.py
cd ..
```

**7. Review Model Performance**:
```bash
# Check feature importance plot
open model/orchestrator_model_feature_importance.png  # Mac
start model\orchestrator_model_feature_importance.png  # Windows
```

### Phase 4: Application Launch

**8. Start Web Application**:
```bash
cd backend
python app.py
```

**9. Access Application**:
- Open browser: http://127.0.0.1:5000
- Test prediction endpoint
- Test deployment bundle generation

### Phase 5: Testing

**10. Run Tests**:
```bash
pytest tests/ -v
```

### Complete Setup Script (All-in-One)

**Create setup script** (`setup.sh` for Linux/Mac or `setup.bat` for Windows):

**Linux/Mac** (`setup.sh`):
```bash
#!/bin/bash
set -e

echo "=== Installing Dependencies ==="
cd backend && pip install -r requirements.txt && cd ..
pip install -r requirements-dev.txt

echo "=== Preprocessing Data ==="
cd preprocessing && python run.py && cd ..

echo "=== Training Model ==="
cd model && python run_model.py && cd ..

echo "=== Setup Complete! ==="
echo "Start the server with: cd backend && python app.py"
```

**Windows** (`setup.bat`):
```batch
@echo off
echo === Installing Dependencies ===
cd backend
pip install -r requirements.txt
cd ..
pip install -r requirements-dev.txt

echo === Preprocessing Data ===
cd preprocessing
python run.py
cd ..

echo === Training Model ===
cd model
python run_model.py
cd ..

echo === Setup Complete! ===
echo Start the server with: cd backend ^&^& python app.py
```

**Run setup script**:
```bash
# Linux/Mac
chmod +x setup.sh
./setup.sh

# Windows
setup.bat
```

### Daily Usage

**Quick Start** (if everything is already set up):
```bash
cd backend
python app.py
```

**With Data Refresh**:
```bash
# Preprocess new data
cd preprocessing && python run.py && cd ..

# Retrain model
cd model && python run_model.py && cd ..

# Start server
cd backend && python app.py
```

---

## 🔍 Verification Commands

### Check Python Version

```bash
python --version
# Should be 3.11 or higher

# Alternative
python3 --version
```

### Check Installed Packages

**List all installed packages**:
```bash
pip list
```

**Check specific packages**:
```bash
# Linux/Mac
pip list | grep -E "Flask|scikit-learn|pandas|pytest"

# Windows (PowerShell)
pip list | Select-String -Pattern "Flask|scikit-learn|pandas|pytest"
```

**Check package versions**:
```bash
pip show Flask
pip show scikit-learn
pip show pandas
pip show pytest
```

### Verify File Structure

**Check directory structure**:
```bash
# Linux/Mac
ls -la backend/
ls -la frontend/
ls -la model/
ls -la preprocessing/
ls -la data/
ls -la tests/

# Windows
dir backend
dir frontend
dir model
dir preprocessing
dir data
dir tests
```

**Check for required files**:
```bash
# Linux/Mac
ls -lh backend/app.py
ls -lh data/Sample_Dataset.csv
ls -lh model/train_model.py
ls -lh preprocessing/preprocess.py

# Windows
dir backend\app.py
dir data\Sample_Dataset.csv
dir model\train_model.py
dir preprocessing\preprocess.py
```

**Check generated files**:
```bash
# Linux/Mac
ls -lh data/workload_dataset_sample.csv
ls -lh model/orchestrator_model.pkl
ls -lh model/orchestrator_model_feature_importance.png

# Windows
dir data\workload_dataset_sample.csv
dir model\orchestrator_model.pkl
dir model\orchestrator_model_feature_importance.png
```

### Check if Server is Running

**Test server endpoint**:
```bash
# Linux/Mac
curl http://127.0.0.1:5000/

# Windows (PowerShell)
Invoke-WebRequest -Uri http://127.0.0.1:5000/

# Windows (CMD)
curl http://127.0.0.1:5000/
```

**Check server health**:
```bash
# Linux/Mac
curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:5000/

# Windows (PowerShell)
(Invoke-WebRequest -Uri http://127.0.0.1:5000/).StatusCode
```

**Test all endpoints**:
```bash
# Test home page
curl http://127.0.0.1:5000/

# Test benchmark page
curl http://127.0.0.1:5000/benchmark

# Test benchmark data
curl http://127.0.0.1:5000/benchmark-data

# Test prediction (should return error without payload)
curl -X POST http://127.0.0.1:5000/predict
```

### Verify Data Quality

**Check preprocessed data**:
```bash
# Count rows
wc -l data/workload_dataset_sample.csv  # Linux/Mac
(Get-Content data\workload_dataset_sample.csv | Measure-Object -Line).Lines  # Windows

# Check file size
ls -lh data/workload_dataset_sample.csv  # Linux/Mac
dir data\workload_dataset_sample.csv     # Windows
```

**Quick data validation**:
```bash
python -c "import pandas as pd; df = pd.read_csv('data/workload_dataset_sample.csv'); print(f'Rows: {len(df)}, Columns: {len(df.columns)}'); print(df.head())"
```

### Verify Model Files

**Check model file exists and size**:
```bash
# Linux/Mac
ls -lh model/orchestrator_model.pkl
file model/orchestrator_model.pkl

# Windows
dir model\orchestrator_model.pkl
```

**Test model loading**:
```bash
python -c "import joblib; model = joblib.load('model/orchestrator_model.pkl'); print('Model loaded successfully'); print(f'Model type: {type(model)}')"
```

**Check feature importance plot**:
```bash
# Linux/Mac
file model/orchestrator_model_feature_importance.png
ls -lh model/orchestrator_model_feature_importance.png

# Windows
dir model\orchestrator_model_feature_importance.png
```

---

## 🐛 Troubleshooting Commands

### Check for Port Conflicts

**Linux/Mac**:
```bash
# Check what's using port 5000
lsof -i :5000

# Kill process on port 5000
lsof -ti :5000 | xargs kill -9

# Alternative: use netstat
netstat -an | grep 5000
```

**Windows**:
```bash
# Check what's using port 5000
netstat -ano | findstr :5000

# Kill process (replace PID with actual process ID)
taskkill /PID <PID> /F
```

**Find and kill Flask process**:
```bash
# Linux/Mac
pkill -f "python.*app.py"

# Windows
taskkill /F /IM python.exe
```

### Clear Python Cache

**Linux/Mac**:
```bash
# From project root
find . -type d -name __pycache__ -exec rm -r {} +
find . -type f -name "*.pyc" -delete
find . -type f -name "*.pyo" -delete
```

**Windows (PowerShell)**:
```powershell
Get-ChildItem -Path . -Include __pycache__,*.pyc,*.pyo -Recurse | Remove-Item -Force -Recurse
```

**Windows (CMD)**:
```batch
for /d /r . %d in (__pycache__) do @if exist "%d" rd /s /q "%d"
for /r . %f in (*.pyc) do @if exist "%f" del /f /q "%f"
```

### Reinstall Dependencies

**Clean reinstall**:
```bash
# Uninstall packages
pip uninstall Flask scikit-learn pandas pytest -y

# Reinstall from requirements
pip install -r backend/requirements.txt
pip install -r requirements-dev.txt
```

**Force reinstall**:
```bash
pip install --force-reinstall -r backend/requirements.txt
pip install --force-reinstall -r requirements-dev.txt
```

**Install specific versions**:
```bash
pip install Flask==2.3.0
pip install scikit-learn==1.3.0
pip install pandas==2.0.0
```

### Fix Import Errors

**Check Python path**:
```bash
python -c "import sys; print('\n'.join(sys.path))"
```

**Test imports**:
```bash
python -c "import flask; print('Flask OK')"
python -c "import sklearn; print('scikit-learn OK')"
python -c "import pandas; print('pandas OK')"
python -c "import pytest; print('pytest OK')"
```

### Fix Data Issues

**Check data file encoding**:
```bash
# Linux/Mac
file data/Sample_Dataset.csv
head -n 1 data/Sample_Dataset.csv | file -

# Windows (PowerShell)
Get-Content data\Sample_Dataset.csv -Encoding UTF8 -TotalCount 1
```

**Fix encoding issues**:
```bash
# Convert to UTF-8 (Linux/Mac)
iconv -f ISO-8859-1 -t UTF-8 data/Sample_Dataset.csv > data/Sample_Dataset_utf8.csv
```

**Check CSV format**:
```bash
# Count columns in first row
head -n 1 data/Sample_Dataset.csv | tr ',' '\n' | wc -l  # Linux/Mac
```

### Fix Model Issues

**Test model loading**:
```bash
python -c "import joblib; model = joblib.load('model/orchestrator_model.pkl'); print('OK')"
```

**Recreate model if corrupted**:
```bash
cd model
python run_model.py
cd ..
```

**Check model file integrity**:
```bash
# Linux/Mac
file model/orchestrator_model.pkl
ls -lh model/orchestrator_model.pkl

# Windows
dir model\orchestrator_model.pkl
```

### Debug Flask Application

**Run with verbose logging**:
```bash
cd backend
FLASK_DEBUG=1 python app.py
```

**Check Flask logs**:
```bash
# Logs are printed to console when debug=True
# Check terminal output for errors
```

**Test individual functions**:
```python
# In Python interactive session
from backend.app import _compute_costs, _recommend_platform

# Test cost calculation
cost_trad, cost_srv, ratio = _compute_costs(1.5, 1024, 120)
print(f"Traditional: ${cost_trad}, Serverless: ${cost_srv}, Ratio: {ratio}")

# Test platform recommendation
platform = _recommend_platform(1.5, 1024, 0, 120)
print(f"Recommended: {platform}")
```

### Common Error Solutions

**"Module not found" error**:
```bash
# Ensure you're in the correct directory
# Install missing package
pip install <package_name>
```

**"Port already in use" error**:
```bash
# Kill process on port 5000 (see port conflict commands above)
# Or change port in app.py
```

**"File not found" error**:
```bash
# Check file paths are correct
# Ensure you're running from project root
pwd  # Linux/Mac
cd   # Windows
```

**"Permission denied" error**:
```bash
# Linux/Mac: Add execute permission
chmod +x preprocessing/run.py
chmod +x model/run_model.py

# Windows: Run as administrator if needed
```

---

## 📝 Development Commands

### Run with Debug Mode (Default)

```bash
cd backend
python app.py
# Debug mode is enabled by default (app.run(debug=True))
```

### Run with Custom Port

Modify `backend/app.py`:
```python
if __name__ == '__main__':
    app.run(debug=True, port=8080)  # Change port here
```

### Run Tests in Watch Mode (if pytest-watch installed)

```bash
pip install pytest-watch
ptw tests/
```

---

## 🎯 Quick Command Cheat Sheet

### Essential Commands

| Task | Command |
|------|---------|
| **Start server** | `cd backend && python app.py` |
| **Preprocess data** | `cd preprocessing && python run.py` |
| **Train model** | `cd model && python run_model.py` |
| **Run tests** | `pytest tests/` |
| **Install deps** | `pip install -r backend/requirements.txt` |
| **Check Python version** | `python --version` |

### API Testing Commands

| Task | Command |
|------|---------|
| **Test prediction** | `curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"cpu_cores":1.5,"memory_mb":1024,"latency_sensitive":0,"execution_time":120,"data_size_mb":300}'` |
| **Get benchmark data** | `curl http://127.0.0.1:5000/benchmark-data` |
| **Download deploy bundle** | `curl -X POST http://127.0.0.1:5000/deploy -H "Content-Type: application/json" -d '{"cpu_cores":1.5,"memory_mb":1024,"latency_sensitive":0,"execution_time":120,"data_size_mb":300}' -o bundle.zip` |

### Data Commands

| Task | Command |
|------|---------|
| **Validate data** | `python validate_data.py data/workload_dataset_sample.csv` |
| **View data stats** | `python -c "import pandas as pd; df = pd.read_csv('data/workload_dataset_sample.csv'); print(df.describe())"` |
| **Count rows** | `wc -l data/workload_dataset_sample.csv` (Linux/Mac) |

### Model Commands

| Task | Command |
|------|---------|
| **Load and test model** | `python -c "import joblib; m = joblib.load('model/orchestrator_model.pkl'); print('OK')"` |
| **View feature importance** | `open model/orchestrator_model_feature_importance.png` (Mac) |

### Troubleshooting Commands

| Task | Command |
|------|---------|
| **Check port** | `lsof -i :5000` (Linux/Mac) or `netstat -ano \| findstr :5000` (Windows) |
| **Clear cache** | `find . -type d -name __pycache__ -exec rm -r {} +` (Linux/Mac) |
| **Reinstall deps** | `pip install --force-reinstall -r backend/requirements.txt` |

### Workflow Commands

| Phase | Commands |
|-------|----------|
| **Setup** | `pip install -r backend/requirements.txt && pip install -r requirements-dev.txt` |
| **Data Prep** | `cd preprocessing && python run.py && cd ..` |
| **Model Train** | `cd model && python run_model.py && cd ..` |
| **Run App** | `cd backend && python app.py` |
| **Test** | `pytest tests/ -v` |

---

## 📌 Important Notes

### General Guidelines

1. **Always run commands from the correct directory** - The project uses relative paths
2. **Python 3.11+ required** - Check with `python --version`
3. **Server runs on port 5000 by default** - Ensure the port is available
4. **Data files must exist** - Ensure `data/Sample_Dataset.csv` is present before preprocessing
5. **Model training is optional** - The app works with rule-based heuristics even without the trained model

### Directory Structure

- **Backend commands**: Run from `backend/` directory or use `python backend/app.py` from root
- **Preprocessing commands**: Run from `preprocessing/` directory
- **Model commands**: Run from `model/` directory
- **Tests**: Run from project root with `pytest tests/`

### Path Conventions

- **Linux/Mac**: Use forward slashes `/` in paths
- **Windows**: Use backslashes `\` in paths or forward slashes `/` (both work)
- **Relative paths**: Always relative to current working directory

### Command Differences by OS

| Task | Linux/Mac | Windows |
|------|-----------|---------|
| **List files** | `ls -la` | `dir` |
| **View file** | `cat file.txt` | `type file.txt` |
| **Kill process** | `kill -9 <PID>` | `taskkill /PID <PID> /F` |
| **Check port** | `lsof -i :5000` | `netstat -ano \| findstr :5000` |
| **Open file** | `open file.png` | `start file.png` |

### Performance Tips

- **Preprocessing**: Takes 1-5 minutes depending on dataset size
- **Model training**: Takes 30 seconds - 2 minutes for typical datasets
- **API response**: Should be < 1 second for predictions
- **Large datasets**: Consider increasing sample size or using parallel processing

### Best Practices

1. **Always validate data** after preprocessing
2. **Check model performance** before deploying
3. **Run tests** after making changes
4. **Use version control** for model files
5. **Document** any custom configurations

---

## 📚 Additional Resources

- **Detailed Documentation**: See `PROJECT_EXPLANATION.md` for comprehensive project documentation
- **Code Comments**: Check source files for inline documentation
- **Error Messages**: Read error messages carefully - they often contain helpful hints

---

**Last Updated**: See PROJECT_EXPLANATION.md for detailed documentation and explanations.


