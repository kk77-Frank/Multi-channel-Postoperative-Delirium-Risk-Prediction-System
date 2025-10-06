# ECG-Based Postoperative Delirium Prediction

A machine learning system for predicting postoperative delirium (POD) risk using 12-lead ECG signals.

## Features

- 🔬 Multi-lead ECG signal processing and feature extraction
- 🤖 Multiple ML models (Random Forest, XGBoost, SVM, etc.)
- 🌐 Web-based interface with Streamlit
- 📊 Comprehensive model evaluation and visualization
- 🔍 SHAP-based interpretability analysis


# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Web Interface (Recommended)

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501`

### Command Line

**Train models:**
```bash
python ecg_delirium_predictor.py \
    --delirium_file data/PND_features.csv \
    --non_delirium_file data/NPND_features.csv \
    --output_dir results
```

**Make predictions:**
```bash
python predict_delirium.py <ecg_file.csv>
```

## Data Format

CSV file with 12-lead ECG signals:
- Required columns: `MDC_ECG_LEAD_I`, `MDC_ECG_LEAD_II`, `MDC_ECG_LEAD_III`, `MDC_ECG_LEAD_aVR`, `MDC_ECG_LEAD_aVL`, `MDC_ECG_LEAD_aVF`, `MDC_ECG_LEAD_V1`, `MDC_ECG_LEAD_V2`, `MDC_ECG_LEAD_V3`, `MDC_ECG_LEAD_V4`, `MDC_ECG_LEAD_V5`, `MDC_ECG_LEAD_V6`
- Sampling rate: 500/1000/2000 Hz
- Minimum length: 100 samples

## Project Structure

```
├── app.py                      # Streamlit web app
├── ecg_delirium_predictor.py  # Model training pipeline
├── multi_model_builder.py     # Multi-model training
├── predict_delirium.py        # Prediction module
├── ecg_modules/               # Core modules
│   ├── preprocessing.py       # Signal preprocessing
│   ├── feature_extraction.py  # Feature extraction
│   └── model_building.py      # Model building
├── data/                      # Data directory
├── models/                    # Trained models
└── results/                   # Output results
```

## Models

The system implements 7 machine learning models:
- Random Forest
- XGBoost (GPU-accelerated)
- ExtraTrees
- Gradient Boosting
- K-Nearest Neighbors
- Support Vector Machine
- Logistic Regression


## Usage Example

```python
from predict_delirium import DeliriumPredictor

# Load model and predict
predictor = DeliriumPredictor(model_path='models/XGBoost_model.pkl')
predictor.analyze_file('patient_ecg.csv', output_dir='results/')
```

## System Requirements

- Python 3.8+
- 8GB RAM minimum
- GPU (optional, for training acceleration)

## Troubleshooting

**Model not found:**
- Train models first using `ecg_delirium_predictor.py`

**GPU not working:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**SHAP not available:**
```bash
pip install shap
```



## Contact

For questions or collaboration: [hallokk77@gmail.com]

