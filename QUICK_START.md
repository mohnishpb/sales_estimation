# 🚗 Vehicle Price Estimator - Quick Start Guide

## Overview
This Streamlit application provides two methods for estimating vehicle prices:
1. **VIN-Based Prediction**: Search for similar vehicles by VIN and estimate price
2. **Manual Input Prediction**: Enter vehicle details manually for ML model predictions

## Quick Start

### Option 1: Use the Startup Script (Recommended)
```bash
python start_app.py
```
This will automatically start both the FastAPI server and Streamlit app.

### Option 2: Manual Start
1. **Start FastAPI Server** (Terminal 1):
   ```bash
   python app.py
   ```
   Server will run on: http://localhost:8000

2. **Start Streamlit App** (Terminal 2):
   ```bash
   streamlit run streamlit_app.py
   ```
   App will run on: http://localhost:8501

## Features

### VIN-Based Prediction
- Enter a VIN number to find similar vehicles
- View detailed information about similar vehicles
- Get price estimates based on historical data
- See price distribution charts

### Manual Input Prediction
- Input vehicle details through dropdown menus
- Get ML model predictions with confidence levels
- View prediction variability and confidence metrics
- Dropdown values populated from actual data

## File Structure
```
sales_estimation/
├── app.py                    # FastAPI server
├── streamlit_app.py          # Streamlit application
├── model_inference_simple.py # ML model (simplified version)
├── data.csv                  # Vehicle data (45,723 records)
├── pkl_files/               # Model files
├── requirements.txt          # Dependencies
├── start_app.py             # Startup script
└── test_app.py              # Test script
```

## API Endpoints
- `GET /estimate_price/{vin}` - Get similar vehicles by VIN
- `POST /estimate_price/` - Predict price based on vehicle data

## Notes
- The application uses a simplified prediction method that works without LightGBM
- All dropdown values are populated from the actual data.csv file
- The system provides confidence levels (High/Medium/Low) based on prediction variability
- Both VIN-based and manual input methods are available

## Troubleshooting
If you encounter issues:
1. Run `python test_app.py` to verify all components work
2. Ensure all files are in the correct directory structure
3. Check that ports 8000 and 8501 are available 