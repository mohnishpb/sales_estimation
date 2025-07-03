# Vehicle Price Estimator - Streamlit Application

This Streamlit application provides two methods for estimating vehicle prices:
1. **VIN-Based Prediction**: Enter a VIN number to find similar vehicles and estimate price
2. **Manual Input Prediction**: Enter vehicle details manually to get ML model predictions

## Features

- **VIN-Based Prediction**: 
  - Search for similar vehicles by VIN
  - Display similar vehicles with their details
  - Calculate price estimates based on similar vehicles
  - Show price distribution charts

- **Manual Input Prediction**:
  - Input vehicle details through dropdown menus
  - Get ML model predictions with confidence levels
  - View prediction variability and confidence metrics

## Setup Instructions

### 1. Create an environment and install dependencies

```bash
conda create -n myenv python=3.10
conda activate myenv
cd sales_estimation
pip install -r requirements.txt
```

### 2. Start the FastAPI Server
First, start the FastAPI server that provides the APIs:
```bash
python src/app.py
```
This will start the server on `http://localhost:8000`

### 3. Run the Streamlit Application
In a new terminal, run:
```bash
streamlit run streamlit_app.py
```
This will start the Streamlit app on `http://localhost:8501`

## File Structure

```
sales_estimation/
├── app.py                 # FastAPI server with VIN and prediction APIs
├── model_inference.py     # ML model inference logic
├── streamlit_app.py       # Streamlit application
├── requirements.txt       # Python dependencies
├── data.csv              # Vehicle data
├── pkl_files/
│   ├── lgbm_model_v1.pkl     # Trained LightGBM model
│   └── preprocessor_v1.pkl   # Data preprocessor
└── README.md             # This file
```

## API Endpoints

The FastAPI server provides two endpoints:

1. `GET /estimate_price/{vin}` - Get similar vehicles by VIN
2. `POST /estimate_price/` - Predict price based on vehicle data

## Usage

### VIN-Based Prediction
1. Navigate to "VIN-Based Prediction" in the sidebar
2. Enter a VIN number (e.g., 1HGBH41JXMN109186)
3. Click "Search Similar Vehicles" to find matches
4. Click "Predict Price" to get price estimation

### Manual Input Prediction
1. Navigate to "Manual Input Prediction" in the sidebar
2. Fill in the vehicle details using the dropdown menus
3. Click "Predict Price" to get ML model prediction
4. View the predicted price, confidence level, and variability

## Notes

- The application requires both the FastAPI server and Streamlit app to be running
- Make sure the `data.csv` file and `pkl_files` directory are in the same directory as the application
- The dropdown values are populated from the actual data in `data.csv`
- The ML model provides confidence levels (High/Medium/Low) based on prediction variability 