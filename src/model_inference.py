import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb

# Import the same helper functions and config used during training
from config import BRAND_TIER_MAP

# --- Global objects loaded once when the script starts ---
MODEL = None
PIPELINE_PATH = 'best_vehicle_price_pipeline_final.pkl'
MAX_LOG_PREDICTION = 15.0 

def load_model_simple():
    """Load model with fallback to simple prediction"""
    global PIPELINE, MODEL
    import joblib
# --- Global objects loaded once when the script starts ---
    try:
        PIPELINE = joblib.load(PIPELINE_PATH)
        print(f"✅ Model pipeline loaded successfully from {PIPELINE_PATH}")
        print(f"    Best model type: {type(PIPELINE.named_steps['regressor']).__name__}")
    except FileNotFoundError:
        print(f"❌ FATAL ERROR: Model pipeline file not found at '{PIPELINE_PATH}'.")
        print("    Please run the training script first to generate the artifact.")
        PIPELINE = None

def simple_price_prediction(input_data: dict) -> dict:
    """
    Simple price prediction based on historical data patterns.
    Used as fallback when ML model is not available.
    """
    try:
        # Load data for simple prediction
        data = pd.read_csv('data.csv')
        
        # Filter data based on input criteria
        filtered_data = data.copy()
        
        # Filter by year (within 2 years)
        year = input_data.get('Lot Year', 2020)
        filtered_data = filtered_data[
            (filtered_data['Lot Year'] >= year - 2) & 
            (filtered_data['Lot Year'] <= year + 2)
        ]
        
        # Filter by make if available
        if 'Lot Make' in input_data and input_data['Lot Make']:
            filtered_data = filtered_data[
                filtered_data['Lot Make'].str.contains(input_data['Lot Make'], case=False, na=False)
            ]
        
        # Filter by odometer (within 20% range)
        odometer = input_data.get('Odometer Reading', 50000)
        filtered_data = filtered_data[
            (filtered_data['Odometer Reading'] >= odometer * 0.8) & 
            (filtered_data['Odometer Reading'] <= odometer * 1.2)
        ]
        
        # If we have enough data, calculate prediction
        if len(filtered_data) >= 5:
            # Remove outliers using IQR method
            Q1 = filtered_data['Sale Price'].quantile(0.25)
            Q3 = filtered_data['Sale Price'].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            
            clean_data = filtered_data[
                (filtered_data['Sale Price'] >= lower) & 
                (filtered_data['Sale Price'] <= upper)
            ]
            
            if len(clean_data) > 0:
                predicted_price = float(clean_data['Sale Price'].median())
                confidence_std = float(clean_data['Sale Price'].std())
                
                # Determine confidence level
                if confidence_std < 2000:
                    confidence_level = "High"
                elif confidence_std < 5000:
                    confidence_level = "Medium"
                else:
                    confidence_level = "Low"
                
                return {
                    'predicted_sale_price': round(predicted_price, 2),
                    'confidence_level': confidence_level,
                    'estimated_prediction_variability': f"${round(confidence_std, 2)}",
                    'method': 'simple_statistical',
                    'data_points_used': len(clean_data)
                }
        
        # Fallback to overall median if not enough filtered data
        overall_median = float(data['Sale Price'].median())
        overall_std = float(data['Sale Price'].std())
        
        return {
            'predicted_sale_price': round(overall_median, 2),
            'confidence_level': "Low",
            'estimated_prediction_variability': f"${round(overall_std, 2)}",
            'method': 'overall_median',
            'data_points_used': len(data)
        }
        
    except Exception as e:
        return {
            'error': f"Simple prediction failed: {e}",
            'predicted_sale_price': 15000.0,
            'confidence_level': "Low",
            'estimated_prediction_variability': "$5000.00",
            'method': 'fallback'
        }


def get_automobile_type(model_name: str) -> str:
    """Classifies a vehicle into a general type based on keywords."""
    if not isinstance(model_name, str): return 'UNKNOWN'
    model_lower = model_name.lower()
    pickup_keys = ['f-150', 'f150', 'silverado', 'sierra', 'ram', 'tundra', 'tacoma', 'titan', 'ranger', 'colorado']
    suv_keys = ['suv', 'explorer', 'cherokee', 'wrangler', 'durango', '4runner', 'highlander', 'rav4', 'tahoe', 'suburban', 'escalade']
    van_keys = ['van', 'sienna', 'odyssey', 'pacifica', 'transit', 'sprinter']
    if any(key in model_lower for key in pickup_keys): return 'PICKUP'
    if any(key in model_lower for key in suv_keys): return 'SUV'
    if any(key in model_lower for key in van_keys): return 'VAN_MINIVAN'
    return 'SEDAN_COUPE_HATCH'


# --- Main Inference Function ---
def predict_price(input_data: dict) -> dict:
    """
    Predicts the sale price of a vehicle using the pre-trained model pipeline.

    This function replicates the feature engineering steps from training and then
    uses the loaded pipeline to preprocess and predict.

    Args:
        input_data (dict): A dictionary containing the vehicle's features.

    Returns:
        dict: A dictionary containing the prediction and associated metadata.
    """
    if not PIPELINE:
        return {'error': 'Model pipeline is not loaded. Cannot make predictions.'}

    try:
        # 1. Create a DataFrame from the input dictionary
        input_df = pd.DataFrame([input_data])

        # 2. Replicate Feature Engineering - This MUST match the training script
        # a. Clean string inputs
        for col in input_df.select_dtypes(include=['object']).columns:
            input_df[col] = input_df[col].str.strip()

        # b. Create 'Vehicle Age'
        input_df['Vehicle Age'] = datetime.now().year - input_df['Lot Year']

        # c. Create 'Brand_Tier'
        #    AT INFERENCE, we CANNOT use price-based heuristics. We rely on our static map.
        #    Any new brand will get a 'Standard' default, which is a safe assumption.
        input_df['Brand_Tier'] = input_df['Lot Make'].apply(
            lambda make: BRAND_TIER_MAP.get(make, 'Standard')
        )

        # d. Create 'Automobile_Type'
        input_df['Automobile_Type'] = input_df['Lot Model'].apply(get_automobile_type)

        # The pipeline's ColumnTransformer will select the correct columns from here.
        # No need to manually drop columns.

        # 3. Prediction
        # The pipeline handles preprocessing and prediction in one step.
        log_prediction = PIPELINE.predict(input_df)[0]

        # 4. Post-processing
        # a. Apply clipping as a safety net against overflow
        log_prediction_clipped = np.clip(log_prediction, a_min=None, a_max=MAX_LOG_PREDICTION)

        # b. Convert prediction back to the original dollar scale
        predicted_price = np.expm1(log_prediction_clipped)

        # c. Calculate Confidence Level (only for tree-based models)
        regressor = PIPELINE.named_steps['regressor']
        if hasattr(regressor, 'feature_importances_'):
            # This is a tree-based model. We can estimate confidence.
            # We must transform the data first before passing it to the regressor
            processed_input = PIPELINE.named_steps['preprocessor'].transform(input_df)

            predictions_per_tree = []
            if isinstance(regressor, (lgb.LGBMRegressor, xgb.XGBRegressor)):
                # For LightGBM/XGBoost, we can get predictions from each iteration
                for i in range(regressor.n_estimators):
                    tree_pred = regressor.predict(processed_input, iteration_range=(i, i + 1))
                    predictions_per_tree.append(np.expm1(tree_pred[0]))
            elif isinstance(regressor, RandomForestRegressor):
                 # For RandomForest, we get predictions from each tree in the forest
                for tree in regressor.estimators_:
                    tree_pred = tree.predict(processed_input)
                    predictions_per_tree.append(np.expm1(tree_pred[0]))

            confidence_std_dev = np.std(predictions_per_tree)

            if confidence_std_dev < 1000:
                confidence_level = "High"
            elif confidence_std_dev < 5000:
                confidence_level = "Medium"
            else:
                confidence_level = "Low"
        else:
            # Linear models don't have an equivalent simple confidence measure
            confidence_level = "N/A (Linear Model)"
            confidence_std_dev = None

    except Exception as e:
        return {'error': f"An error occurred during prediction: {e}"}

    # 5. Format and Return Output
    result = {
        'predicted_sale_price': round(predicted_price, 2),
        'confidence_level': confidence_level,
        'metadata': {
            'model_used': type(PIPELINE.named_steps['regressor']).__name__,
            'brand_tier_assigned': input_df['Brand_Tier'].iloc[0],
            'automobile_type_assigned': input_df['Automobile_Type'].iloc[0],
            'prediction_variability_est': f"${round(confidence_std_dev, 2)}" if confidence_std_dev is not None else "N/A"
        }
    }

    return result

# --- USAGE EXAMPLE ---
if __name__ == '__main__':
    # Example 1: A standard vehicle
    sample_input_1 = {
        'Lot Year': 2019, 'Odometer Reading': 60000, 'Lot Make': 'TOYT', 
        'Lot Model': 'CAMRY SE', 'Lot Run Condition': 'RUN & DRIVE', 
        'Sale Title Type': 'CERTIFICATE OF TITLE', 'Damage Type Description': 'MINOR DENT/SCRATCHES', 
        'Lot Fuel Type': 'GAS'
    }
    prediction_1 = predict_price(sample_input_1)
    print("\n--- Prediction 1 (Standard Car) ---")
    print(prediction_1)

    # Example 2: A known luxury/exotic car
    sample_input_2 = {
        'Lot Year': 2021, 'Odometer Reading': 15000, 'Lot Make': 'PORS', 
        'Lot Model': '911 TURBO', 'Lot Run Condition': 'RUN & DRIVE', 
        'Sale Title Type': 'CERTIFICATE OF TITLE', 'Damage Type Description': 'None', 
        'Lot Fuel Type': 'GAS'
    }
    prediction_2 = predict_price(sample_input_2)
    print("\n--- Prediction 2 (Luxury Car) ---")
    print(prediction_2)

    # Example 3: A completely new, unseen brand at inference time
    # The system should gracefully handle this by assigning 'Standard' tier.
    sample_input_3 = {
        'Lot Year': 2020, 'Odometer Reading': 8000, 'Lot Make': 'GENESIS', # Assuming GENESIS is not in our static map
        'Lot Model': 'G70', 'Lot Run Condition': 'RUN & DRIVE', 
        'Sale Title Type': 'CERTIFICATE OF TITLE', 'Damage Type Description': 'None', 
        'Lot Fuel Type': 'GAS'
    }
    prediction_3 = predict_price(sample_input_3)
    print("\n--- Prediction 3 (Unknown Brand at Inference) ---")
    print(prediction_3)

    # Example 4: A pickup truck to test the Automobile_Type feature
    sample_input_4 = {
        'Lot Year': 2018, 'Odometer Reading': 85000, 'Lot Make': 'FORD',
        'Lot Model': 'F-150 SUPERCREW', 'Lot Run Condition': 'RUN & DRIVE',
        'Sale Title Type': 'CERTIFICATE OF TITLE', 'Damage Type Description': 'REAR END',
        'Lot Fuel Type': 'GAS'
    }
    prediction_4 = predict_price(sample_input_4)
    print("\n--- Prediction 4 (Pickup Truck) ---")
    print(prediction_4)