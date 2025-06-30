import pandas as pd
import numpy as np
from datetime import datetime
import os
# --- Global objects loaded once when the script starts ---
MODEL = None
PREPROCESSOR = None

def load_model_simple():
    """Load model with fallback to simple prediction"""
    global MODEL, PREPROCESSOR
    
    try:
        import joblib
        # Use relative paths for portability
        model_path = os.path.join('pkl_files', 'lgbm_model_v1.pkl')
        preprocessor_path = os.path.join('pkl_files', 'preprocessor_v1.pkl')
        
        MODEL = joblib.load(model_path)
        PREPROCESSOR = joblib.load(preprocessor_path)
        print("Model and preprocessor loaded successfully.")
        return True
    except Exception as e:
        print(f"Warning: Could not load ML model ({e}). Using simple prediction method.")
        MODEL, PREPROCESSOR = None, None
        return False

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

def predict_price(input_data: dict) -> dict:
    """
    Predicts the sale price of a vehicle using either ML model or simple statistical method.
    
    Args:
        input_data (dict): A dictionary containing the vehicle's features.
                           Keys must match the column names used during training.
                           Example:
                           {
                               'Lot Year': 2018,
                               'Odometer Reading': 50000,
                               'Lot Make': 'FORD',
                               'Lot Model': 'FUSION',
                               'Lot Run Condition': 'RUN & DRIVE',
                               'Sale Title Type': 'CERTIFICATE OF TITLE',
                               'Damage Type Description': 'FRONT END',
                               'Lot Fuel Type': 'GAS'
                           }

    Returns:
        dict: A dictionary containing the predicted price and a confidence measure.
    """
    # Try to load model if not already loaded
    if MODEL is None and PREPROCESSOR is None:
        load_model_simple()
    
    # If ML model is available, use it
    if MODEL is not None and PREPROCESSOR is not None:
        try:
            # --- 1. Input Validation and DataFrame Creation ---
            input_df = pd.DataFrame([input_data])
            
            # Ensure all required columns are present
            required_cols = PREPROCESSOR.feature_names_in_
            expected_keys = [col for col in required_cols if col != 'Vehicle Age'] + ['Lot Year']
            
            for col in expected_keys:
                 if col not in input_df.columns:
                     return {'error': f"Missing required feature in input data: '{col}'"}

            # --- 2. Preprocessing ---
            # a. Feature Engineering: Create 'Vehicle Age'
            current_year = datetime.now().year
            input_df['Vehicle Age'] = current_year - input_df['Lot Year']
            
            # b. Apply the loaded preprocessor
            input_processed = PREPROCESSOR.transform(input_df)

            # --- 3. Prediction ---
            # a. Get the log-scale prediction
            log_prediction = MODEL.predict(input_processed)[0]
            
            # b. Convert prediction back to the original dollar scale
            predicted_price = np.expm1(log_prediction)
            
            # c. Get prediction confidence
            predictions_per_tree = []
            for i in range(MODEL.n_estimators_):
                tree_pred = MODEL.predict(input_processed, start_iteration=i, num_iteration=1)
                predictions_per_tree.append(np.expm1(tree_pred[0]))

            confidence_std_dev = np.std(predictions_per_tree)
            
            # Normalize the std dev to a "confidence level"
            if confidence_std_dev < 1000:
                confidence_level = "High"
            elif confidence_std_dev < 3000:
                confidence_level = "Medium"
            else:
                confidence_level = "Low"

            # --- 4. Format and Return Output ---
            result = {
                'predicted_sale_price': round(predicted_price, 2),
                'confidence_level': confidence_level,
                'estimated_prediction_variability': f"${round(confidence_std_dev, 2)}",
                'method': 'ml_model'
            }
            
            return result
            
        except Exception as e:
            print(f"ML model prediction failed: {e}. Falling back to simple method.")
            return simple_price_prediction(input_data)
    
    else:
        # Use simple statistical method
        return simple_price_prediction(input_data)

# --- USAGE EXAMPLE ---
if __name__ == '__main__':
    # Example 1: A standard vehicle
    sample_input_1 = {
        'Lot Year': 2018,
        'Odometer Reading': 55000,
        'Lot Make': 'FORD',
        'Lot Model': 'FOCUS SEL',
        'Lot Run Condition': 'RUN & DRIVE',
        'Sale Title Type': 'CERTIFICATE OF TITLE',
        'Damage Type Description': 'FRONT END',
        'Lot Fuel Type': 'GAS'
    }
    
    prediction_1 = predict_price(sample_input_1)
    print("\n--- Prediction 1 ---")
    print(prediction_1)

    # Example 2: A rare or unusual vehicle
    sample_input_2 = {
        'Lot Year': 2022,
        'Odometer Reading': 15000,
        'Lot Make': 'TOYOTA',
        'Lot Model': 'CAMRY',
        'Lot Run Condition': 'RUN & DRIVE',
        'Sale Title Type': 'CERTIFICATE OF TITLE',
        'Damage Type Description': 'FRONT END',
        'Lot Fuel Type': 'GAS'
    }
    
    prediction_2 = predict_price(sample_input_2)
    print("\n--- Prediction 2 ---")
    print(prediction_2) 