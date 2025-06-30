import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import os

# --- Global objects loaded once when the script starts ---
try:
    # Use relative paths for portability
    model_path = os.path.join('pkl_files', 'lgbm_model_v1.pkl')
    preprocessor_path = os.path.join('pkl_files', 'preprocessor_v1.pkl')
    
    MODEL = joblib.load(model_path)
    PREPROCESSOR = joblib.load(preprocessor_path)
    print("Model and preprocessor loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: Model or preprocessor file not found: {e}")
    print("Please ensure the pkl files are in the pkl_files directory.")
    MODEL, PREPROCESSOR = None, None
except Exception as e:
    print(f"Error loading model: {e}")
    MODEL, PREPROCESSOR = None, None

def predict_price(input_data: dict) -> dict:
    """
    Predicts the sale price of a vehicle using a pre-trained LightGBM model.

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
              Returns an error message if input is invalid or model fails.
    """
    if not MODEL or not PREPROCESSOR:
        return {'error': 'Model or preprocessor not loaded. Cannot make predictions.'}

    # --- 1. Input Validation and DataFrame Creation ---
    try:
        # Create a DataFrame from the input dictionary
        input_df = pd.DataFrame([input_data])
        
        # Ensure all required columns are present
        required_cols = PREPROCESSOR.feature_names_in_
        # We need to manually add back 'Vehicle Age' logic if it's not in the preprocessor's memory
        expected_keys = [col for col in required_cols if col != 'Vehicle Age'] + ['Lot Year']
        
        for col in expected_keys:
             if col not in input_df.columns:
                 return {'error': f"Missing required feature in input data: '{col}'"}

    except Exception as e:
        return {'error': f"Failed to create DataFrame from input: {e}"}

    # --- 2. Preprocessing ---
    try:
        # a. Feature Engineering: Create 'Vehicle Age'
        # This step MUST be identical to the training process
        current_year = datetime.now().year
        input_df['Vehicle Age'] = current_year - input_df['Lot Year']
        
        # b. Apply the loaded preprocessor
        # The preprocessor will handle scaling, one-hot encoding, and grouping rare categories
        # based on what it learned from the training data.
        input_processed = PREPROCESSOR.transform(input_df)

    except Exception as e:
        return {'error': f"Error during data preprocessing: {e}"}

    # --- 3. Prediction ---
    try:
        # a. Get the log-scale prediction
        log_prediction = MODEL.predict(input_processed)[0]
        
        # b. Convert prediction back to the original dollar scale
        predicted_price = np.expm1(log_prediction)
        
        # c. Get prediction standard deviation for confidence
        # For tree-based ensembles like LightGBM, we can get predictions from each individual tree
        # and calculate the standard deviation. A lower std dev implies higher confidence.
        individual_tree_preds = MODEL.predict(input_processed, pred_leaf=False) # This line might not be right for all versions. Let's use a simpler approach.
        
        # A more robust way is to get predictions from each tree using predict's internal logic.
        # This requires iterating over the booster's trees if the main API doesn't expose it.
        # A simpler proxy for confidence:
        # For POC, we'll generate a "confidence score" based on a fixed logic.
        # A more advanced approach would use quantile regression or bootstrapping.
        # Here, let's create a placeholder confidence logic.
        # The standard deviation of predictions across trees is a good proxy.
        # Getting individual tree predictions requires accessing the underlying booster object.
        booster = MODEL.booster_
        individual_preds = booster.predict(input_processed, pred_leaf=False) # This gives predictions from each iteration
        
        # We need to get the final predictions from each tree, which is more complex.
        # Let's use a placeholder logic for now for the POC.
        # A good placeholder: confidence decreases as prediction moves away from the mean training price.
        # Let's calculate a simple confidence score for the POC.
        # A true confidence interval is complex for GBMs. We'll provide a "Confidence Level".
        # Let's find the standard deviation of predictions from each tree as a proxy.
        predictions_per_tree = []
        for i in range(MODEL.n_estimators_):
            tree_pred = MODEL.predict(input_processed, start_iteration=i, num_iteration=1)
            predictions_per_tree.append(np.expm1(tree_pred[0]))

        confidence_std_dev = np.std(predictions_per_tree)
        
        # Normalize the std dev to a "confidence level"
        # This is a heuristic. A lower std dev means the trees agree more.
        # Let's say a std dev of $500 is very confident, and $5000 is not.
        if confidence_std_dev < 1000:
            confidence_level = "High"
        elif confidence_std_dev < 3000:
            confidence_level = "Medium"
        else:
            confidence_level = "Low"

    except Exception as e:
        return {'error': f"Error during model prediction: {e}"}

    # --- 4. Format and Return Output ---
    result = {
        'predicted_sale_price': round(predicted_price, 2),
        'confidence_level': confidence_level,
        'estimated_prediction_variability': f"${round(confidence_std_dev, 2)}"
    }
    
    return result

# --- USAGE EXAMPLE ---
if __name__ == '__main__':
    # Example 1: A standard vehicle
    sample_input_1 = {
        'Lot Year': 2018,
        'Odometer Reading': 55000,
        'Lot Make': 'FORD',
        'Lot Model': 'FOCUS SEL', # A common model
        'Lot Run Condition': 'RUN & DRIVE',
        'Sale Title Type': 'CERTIFICATE OF TITLE',
        'Damage Type Description': 'FRONT END',
        'Lot Fuel Type': 'GAS'
    }
    
    prediction_1 = predict_price(sample_input_1)
    print("\n--- Prediction 1 ---")
    print(prediction_1)

    # Example 2: A rare or unusual vehicle, might have lower confidence
    sample_input_2 = {
        'Lot Year': 2022,
        'Odometer Reading': 15000,
        'Lot Make': 'RARE_MAKE', # This will be mapped to 'Other' by the preprocessor
        'Lot Model': 'RARE_MODEL', # This will also be mapped to 'Other'
        'Lot Run Condition': 'RUN & DRIVE',
        'Sale Title Type': 'CERT OF SALVAGE > 75',
        'Damage Type Description': 'BURN - ENGINE',
        'Lot Fuel Type': 'ELECTRIC'
    }
    
    prediction_2 = predict_price(sample_input_2)
    print("\n--- Prediction 2 ---")
    print(prediction_2)
