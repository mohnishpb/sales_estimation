import streamlit as st
import pandas as pd
import requests
import json
from model_inference_simple import predict_price
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Vehicle Price Estimator",
    page_icon="🚗",
    layout="wide"
)

# Global variables
API_BASE_URL = "http://localhost:8000"

def load_preprocessor():
    """Load the preprocessor to get feature names and categories"""
    try:
        preprocessor = joblib.load('pkl_files/preprocessor_v1.pkl')
        return preprocessor
    except Exception as e:
        st.error(f"Error loading preprocessor: {e}")
        return None

def get_unique_values_from_data():
    """Get unique values for categorical features from the data.csv file"""
    try:
        data = pd.read_csv('data.csv')
        unique_values = {}
        
        # Get unique values for categorical columns
        categorical_cols = ['Lot Make', 'Lot Model', 'Lot Run Condition', 
                          'Sale Title Type', 'Damage Type Description', 'Lot Fuel Type']
        
        for col in categorical_cols:
            if col in data.columns:
                unique_values[col] = sorted(data[col].dropna().unique().tolist())
        
        return unique_values
    except Exception as e:
        st.error(f"Error loading data for dropdowns: {e}")
        return {}

def call_vin_api(vin):
    """Call the VIN-based API to get similar vehicles"""
    try:
        response = requests.get(f"{API_BASE_URL}/estimate_price/{vin}")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Error calling VIN API: {e}")
        return None

def call_prediction_api(vehicle_data):
    """Call the prediction API with vehicle data"""
    try:
        response = requests.post(f"{API_BASE_URL}/estimate_price/", json={"df": vehicle_data})
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Error calling prediction API: {e}")
        return None

def main():
    st.title("🚗 Vehicle Price Estimator")
    st.markdown("---")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Choose a page:",
        ["VIN-Based Prediction", "Manual Input Prediction"]
    )
    
    if page == "VIN-Based Prediction":
        vin_based_prediction()
    else:
        manual_input_prediction()

def vin_based_prediction():
    st.header("VIN-Based Price Prediction")
    st.markdown("Enter a VIN number to find similar vehicles and estimate the price.")
    
    # VIN input
    vin = st.text_input("Enter VIN Number:", placeholder="e.g., 1HGBH41JXMN109186")
    
    if st.button("Search Similar Vehicles", type="primary"):
        if vin:
            with st.spinner("Searching for similar vehicles..."):
                # Call the VIN API
                similar_vehicles = call_vin_api(vin)
                
                if similar_vehicles:
                    st.success(f"Found {len(similar_vehicles)} similar vehicles!")
                    
                    # Display the similar vehicles
                    st.subheader("Similar Vehicles Found:")
                    df_similar = pd.DataFrame(similar_vehicles)
                    st.dataframe(df_similar, use_container_width=True)
                    
                    # Show statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Number of Matches", len(similar_vehicles))
                    with col2:
                        if 'Sale Price' in df_similar.columns:
                            avg_price = df_similar['Sale Price'].mean()
                            st.metric("Average Price", f"${avg_price:,.2f}")
                    with col3:
                        if 'Sale Price' in df_similar.columns:
                            median_price = df_similar['Sale Price'].median()
                            st.metric("Median Price", f"${median_price:,.2f}")
                    
                    # Prediction button
                    if st.button("Predict Price", type="secondary"):
                        with st.spinner("Calculating price prediction..."):
                            # Call the prediction API
                            prediction_result = call_prediction_api(similar_vehicles)
                            
                            if prediction_result:
                                st.subheader("Price Prediction Results:")
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric(
                                        "Estimated Price", 
                                        f"${prediction_result['estimated_price']:,.2f}",
                                        delta=f"Based on {prediction_result['matches_found']} similar vehicles"
                                    )
                                with col2:
                                    st.metric("Matches Used", prediction_result['matches_found'])
                                
                                # Show price distribution
                                if 'Sale Price' in df_similar.columns:
                                    st.subheader("Price Distribution of Similar Vehicles")
                                    st.bar_chart(df_similar['Sale Price'])
        else:
            st.warning("Please enter a VIN number.")

def manual_input_prediction():
    st.header("Manual Input Price Prediction")
    st.markdown("Enter vehicle details manually to get a price prediction using our ML model.")
    
    # Load preprocessor and unique values
    preprocessor = load_preprocessor()
    unique_values = get_unique_values_from_data()
    
    if not preprocessor or not unique_values:
        st.error("Unable to load model components. Please check if the files are available.")
        return
    
    # Create form for manual input
    with st.form("manual_prediction_form"):
        st.subheader("Vehicle Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            lot_year = st.number_input("Lot Year", min_value=1900, max_value=2024, value=2020)
            odometer_reading = st.number_input("Odometer Reading", min_value=0, value=50000)
            
            # Categorical inputs with dropdowns
            lot_make = st.selectbox("Lot Make", unique_values.get('Lot Make', []))
            lot_model = st.selectbox("Lot Model", unique_values.get('Lot Model', []))
            lot_run_condition = st.selectbox("Lot Run Condition", unique_values.get('Lot Run Condition', []))
        
        with col2:
            sale_title_type = st.selectbox("Sale Title Type", unique_values.get('Sale Title Type', []))
            damage_type_description = st.selectbox("Damage Type Description", unique_values.get('Damage Type Description', []))
            lot_fuel_type = st.selectbox("Lot Fuel Type", unique_values.get('Lot Fuel Type', []))
        
        submitted = st.form_submit_button("Predict Price", type="primary")
        
        if submitted:
            # Prepare input data
            input_data = {
                'Lot Year': lot_year,
                'Odometer Reading': odometer_reading,
                'Lot Make': lot_make,
                'Lot Model': lot_model,
                'Lot Run Condition': lot_run_condition,
                'Sale Title Type': sale_title_type,
                'Damage Type Description': damage_type_description,
                'Lot Fuel Type': lot_fuel_type
            }
            
            with st.spinner("Making prediction..."):
                # Call the ML model
                prediction_result = predict_price(input_data)
                
                if 'error' in prediction_result:
                    st.error(f"Prediction Error: {prediction_result['error']}")
                else:
                    st.success("Prediction completed successfully!")
                    
                    # Display results
                    st.subheader("ML Model Prediction Results:")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "Predicted Sale Price", 
                            f"${prediction_result['predicted_sale_price']:,.2f}"
                        )
                    with col2:
                        confidence_color = {
                            "High": "green",
                            "Medium": "orange", 
                            "Low": "red"
                        }.get(prediction_result['confidence_level'], "gray")
                        
                        st.metric(
                            "Confidence Level", 
                            prediction_result['confidence_level'],
                            delta=prediction_result['estimated_prediction_variability']
                        )
                    with col3:
                        st.metric(
                            "Prediction Variability",
                            prediction_result['estimated_prediction_variability']
                        )
                    
                    # Show input summary
                    st.subheader("Input Summary")
                    input_df = pd.DataFrame([input_data])
                    st.dataframe(input_df, use_container_width=True)

if __name__ == "__main__":
    main() 