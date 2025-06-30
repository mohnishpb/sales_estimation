#!/usr/bin/env python3
"""
Test script to verify the application components work correctly.
"""

import sys
import os
import pandas as pd
from model_inference_simple import predict_price

def test_model_inference():
    """Test the model inference functionality"""
    print("🧪 Testing Model Inference...")
    
    # Test input data
    test_input = {
        'Lot Year': 2018,
        'Odometer Reading': 55000,
        'Lot Make': 'FORD',
        'Lot Model': 'FOCUS SEL',
        'Lot Run Condition': 'RUN & DRIVE',
        'Sale Title Type': 'CERTIFICATE OF TITLE',
        'Damage Type Description': 'FRONT END',
        'Lot Fuel Type': 'GAS'
    }
    
    try:
        result = predict_price(test_input)
        print("✅ Model inference test passed!")
        predicted_price = result.get('predicted_sale_price', 'N/A')
        if isinstance(predicted_price, (int, float)):
            print(f"   Predicted Price: ${predicted_price:,.2f}")
        else:
            print(f"   Predicted Price: {predicted_price}")
        print(f"   Confidence Level: {result.get('confidence_level', 'N/A')}")
        print(f"   Variability: {result.get('estimated_prediction_variability', 'N/A')}")
        return True
    except Exception as e:
        print(f"❌ Model inference test failed: {e}")
        return False

def test_data_loading():
    """Test data loading functionality"""
    print("\n📊 Testing Data Loading...")
    
    try:
        data = pd.read_csv('data.csv')
        print(f"✅ Data loading test passed!")
        print(f"   Total records: {len(data):,}")
        print(f"   Columns: {list(data.columns)}")
        
        # Test categorical columns
        categorical_cols = ['Lot Make', 'Lot Model', 'Lot Run Condition', 
                          'Sale Title Type', 'Damage Type Description', 'Lot Fuel Type']
        
        for col in categorical_cols:
            if col in data.columns:
                unique_count = data[col].nunique()
                print(f"   {col}: {unique_count} unique values")
        
        return True
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        return False

def test_file_structure():
    """Test if all required files exist"""
    print("\n📁 Testing File Structure...")
    
    required_files = [
        'app.py',
        'streamlit_app.py', 
        'model_inference.py',
        'data.csv',
        'requirements.txt',
        'README.md'
    ]
    
    required_dirs = ['pkl_files']
    
    all_good = True
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - Missing")
            all_good = False
    
    for dir in required_dirs:
        if os.path.exists(dir):
            print(f"✅ {dir}/")
            # Check pkl files
            if dir == 'pkl_files':
                pkl_files = ['lgbm_model_v1.pkl', 'preprocessor_v1.pkl']
                for pkl in pkl_files:
                    pkl_path = os.path.join(dir, pkl)
                    if os.path.exists(pkl_path):
                        print(f"   ✅ {pkl}")
                    else:
                        print(f"   ❌ {pkl} - Missing")
                        all_good = False
        else:
            print(f"❌ {dir}/ - Missing")
            all_good = False
    
    return all_good

def main():
    print("🚗 Vehicle Price Estimator - Component Tests")
    print("=" * 50)
    
    # Test file structure
    file_structure_ok = test_file_structure()
    
    if not file_structure_ok:
        print("\n❌ File structure test failed. Please ensure all required files are present.")
        return
    
    # Test data loading
    data_loading_ok = test_data_loading()
    
    # Test model inference
    model_inference_ok = test_model_inference()
    
    print("\n" + "=" * 50)
    print("📋 Test Summary:")
    print(f"   File Structure: {'✅ PASS' if file_structure_ok else '❌ FAIL'}")
    print(f"   Data Loading: {'✅ PASS' if data_loading_ok else '❌ FAIL'}")
    print(f"   Model Inference: {'✅ PASS' if model_inference_ok else '❌ FAIL'}")
    
    if all([file_structure_ok, data_loading_ok, model_inference_ok]):
        print("\n🎉 All tests passed! The application should work correctly.")
        print("\n📝 To run the application:")
        print("   1. Start FastAPI server: python app.py")
        print("   2. Start Streamlit app: streamlit run streamlit_app.py")
        print("   3. Or use the startup script: python start_app.py")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main() 