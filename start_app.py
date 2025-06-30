#!/usr/bin/env python3
"""
Startup script for the Vehicle Price Estimator application.
This script can start both the FastAPI server and Streamlit app.
"""

import subprocess
import sys
import time
import os
import signal
import threading

def start_fastapi_server():
    """Start the FastAPI server"""
    print("Starting FastAPI server...")
    try:
        process = subprocess.Popen([sys.executable, "src/app.py"], 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE)
        time.sleep(3)  # Wait for server to start
        if process.poll() is None:
            print("✅ FastAPI server started successfully on http://localhost:8000")
            return process
        else:
            print("❌ Failed to start FastAPI server")
            return None
    except Exception as e:
        print(f"❌ Error starting FastAPI server: {e}")
        return None

def start_streamlit_app():
    """Start the Streamlit app"""
    print("Starting Streamlit app...")
    try:
        process = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"], 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE)
        time.sleep(5)  # Wait for Streamlit to start
        if process.poll() is None:
            print("✅ Streamlit app started successfully on http://localhost:8501")
            return process
        else:
            print("❌ Failed to start Streamlit app")
            return None
    except Exception as e:
        print(f"❌ Error starting Streamlit app: {e}")
        return None

def main():
    print("🚗 Vehicle Price Estimator - Startup Script")
    print("=" * 50)
    
    # Check if required files exist
    required_files = ["src/app.py", "streamlit_app.py", "data.csv"]
    required_dirs = ["pkl_files"]
    
    for file in required_files:
        if not os.path.exists(file):
            print(f"❌ Required file not found: {file}")
            return
    
    for dir in required_dirs:
        if not os.path.exists(dir):
            print(f"❌ Required directory not found: {dir}")
            return
    
    print("✅ All required files found")
    
    # Start FastAPI server
    fastapi_process = start_fastapi_server()
    if not fastapi_process:
        print("Cannot continue without FastAPI server")
        return
    
    # Start Streamlit app
    streamlit_process = start_streamlit_app()
    if not streamlit_process:
        print("Cannot continue without Streamlit app")
        fastapi_process.terminate()
        return
    
    print("\n🎉 Both services started successfully!")
    print("📱 Streamlit app: http://localhost:8501")
    print("🔌 FastAPI server: http://localhost:8000")
    print("\nPress Ctrl+C to stop both services...")
    
    try:
        # Keep the script running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping services...")
        fastapi_process.terminate()
        streamlit_process.terminate()
        print("✅ Services stopped")

if __name__ == "__main__":
    main() 