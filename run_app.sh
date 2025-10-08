#!/bin/bash

# Health Insurance Cost Predictor - Startup Script

echo "🏥 Health Insurance Cost Predictor"
echo "=================================="
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run the setup first:"
    echo "  python -m venv .venv"
    echo "  source .venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment and run app
echo "🚀 Starting Streamlit application..."
echo "📝 Open your browser to: http://localhost:8501"
echo "🛑 Press Ctrl+C to stop the application"
echo ""

source .venv/bin/activate
streamlit run app.py --server.port 8501 --server.address localhost