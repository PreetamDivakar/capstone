"""
Vercel serverless function wrapper for Flask app
"""
import sys
import os

# Get the directory where this file is located (api/)
api_dir = os.path.dirname(os.path.abspath(__file__))
# Get the project root (one level up from api/)
project_root = os.path.dirname(api_dir)

# Add the project root to the Python path so we can import backend.app
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the Flask app
from backend.app import app

# Export the app for Vercel
# Vercel's Python runtime automatically detects Flask apps
# Just export the app variable
__all__ = ['app']

