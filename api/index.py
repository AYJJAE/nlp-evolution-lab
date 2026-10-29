"""
Vercel Serverless Entrypoint for NLP Evolution Lab
Exports WSGI 'app' for Vercel Python Runtime.
"""

import sys
import os

# Ensure the root directory is on the Python module search path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from server import app

# Vercel's @vercel/python looks for 'app'
if __name__ == "__main__":
    app.run()

