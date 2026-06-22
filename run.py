import os
import sys

# Forcefully append the current working directory to system path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Execute the actual UI application core directly
from src.ui import app