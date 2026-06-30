import os
import sys
import streamlit.web.cli as stcli

# 1. Register the project root directory in Python search paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 2. Enforce Streamlit to internally boot-up our target application file
if __name__ == "__main__":
    sys.argv = ["streamlit", "run", "src/ui/app.py"]
    sys.exit(stcli.main())