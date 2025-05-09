import subprocess
import sys
import os

# Get the Python executable from the current virtual environment
python_executable = sys.executable

scripts = [
    'src/eda.py',
    'src/train_gb.py',  # Full feature set
    'src/train_rf.py',  # Full feature set
    'src/predict.py',
    'src/evaluate.py',
    'src/visualize.py',
    'src/EDA_preprocess.py'
]

# Run each script
for script in scripts:
    if os.path.exists(script):
        print(f"Running {script}...")
        result = subprocess.run([python_executable, script], check=True)
    else:
        print(f"Script {script} not found!")

# Prompt to run Streamlit app
run_streamlit = input("Do you want to run the Streamlit app (estimate_price.py)? [y/n]: ").strip().lower()
if run_streamlit == 'y':
    streamlit_script = 'src/estimate_price.py'
    if os.path.exists(streamlit_script):
        print(f"Running {streamlit_script} with Streamlit...")
        result = subprocess.run([python_executable, '-m', 'streamlit', 'run', streamlit_script])
    else:
        print(f"Script {streamlit_script} not found!")