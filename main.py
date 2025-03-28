import os
import subprocess
import sys

# Use the Python interpreter from the active environment
python_executable = sys.executable

# List of all subscript
scripts = [
    "src/train_lr.py",
    "src/train_rf.py",
    "src/predict.py",
    "src/evaluate.py"
]

# Run regular scripts
for script in scripts:
    print(f"Running {script}...")
    result = subprocess.run([python_executable, script], check=True)
    if result.returncode != 0:
        print(f"Error running {script}")
        sys.exit(1)

# Run the Streamlit app separately
streamlit_script = "src/estimate_price.py"
print(f"Running Streamlit app: {streamlit_script}...")
try:
    subprocess.run([python_executable, "-m", "streamlit", "run", streamlit_script], check=True)
except subprocess.CalledProcessError as e:
    print(f"Error running Streamlit: {e}")
    sys.exit(1)

print("Pipeline executed successfully!")