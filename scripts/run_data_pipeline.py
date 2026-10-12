"""
TALUS - one-command data pipeline
=================================
Runs every data step in order:
  1. generate_data_and_train.py   (only if the clean master CSV is missing)
  2. generate_images.py           (only if no images exist)
  3. generate_raw_data.py         (messy slope export + rock/soil properties)
  4. preprocessing.py             (clean everything -> data/processed/)
  5. build_eda_notebook.py        (re-executes notebooks/01_preprocessing_eda.ipynb)  [--notebook]

Usage:  python scripts/run_data_pipeline.py            (steps 1-4)
        python scripts/run_data_pipeline.py --notebook (steps 1-5)
"""
import os
import sys
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SC = os.path.join(ROOT, "scripts")
run = lambda script, *a: subprocess.run([sys.executable, os.path.join(SC, script), *a], check=True, cwd=ROOT)

if not os.path.exists(os.path.join(ROOT, "data", "raw", "slope_monitoring_synthetic.csv")):
    print(">> clean master CSV missing - running generate_data_and_train.py"); run("generate_data_and_train.py")
if not os.path.isdir(os.path.join(ROOT, "data", "raw", "images")):
    print(">> generating images"); run("generate_images.py")
print(">> generating raw field data"); run("generate_raw_data.py")
print(">> preprocessing"); run("preprocessing.py")
if "--notebook" in sys.argv:
    print(">> rebuilding EDA notebook"); run("build_eda_notebook.py")
print("\nDone. Processed files are in data/processed/ ; start the app with run_backend.py and run_frontend.py")
