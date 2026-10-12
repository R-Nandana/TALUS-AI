import os, sys, subprocess
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
app_path = os.path.join(ROOT_DIR, "frontend", "app.py")
if __name__ == "__main__":
    port = os.getenv("PORT", "8501")
    subprocess.run([sys.executable, "-m", "streamlit", "run", app_path, "--server.address=0.0.0.0", f"--server.port={port}"])
