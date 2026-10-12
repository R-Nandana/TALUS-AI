import os, sys, uvicorn
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

if __name__ == "__main__":
    host = os.getenv("TALUS_HOST", "0.0.0.0")
    port = int(os.getenv("PORT", os.getenv("TALUS_PORT", "8000")))
    uvicorn.run("backend.main:app", host=host, port=port, reload=os.getenv("TALUS_RELOAD", "false").lower() == "true")
