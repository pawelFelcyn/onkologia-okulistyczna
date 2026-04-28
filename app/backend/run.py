import subprocess
import sys


subprocess.run([sys.executable, "-m", "uvicorn", "main:app"], check=True)
