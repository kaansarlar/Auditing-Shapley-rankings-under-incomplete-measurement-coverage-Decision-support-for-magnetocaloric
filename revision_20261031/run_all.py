"""Regenerate the revision analyses and figures from included inputs, offline."""
from pathlib import Path
import subprocess,sys
root=Path(__file__).resolve().parent
for script in ['run_magnetic.py','run_acquisition.py','run_matminer.py','plot_magnetic.py','plot_matminer.py']:
    subprocess.run([sys.executable,str(root/'revision_code'/script)],check=True,cwd=root)
print('All revision analyses and figures completed.')
