"""
00_check_environment.py
-----------------------
Verify that all required packages are installed and working.
Saves output to results/environment_check.txt
"""

import sys
import os
import datetime

def check_environment():
    results = []
    results.append(f"Environment Check — {datetime.datetime.now().isoformat()}")
    results.append(f"Python: {sys.version}")
    results.append(f"Platform: {sys.platform}")

    # PyTorch
    try:
        import torch
        results.append(f"PyTorch: {torch.__version__}")
        results.append(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            results.append(f"CUDA version: {torch.version.cuda}")
            results.append(f"GPU: {torch.cuda.get_device_name(0)}")
            results.append(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            results.append("WARNING: CUDA not available — training will use CPU")
    except ImportError:
        results.append("ERROR: PyTorch not installed")

    # OpenCV
    try:
        import cv2
        results.append(f"OpenCV: {cv2.__version__}")
    except ImportError:
        results.append("ERROR: OpenCV not installed")

    # NumPy
    try:
        import numpy as np
        results.append(f"NumPy: {np.__version__}")
    except ImportError:
        results.append("ERROR: NumPy not installed")

    # Matplotlib
    try:
        import matplotlib
        results.append(f"Matplotlib: {matplotlib.__version__}")
    except ImportError:
        results.append("ERROR: Matplotlib not installed")

    # Pandas
    try:
        import pandas as pd
        results.append(f"Pandas: {pd.__version__}")
    except ImportError:
        results.append("ERROR: Pandas not installed")

    # tqdm
    try:
        import tqdm
        results.append(f"tqdm: {tqdm.__version__}")
    except ImportError:
        results.append("ERROR: tqdm not installed")

    # scikit-learn
    try:
        import sklearn
        results.append(f"scikit-learn: {sklearn.__version__}")
    except ImportError:
        results.append("ERROR: scikit-learn not installed")

    # Print and save
    output = "\n".join(results)
    print(output)

    # Save to results
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "environment_check.txt")
    with open(out_path, "w") as f:
        f.write(output + "\n")
    print(f"\nSaved to: {out_path}")

if __name__ == "__main__":
    check_environment()
