import sys
import platform
import sklearn
import tensorflow as tf
import shap
import numpy as np

print("="*50)
print("EXPERIMENTAL ENVIRONMENT INFO")
print("="*50)
print(f"Python version: {sys.version}")
print(f"OS: {platform.system()} {platform.release()}")
print(f"Processor: {platform.processor()}")
print(f"scikit-learn: {sklearn.__version__}")
print(f"TensorFlow: {tf.__version__}")
print(f"SHAP: {shap.__version__}")
print(f"NumPy: {np.__version__}")
print("="*50)