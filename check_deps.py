import torch
import sys

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
else:
    print("Device: CPU")

try:
    import sklearn
    print(f"scikit-learn: {sklearn.__version__}")
except ImportError:
    print("scikit-learn: NOT INSTALLED")

try:
    import numpy
    print(f"numpy: {numpy.__version__}")
except ImportError:
    print("numpy: NOT INSTALLED")
