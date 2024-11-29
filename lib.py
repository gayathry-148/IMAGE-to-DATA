import subprocess
import sys

# List of libraries to install or upgrade
libraries = [
    "jax", "jaxlib", "optax", "numpy", "pandas", "flax", "tensorflow", "onnx", 
    "onnxruntime", "Pillow", "tqdm"
]

# Function to install or upgrade libraries
def install_or_upgrade(libraries):
    for lib in libraries:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", lib])

# Install or upgrade the libraries
install_or_upgrade(libraries)
