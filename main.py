import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Run download_images.py
print("Downloading images...")
os.system('python src/download_images.py')

# Run process_images.py
print("Processing images...")
os.system('python src/process_images.py')

# Run train.py
print("Training model...")
os.system('python src/train.py')

print("All steps completed successfully.")
