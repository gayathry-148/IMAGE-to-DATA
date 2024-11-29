import pandas as pd
import os
from utils import download_images  # Assuming download_images is implemented in src/utils.py

# Load the test.csv file containing image links
test_file = 'dataset/test.csv'  # Path to test.csv
test_data = pd.read_csv(test_file)

# Extract image links from test.csv
image_links = test_data['image_link'].tolist()

# Set folder to save test images
test_image_folder = 'test_images'

# Create the folder if it doesn't exist
if not os.path.exists(test_image_folder):
    os.makedirs(test_image_folder)

# Download test images using the same download_images function
print("Downloading test images...")
download_images(image_links, test_image_folder, allow_multiprocessing=True)
