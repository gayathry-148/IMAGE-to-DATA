import os
import pandas as pd
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils import download_images


# Path to dataset
train_csv_path = 'dataset/train.csv'
train_image_folder = 'train_images'

# Create folder if it doesn't exist
os.makedirs(train_image_folder, exist_ok=True)

# Read CSV and get image links
train_data = pd.read_csv(train_csv_path)
image_links = train_data['image_link'].tolist()

# Download images using the utility function
download_images(image_links, train_image_folder, allow_multiprocessing=True)
