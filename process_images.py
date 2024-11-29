import os
from PIL import Image
import numpy as np
import jax.numpy as jnp
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import multiprocessing
from utils import create_placeholder_image

# Folder paths
train_image_folder = 'test_images'
processed_image_folder = 'processed_test'

# Create the processed folder if it doesn't exist
os.makedirs(processed_image_folder, exist_ok=True)

# List all images in the train folder
image_filenames = os.listdir(train_image_folder)

def process_image(image_name, train_image_folder, processed_image_folder):
    image_path = os.path.join(train_image_folder, image_name)
    processed_image_path = os.path.join(processed_image_folder, image_name.split('.')[0] + '.npy')

    # Skip if already processed
    if os.path.exists(processed_image_path):
        return f"Skipping {image_name}, already processed."

    try:
        image = Image.open(image_path).convert('RGB')
        image_resized = image.resize((128, 128))  # Resize to 128x128
        image_array = np.array(image_resized)
        processed_image = jnp.array(image_array) / 255.0  # Normalize image
        # Save processed image as a numpy array file (.npy)
        np.save(processed_image_path, processed_image)
        return f"Processed {image_name}"
    except:
        # Handle invalid images
        create_placeholder_image(os.path.join(processed_image_folder, image_name))
        return f"Invalid {image_name}, placeholder created."

# Use multiprocessing to process images in parallel
if __name__ == '__main__':
    num_workers = min(8, multiprocessing.cpu_count())  # Limit to 8 or available CPUs
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Use partial to pass folder paths
        process_func = partial(process_image, train_image_folder=train_image_folder, processed_image_folder=processed_image_folder)
        
        # Process images in parallel
        results = list(tqdm(executor.map(process_func, image_filenames), total=len(image_filenames)))

    # Print the results
    for result in results:
        print(result)
