import os
import jax
import jax.numpy as jnp
import optax
import numpy as np
import pandas as pd
from flax import linen as nn
from jax import random
from constants import allowed_units
from utils import parse_string
import tensorflow as tf
#import onnx
#import onnxruntime
from flax.training import checkpoints
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Define EntityValueModel as provided
class EntityValueModel(nn.Module):
    num_units: int  # Number of unique units

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=128, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        value_output = nn.Dense(features=1)(x)  # Regression
        unit_logits = nn.Dense(features=self.num_units)(x)  # Classification
        return value_output.squeeze(), unit_logits

# Function to load data from train.csv
def load_data(processed_folder, csv_file):
    data = pd.read_csv(csv_file)
    images = []
    labels = []

    for _, row in data.iterrows():
        image_path = os.path.join(processed_folder, f"{row['Index']}.npy")  # Assuming images are preprocessed to .npy
        if os.path.exists(image_path):
            image = np.load(image_path)
            images.append(image)
            parsed_value = row['parsed_value'] if not pd.isna(row['parsed_value']) else 0.0
            unit_index = allowed_units.index(row['unit']) if row['unit'] in allowed_units else 0
            labels.append([parsed_value, unit_index])
    return jnp.array(images), jnp.array(labels)

# Load images and labels from train.csv
images, labels = load_data('processed_train', 'dataset/train.csv')

# Initialize the model and optimizer
model = EntityValueModel(num_units=len(allowed_units))
key = random.PRNGKey(0)
params = model.init(key, jnp.ones((1, 128, 128, 3)))  # Modify input shape as per image resolution

# Define optimizer and loss function
optimizer = optax.adam(learning_rate=1e-3)
opt_state = optimizer.init(params)

# Define loss function
def loss_fn(params, batch):
    inputs, targets = batch
    value_output, unit_logits = model.apply(params, inputs)
    value_loss = jnp.mean((value_output - targets[:, 0]) ** 2)  # Mean squared error for regression
    unit_loss = optax.softmax_cross_entropy(logits=unit_logits, labels=targets[:, 1])  # Softmax cross-entropy for classification
    return value_loss + unit_loss

# Training loop
for epoch in range(10):
    loss, grads = jax.value_and_grad(loss_fn)(params, (images, labels))
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    print(f"Epoch {epoch}, Loss: {loss}")

# Saving the model in different formats
# 1. Save as Flax Checkpoint
checkpoint_dir = './checkpoints'
checkpoints.save_checkpoint(ckpt_dir=checkpoint_dir, target=params, step=epoch, overwrite=True)
print(f"Model saved as JAX Flax checkpoint in {checkpoint_dir}")

# 2. Save as H5 (via Keras)
def save_as_h5(model_params, filepath='model.h5'):
    # Since Keras expects a TensorFlow model, here we simulate converting the Flax model to Keras
    keras_model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        tf.keras.layers.AveragePooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.AveragePooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.AveragePooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(1),  # For value_output (regression)
        tf.keras.layers.Dense(len(allowed_units), activation='softmax')  # For unit_logits (classification)
    ])
    keras_model.compile(optimizer='adam', loss='mse')
    keras_model.save(filepath)
    print(f"Model saved as H5 file in {filepath}")

save_as_h5(params, filepath='entity_value_model.h5')
