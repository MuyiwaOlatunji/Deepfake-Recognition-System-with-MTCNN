import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Paths to the dataset
real_dir = "C:\\Users\\A M TECH\\Desktop\\py playground\\Deepfake sys\\dataset\\real_faces"
fake_dir = "C:\\Users\\A M TECH\\Desktop\\py playground\\Deepfake sys\\dataset\\fake_faces"

# Function to collect image paths and labels
def collect_image_paths(max_images_per_class=5000):
    paths = []
    labels = []
    for dir_path, label in [(real_dir, 0), (fake_dir, 1)]:
        if not os.path.exists(dir_path):
            logging.error(f"Directory not found: {dir_path}")
            return [], []
        img_files = [f for f in os.listdir(dir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        img_files = img_files[:max_images_per_class]
        for img_name in img_files:
            img_path = os.path.join(dir_path, img_name)
            paths.append(img_path)
            labels.append(label)
        logging.info(f"Collected {len(img_files)} image paths from {dir_path}")
    return paths, labels

# Collect paths and split
logging.info("Collecting image paths and splitting data...")
paths, labels = collect_image_paths(max_images_per_class=5000)
if not paths:
    logging.error("No valid images found. Exiting.")
    exit(1)

paths_train, paths_temp, y_train, y_temp = train_test_split(paths, labels, test_size=0.3, random_state=42)
paths_val, paths_test, y_val, y_test = train_test_split(paths_temp, y_temp, test_size=0.5, random_state=42)

# Save splits to CSV
split_data = {
    'train': (paths_train, y_train),
    'val': (paths_val, y_val),
    'test': (paths_test, y_test)
}
for split_name, (split_paths, split_labels) in split_data.items():
    df = pd.DataFrame({'path': split_paths, 'label': split_labels})
    df.to_csv(f'{split_name}_split.csv', index=False)
    logging.info(f"Saved {split_name} split with {len(split_paths)} images")

# Data augmentation and preprocessing
logging.info("Setting up data augmentation...")
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

# Function to create data generator
def create_generator(paths, labels, batch_size=16):
    def gen():
        for img_path, label in zip(paths, labels):
            img = cv2.imread(img_path)
            if img is None:
                logging.warning(f"Failed to load image: {img_path}")
                continue
            img = cv2.resize(img, (64, 64))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            yield img, label
    dataset = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(64, 64, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# Create datasets
batch_size = 16
train_dataset = create_generator(paths_train, y_train, batch_size)
val_dataset = create_generator(paths_val, y_val, batch_size)
test_dataset = create_generator(paths_test, y_test, batch_size)

logging.info("Data preparation complete! Data is ready for training.")