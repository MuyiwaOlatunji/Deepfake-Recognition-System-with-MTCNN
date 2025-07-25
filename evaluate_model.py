import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report
import logging
import numpy as np
import cv2
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load test data
logging.info("Loading test data...")
test_df = pd.read_csv('test_split.csv')

# Create test dataset
def create_generator(df, batch_size=16):
    def gen():
        for _, row in df.iterrows():
            img = cv2.imread(row['path'])
            if img is None:
                logging.warning(f"Failed to load image: {row['path']}")
                continue
            img = cv2.resize(img, (64, 64))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            yield img, row['label']
    dataset = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(64, 64, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

test_dataset = create_generator(test_df, batch_size=16)

# Load the best model
logging.info("Loading model...")
model = load_model('best_model.keras')

# Evaluate the model
logging.info("Evaluating model...")
test_loss, test_accuracy = model.evaluate(test_dataset)
logging.info(f"Test Accuracy: {test_accuracy:.2f}")

# Make predictions
logging.info("Making predictions...")
y_pred = []
y_true = []
for images, labels in test_dataset:
    preds = (model.predict(images) > 0.5).astype("int32")
    y_pred.extend(preds.flatten())
    y_true.extend(labels.numpy())

# Print detailed report
logging.info("Generating classification report...")
print(classification_report(y_true, y_pred, target_names=['Real', 'Fake']))