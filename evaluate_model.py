import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import logging
import numpy as np
import cv2
import os
import seaborn as sns
import matplotlib.pyplot as plt

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load test data
logging.info("Loading test data...")
try:
    test_df = pd.read_csv('test_split.csv')
    logging.info(f"Test dataset size: {len(test_df)} images")
except Exception as e:
    logging.error(f"Failed to load test_split.csv: {e}")
    raise

# Create test dataset
def create_generator(df, batch_size=8):
    datagen = ImageDataGenerator(rescale=1./255)
    def gen():
        for _, row in df.iterrows():
            img = cv2.imread(row['path'])
            if img is None:
                logging.warning(f"Failed to load image: {row['path']}")
                continue
            img = cv2.resize(img, (64, 64))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype('float32')
            img = datagen.standardize(np.expand_dims(img, axis=0))[0]
            yield img, row['label']
    dataset = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(64, 64, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE).cache()
    return dataset

# Video prediction function
def predict_video(video_path, model, frame_interval=10, resize=(64, 64), batch_size=8):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frame = cv2.resize(frame, resize)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype('float32') / 255.0
            frames.append(frame)
        frame_count += 1
        if len(frames) == batch_size:  # Process in batches
            preds = model.predict(np.array(frames), verbose=0)
            frames = []  # Clear frames to save memory
            yield preds.flatten()
    cap.release()
    if frames:  # Process remaining frames
        preds = model.predict(np.array(frames), verbose=0)
        yield preds.flatten()

# Aggregate video predictions
def aggregate_video_predictions(video_path, model):
    preds = []
    for batch_preds in predict_video(video_path, model):
        preds.extend(batch_preds)
    if preds:
        # Smooth predictions with exponential moving average
        alpha = 0.3
        smoothed_preds = [preds[0]]
        for i in range(1, len(preds)):
            smoothed_preds.append(alpha * preds[i] + (1 - alpha) * smoothed_preds[-1])
        video_pred = np.mean(smoothed_preds) > 0.5
        return int(video_pred), np.mean(smoothed_preds)
    return None, None

batch_size = 8
test_dataset = create_generator(test_df, batch_size=batch_size)

# Load the best model
logging.info("Loading model...")
try:
    tf.keras.backend.clear_session()
    model = load_model('best_model.keras')
except Exception as e:
    logging.error(f"Failed to load model: {e}")
    raise

# Evaluate the model
logging.info("Evaluating model...")
try:
    steps = (len(test_df) + batch_size - 1) // batch_size
    logging.info(f"Evaluation steps: {steps}")
    test_loss, test_accuracy = model.evaluate(test_dataset, steps=steps, verbose=0)
    logging.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
except Exception as e:
    logging.error(f"Evaluation failed: {e}")
    raise

# Make predictions
logging.info("Making predictions...")
y_pred = []
y_true = []
try:
    test_dataset = create_generator(test_df, batch_size=batch_size)
    for images, labels in test_dataset.take(steps):
        preds = (model.predict(images, verbose=0) > 0.5).astype("int32")
        y_pred.extend(preds.flatten())
        y_true.extend(labels.numpy())
    logging.info(f"Processed {len(y_true)} samples")
except Exception as e:
    logging.error(f"Prediction failed: {e}")
    raise

# Print detailed report
logging.info("Generating classification report...")
print(classification_report(y_true, y_pred, target_names=['Real', 'Fake']))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.close()

# Debug label distribution
logging.info("Label distribution in test set:")
print(test_df['label'].value_counts())

# Example video prediction
# video_path = "path/to/test/video.mp4"
# label, confidence = predict_video(video_path, model)
# logging.info(f"Video prediction: {'Fake' if label else 'Real'}, Confidence: {confidence:.4f}")