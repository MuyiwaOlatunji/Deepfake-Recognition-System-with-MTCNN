import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from sklearn.utils.class_weight import compute_class_weight
import logging
import os
import cv2
from tensorflow.keras.mixed_precision import set_global_policy

# Suppress TensorFlow warnings and enable mixed precision
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
set_global_policy('mixed_float16')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to create data generator
def create_generator(df, batch_size=8):
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    def gen():
        for _, row in df.iterrows():
            img = cv2.imread(row['path'])
            if img is None:
                continue
            img = cv2.resize(img, (64, 64))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = datagen.random_transform(img.astype('float32'))
            yield img / 255.0, row['label']
    dataset = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(64, 64, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE).cache()
    return dataset

# Load split data
logging.info("Loading split data...")
try:
    train_df = pd.read_csv('train_split.csv')
    val_df = pd.read_csv('val_split.csv')
except Exception as e:
    logging.error(f"Failed to load CSV files: {e}")
    raise

# Create datasets
batch_size = 8
train_dataset = create_generator(train_df, batch_size=batch_size)
val_dataset = create_generator(val_df, batch_size=batch_size)

# Calculate steps per epoch
steps_per_epoch = len(train_df) // batch_size
validation_steps = len(val_df) // batch_size

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(train_df['label']), y=train_df['label'])
class_weight_dict = dict(enumerate(class_weights))

# Build the model
logging.info("Building the model...")
tf.keras.backend.clear_session()
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
base_model.trainable = False

model = Sequential([
    Input(shape=(64, 64, 3)),
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.6),
    Dense(1, activation='sigmoid', dtype='float32')  # Mixed precision compatibility
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
logging.info("Starting model training...")
try:
    # Initial training
    model.fit(
        train_dataset,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_dataset,
        validation_steps=validation_steps,
        epochs=12,  # Reduced epochs
        class_weight=class_weight_dict,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                'best_model.keras',
                save_best_only=True,
                monitor='val_loss',
                mode='min'
            ),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        ]
    )

    # Fine-tuning
    logging.info("Fine-tuning model...")
    base_model.trainable = True
    # Freeze all but the last 20 layers to save memory
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # Very low learning rate
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(
        train_dataset,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_dataset,
        validation_steps=validation_steps,
        epochs=5,  # Short fine-tuning phase
        class_weight=class_weight_dict,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                'best_model_finetuned.keras',
                save_best_only=True,
                monitor='val_loss',
                mode='min'
            ),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        ]
    )

    logging.info("Model training and fine-tuning complete!")
    model.save('best_model_finetuned.keras')

except Exception as e:
    logging.error(f"Training failed: {e}")
    raise

logging.info("Model training complete!")
model.save('best_model.keras')