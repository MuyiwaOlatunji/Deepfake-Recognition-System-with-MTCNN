from flask import Flask, request, render_template
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from mtcnn import MTCNN
import logging
import os
import tempfile
import time
import psutil

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = Flask(__name__)
try:
    tf.keras.backend.clear_session()
    model = load_model('best_model_finetuned.keras')  # Use fine-tuned model
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error(f"Failed to load model: {e}")
    raise

detector = MTCNN()

# Preprocessing
datagen = ImageDataGenerator(rescale=1./255)

# ... (previous imports and setup remain the same)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
                temp_filename = temp_file.name
                try:
                    file.save(temp_filename)
                    logging.info(f"Saved temporary file: {temp_filename}")
                except Exception as e:
                    logging.error(f"Failed to save video file: {e}")
                    return render_template('index.html', result="Failed to save video file")
            
            # Process video
            cap = cv2.VideoCapture(temp_filename)
            if not cap.isOpened():
                cap.release()
                try:
                    os.unlink(temp_filename)
                except Exception as e:
                    logging.warning(f"Failed to delete {temp_filename}: {e}")
                return render_template('index.html', result="Invalid video file")
            
            predictions = []
            frame_count = 0
            max_frames = 50
            skip_frames = 10
            batch_size = 8
            batch_frames = []

            while cap.isOpened() and frame_count < max_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count * skip_frames)
                ret, frame = cap.read()
                if not ret:
                    break
                if frame is None or frame.size == 0:
                    logging.warning(f"Empty frame at index {frame_count}")
                    frame_count += 1
                    continue
                faces = detector.detect_faces(frame)
                if faces:
                    x, y, w, h = faces[0]['box']
                    if w > 0 and h > 0:
                        x, y = max(0, x), max(0, y)
                        face_img = frame[y:y+h, x:x+w]
                        if face_img.size == 0:
                            logging.warning(f"Empty face image at frame {frame_count}")
                            frame_count += 1
                            continue
                        face_img = cv2.resize(face_img, (96, 96))
                        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                        face_img = face_img.astype('float32')
                        batch_frames.append(face_img)
                    else:
                        logging.warning(f"Invalid face bounding box at frame {frame_count}")
                frame_count += 1

                if len(batch_frames) == batch_size or (frame_count >= max_frames and batch_frames):
                    batch_frames = np.array(batch_frames)
                    batch_frames = datagen.standardize(batch_frames)
                    try:
                        batch_preds = model.predict(batch_frames, verbose=0)
                        predictions.extend(batch_preds.flatten())
                    except Exception as e:
                        logging.warning(f"Prediction failed for batch at frame {frame_count}: {e}")
                    batch_frames = []
                    logging.info(f"Memory usage: {psutil.virtual_memory().percent}%")
            
            if batch_frames:
                batch_frames = np.array(batch_frames)
                batch_frames = datagen.standardize(batch_frames)
                try:
                    batch_preds = model.predict(batch_frames, verbose=0)
                    predictions.extend(batch_preds.flatten())
                except Exception as e:
                    logging.warning(f"Prediction failed for remaining frames: {e}")
            
            cap.release()
            cv2.destroyAllWindows()
            
            for attempt in range(10):
                try:
                    os.unlink(temp_filename)
                    logging.info(f"Successfully deleted {temp_filename}")
                    break
                except PermissionError:
                    logging.warning(f"Attempt {attempt + 1}: PermissionError deleting {temp_filename}. Retrying...")
                    time.sleep(0.5)
                except Exception as e:
                    logging.error(f"Failed to delete {temp_filename}: {e}")
                    break
            
            if not predictions:
                return render_template('index.html', result="No faces detected in the video")
            
            if predictions:
                alpha = 0.3
                smoothed_preds = [predictions[0]]
                for i in range(1, len(predictions)):
                    smoothed_preds.append(alpha * predictions[i] + (1 - alpha) * smoothed_preds[-1])
                avg_prediction = np.mean(smoothed_preds)
                result = 'Fake' if avg_prediction > 0.5 else 'Real'
                confidence = avg_prediction if avg_prediction > 0.5 else 1 - avg_prediction
                return render_template('index.html', result=f"{result} (Confidence: {confidence:.2%})")
        
        return render_template('index.html', result="Please upload a valid video file (MP4, AVI, MOV)")
    
    return render_template('index.html', result=None)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)