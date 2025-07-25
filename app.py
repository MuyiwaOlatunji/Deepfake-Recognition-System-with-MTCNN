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
from flask import Flask, request, render_template, send_from_directory, Response

# **Logging Setup**
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# **Suppress TensorFlow Warnings**
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# **Load Fine-Tuned Model**
MODEL_PATH = 'best_model_finetuned.keras'  # Adjust this path if your model is located elsewhere
try:
    model = load_model(MODEL_PATH)
    logging.info(f"Model loaded successfully from {MODEL_PATH}")
    logging.info(f"Model input shape: {model.input_shape}")
except Exception as e:
    logging.error(f"Failed to load model: {e}")
    raise

# **Initialize Face Detector (MTCNN)**
try:
    detector = MTCNN(min_face_size=20)
    logging.info("MTCNN initialized with min_face_size=20")
except TypeError:
    detector = MTCNN()
    logging.info("MTCNN initialized with default settings")

# **Preprocessing Setup**
datagen = ImageDataGenerator(rescale=1./255)

# **Initialize Flask App**
app = Flask(__name__)

# **Favicon Route to Prevent 404 Errors**
@app.route('/favicon.ico')
def favicon():
    favicon_path = os.path.join(app.root_path, 'static', 'favicon.ico')
    if os.path.exists(favicon_path):
        return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')
    return Response(status=204)

# **Main Route for Video Upload and Processing**
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        # **Validate Uploaded File**
        if file and file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
            # **Save Video to Temporary File**
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
                temp_filename = temp_file.name
                try:
                    file.save(temp_filename)
                    logging.info(f"Saved temporary file: {temp_filename}")
                except Exception as e:
                    logging.error(f"Failed to save video file: {e}")
                    return render_template('index.html', result="Failed to save video file")

            # **Open Video File**
            cap = cv2.VideoCapture(temp_filename)
            if not cap.isOpened():
                cap.release()
                try:
                    os.unlink(temp_filename)
                except Exception as e:
                    logging.warning(f"Failed to delete {temp_filename}: {e}")
                return render_template('index.html', result="Invalid video file")

            # **Initialize Variables for Processing**
            predictions = []
            frame_count = 0
            max_frames = 30  # Limit total frames processed
            skip_frames = 1  # Process every frame (adjust for faster processing if needed)
            batch_size = 2   # Number of faces to predict at once
            batch_frames = []

            logging.debug(f"Starting video processing: max_frames={max_frames}, skip_frames={skip_frames}, batch_size={batch_size}")
            logging.info(f"Initial memory usage: {psutil.virtual_memory().percent}%")

            # **Process Video Frames**
            while cap.isOpened() and frame_count < max_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count * skip_frames)
                ret, frame = cap.read()
                if not ret:
                    logging.debug(f"No more frames at index {frame_count}")
                    break
                if frame is None or frame.size == 0:
                    logging.warning(f"Empty frame at index {frame_count}")
                    frame_count += 1
                    continue

                # **Preprocess Frame: Resize and Sharpen**
                frame = cv2.resize(frame, (640, 360))  # Reduce resolution for memory efficiency
                kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])  # Sharpening kernel
                frame = cv2.filter2D(frame, -1, kernel)

                # **Detect Faces**
                faces = detector.detect_faces(frame)
                logging.debug(f"Detected {len(faces)} faces in frame {frame_count}")
                if faces:
                    x, y, w, h = faces[0]['box']  # Use the first detected face
                    if w > 0 and h > 0:
                        x, y = max(0, x), max(0, y)  # Ensure non-negative coordinates
                        face_img = frame[y:y+h, x:x+w]
                        if face_img.size == 0:
                            logging.warning(f"Empty face image at frame {frame_count}")
                            frame_count += 1
                            continue

                        # **Preprocess Face for Model**
                        face_img = cv2.resize(face_img, (64, 64))  # Adjust size to match model input
                        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                        face_img = face_img.astype('float32')
                        batch_frames.append(face_img)
                        logging.debug(f"Added face image to batch at frame {frame_count}")
                    else:
                        logging.warning(f"Invalid face bounding box at frame {frame_count}: w={w}, h={h}")
                frame_count += 1

                # **Process Batch of Faces**
                if len(batch_frames) == batch_size or (frame_count >= max_frames and batch_frames):
                    batch_frames = np.array(batch_frames)
                    batch_frames = datagen.standardize(batch_frames)  # Normalize pixel values
                    try:
                        batch_preds = model.predict(batch_frames, verbose=0)
                        predictions.extend(batch_preds.flatten())
                        logging.debug(f"Predictions for batch: {batch_preds.flatten()}")
                    except Exception as e:
                        logging.warning(f"Prediction failed for batch at frame {frame_count}: {e}")
                    batch_frames = []  # Clear batch
                    logging.info(f"Memory usage after batch: {psutil.virtual_memory().percent}%")

            # **Process Any Remaining Frames**
            if batch_frames:
                batch_frames = np.array(batch_frames)
                batch_frames = datagen.standardize(batch_frames)
                try:
                    batch_preds = model.predict(batch_frames, verbose=0)
                    predictions.extend(batch_preds.flatten())
                    logging.debug(f"Predictions for remaining frames: {batch_preds.flatten()}")
                except Exception as e:
                    logging.warning(f"Prediction failed for remaining frames: {e}")
                logging.info(f"Memory usage after remaining frames: {psutil.virtual_memory().percent}%")

            logging.info(f"Final memory usage: {psutil.virtual_memory().percent}%")

            # **Clean Up**
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

            # **Generate Result**
            if not predictions:
                logging.info("No predictions generated")
                return render_template('index.html', result="No faces detected in the video")

            # **Smooth Predictions and Compute Average**
            alpha = 0.3  # Smoothing factor
            smoothed_preds = [predictions[0]]
            for i in range(1, len(predictions)):
                smoothed_preds.append(alpha * predictions[i] + (1 - alpha) * smoothed_preds[-1])
            avg_prediction = np.mean(smoothed_preds)
            result = 'Fake' if avg_prediction > 0.5 else 'Real'
            confidence = avg_prediction if avg_prediction > 0.5 else 1 - avg_prediction
            logging.info(f"Final result: {result} (Confidence: {confidence:.2%})")
            return render_template('index.html', result=f"{result} (Confidence: {confidence:.2%})")

        logging.warning("Invalid file uploaded")
        return render_template('index.html', result="Please upload a valid video file (MP4, AVI, MOV)")

    # **Render Main Page for GET Request**
    return render_template('index.html', result=None)

# **Run the App**
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)