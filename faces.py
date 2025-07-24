import cv2
import os
from mtcnn import MTCNN

# Function to extract faces from a video and save them
def extract_faces(video_path, output_dir):
    detector = MTCNN()  # Initialize face detector
    cap = cv2.VideoCapture(video_path)  # Open video
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    frame_count = 0
    num_faces = 0
    while cap.isOpened():
        ret, frame = cap.read()  # Read frame
        if not ret:  # Break if no more frames
            break
        faces = detector.detect_faces(frame)  # Detect faces
        for i, face in enumerate(faces):
            x, y, w, h = face['box']  # Get face coordinates
            face_img = frame[y:y+h, x:x+w]  # Crop face
            # Save face with a unique name
            img_name = f"{os.path.basename(video_path)}_{frame_count}_{i}.jpg"
            cv2.imwrite(os.path.join(output_dir, img_name), face_img)
            num_faces += 1
        frame_count += 1
    cap.release()
    print(f"Extracted {num_faces} faces from {video_path}")

# Define input and output directories (replace with your paths)
real_videos_dir = "C:\\Users\\A M TECH\\Videos\\Celeb Real"
fake_videos_dir = "C:\\Users\\A M TECH\\Videos\\Celeb Fake"
real_output_dir = "C:\\Users\\A M TECH\\Desktop\\py playground\\Deepfake sys\\dataset\\real_faces"
fake_output_dir = "C:\\Users\\A M TECH\\Desktop\\py playground\\Deepfake sys\\dataset\\fake_faces"

# Create output directories if they don’t exist
os.makedirs(real_output_dir, exist_ok=True)
os.makedirs(fake_output_dir, exist_ok=True)

# Define video file extensions to process
video_extensions = ['.mp4', '.avi', '.mov', '.mkv']

# Process real videos
print("Extracting faces from real videos...")
for video in os.listdir(real_videos_dir):
    if os.path.splitext(video)[1].lower() in video_extensions:
        video_path = os.path.join(real_videos_dir, video)
        print(f"Processing video: {video_path}")
        try:
            extract_faces(video_path, real_output_dir)
        except Exception as e:
            print(f"Error processing {video_path}: {e}")

# Process fake videos
print("Extracting faces from fake videos...")
for video in os.listdir(fake_videos_dir):
    if os.path.splitext(video)[1].lower() in video_extensions:
        video_path = os.path.join(fake_videos_dir, video)
        print(f"Processing video: {video_path}")
        try:
            extract_faces(video_path, fake_output_dir)
        except Exception as e:
            print(f"Error processing {video_path}: {e}")

print("Face extraction complete!")