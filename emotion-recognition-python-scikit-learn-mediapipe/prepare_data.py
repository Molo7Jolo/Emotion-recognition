import os
import cv2
import numpy as np
from utils import get_face_landmarks

# Path to your dataset
data_dir = './data'  # make sure it contains 'happy', 'sad', 'surprised'

if not os.path.exists(data_dir):
    raise FileNotFoundError(f"❌ Dataset folder not found: {data_dir}")

output = []

for emotion_indx, emotion in enumerate(sorted(os.listdir(data_dir))):
    emotion_folder = os.path.join(data_dir, emotion)
    if not os.path.isdir(emotion_folder):
        continue

    print(f"Processing emotion: {emotion}")

    for image_file in os.listdir(emotion_folder):
        image_path = os.path.join(emotion_folder, image_file)
        image = cv2.imread(image_path)
        if image is None:
            print(f"⚠️ Could not read image: {image_path}")
            continue

        landmarks = get_face_landmarks(image)
        if landmarks and len(landmarks) == 1404:
            landmarks.append(int(emotion_indx))
            output.append(landmarks)

if output:
    np.savetxt('data.txt', np.asarray(output), fmt='%s')
    print(f"✅ Saved {len(output)} samples to data.txt")
else:
    print("❌ No valid data extracted.")
