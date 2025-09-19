import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import cv2
from utils import get_face_landmarks

# ------------------ Load dataset ------------------
data_file = 'data.txt'

try:
    data = np.loadtxt(data_file)
except OSError:
    raise FileNotFoundError(f"❌ Dataset file not found: {data_file}. Run prepare_data.py first.")

X = data[:, :-1]  # features
y = data[:, -1]   # labels

# ------------------ Split train/test ------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y
)

# ------------------ Train Random Forest ------------------
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)

# ------------------ Evaluate ------------------
y_pred = rf_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy*100:.2f}%")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ------------------ Save model ------------------
with open('model', 'wb') as f:
    pickle.dump(rf_classifier, f)
print("✅ Model saved as './model'")

# ------------------ Webcam / Emotion Recognition ------------------
emotion_labels = ['happy', 'sad', 'surprised']
cap = None

# Meter parameters
meter_width = 300
meter_height = 60
x_offset = 50
y_offset = 50
arrow_x = x_offset

# Arrow positions for each emotion
emotion_positions = {
    "sad": x_offset + 30,
    "happy": x_offset + 150,
    "surprised": x_offset + 270
}

# Colors for zones
zone_colors = {
    "sad": (255, 100, 100),
    "happy": (100, 255, 100),
    "surprised": (100, 100, 255)
}

def draw_pretty_meter(frame, arrow_pos, emotion_text):
    # Draw gradient zones
    cv2.rectangle(frame, (x_offset, y_offset),
                  (x_offset + meter_width, y_offset + meter_height),
                  (230, 230, 230), -1)
    
    # Draw colored zones
    cv2.rectangle(frame, (x_offset, y_offset),
                  (emotion_positions["sad"] + 20, y_offset + meter_height),
                  zone_colors["sad"], -1)
    cv2.rectangle(frame, (emotion_positions["sad"] + 20, y_offset),
                  (emotion_positions["happy"] + 20, y_offset + meter_height),
                  zone_colors["happy"], -1)
    cv2.rectangle(frame, (emotion_positions["happy"] + 20, y_offset),
                  (x_offset + meter_width, y_offset + meter_height),
                  zone_colors["surprised"], -1)

    # Draw meter border
    cv2.rectangle(frame, (x_offset, y_offset),
                  (x_offset + meter_width, y_offset + meter_height),
                  (50, 50, 50), 3, lineType=cv2.LINE_AA)

    # Draw emotion labels
    cv2.putText(frame, "Sad", (emotion_positions["sad"] - 15, y_offset + meter_height + 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (50, 50, 50), 2)
    cv2.putText(frame, "Happy", (emotion_positions["happy"] - 30, y_offset + meter_height + 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (50, 50, 50), 2)
    cv2.putText(frame, "Surprised", (emotion_positions["surprised"] - 50, y_offset + meter_height + 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (50, 50, 50), 2)

    # Draw arrow shadow
    arrow_y_start = y_offset + meter_height
    arrow_y_end = y_offset - 10
    cv2.arrowedLine(frame, (arrow_pos + 2, arrow_y_start + 2),
                    (arrow_pos + 2, arrow_y_end + 2), (50, 50, 50), 6, tipLength=0.4)

    # Draw main arrow
    cv2.arrowedLine(frame, (arrow_pos, arrow_y_start),
                    (arrow_pos, arrow_y_end), (0, 0, 255), 5, tipLength=0.4)

    # Draw emotion text above arrow
    cv2.putText(frame, emotion_text, (arrow_pos - 40, arrow_y_end - 15),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)

def start_webcam():
    global cap, arrow_x
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("❌ Could not open webcam")

    print("🎥 Press 'q' to quit the webcam window")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to grab frame")
            break

        landmarks = get_face_landmarks(frame)
        if landmarks and len(landmarks) == 1404:
            X_input = np.array(landmarks).reshape(1, -1)
            predicted_index = int(rf_classifier.predict(X_input)[0])
            emotion_text = emotion_labels[predicted_index]

            # Smooth arrow movement
            target_x = emotion_positions[emotion_text]
            arrow_x = int(arrow_x + (target_x - arrow_x) * 0.2)

            # Draw fancy meter
            draw_pretty_meter(frame, arrow_x, emotion_text)

        cv2.imshow("Emotion Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            close_camera()
            break

def close_camera():
    global cap
    if cap is not None:
        cap.release()
        cap = None
    cv2.destroyAllWindows()
    print("🛑 Webcam closed successfully.")

if __name__ == "__main__":
    start_webcam()
