import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import cv2
from utils import get_face_landmarks

# Load dataset
data = np.loadtxt('data.txt')
X = data[:, :-1]  # features
y = data[:, -1]   # labels

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y
)

# Train Random Forest
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)

# Evaluate
y_pred = rf_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100:.2f}%")
print(confusion_matrix(y_test, y_pred))

# Save model
with open('model', 'wb') as f:
    pickle.dump(rf_classifier, f)

# -------- Live webcam emotion recognition --------
emotion_labels = ['happy', 'sad', 'surprised']  # match your folder names

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception("❌ Could not open camera")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    landmarks = get_face_landmarks(frame)
    if landmarks and len(landmarks) == 1404:
        X_input = np.array(landmarks).reshape(1, -1)
        predicted_index = int(rf_classifier.predict(X_input)[0])
        emotion_text = emotion_labels[predicted_index]
        cv2.putText(frame, emotion_text, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Emotion Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
