import cv2
import mediapipe as mp

def get_face_landmarks(image, draw=False, static_image_mode=True):
    """
    Extract face landmarks from an image using MediaPipe.
    Returns a list of 1404 normalized coordinates (468 points × 3).
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=static_image_mode,
        max_num_faces=1,
        min_detection_confidence=0.5
    )

    results = face_mesh.process(image_rgb)
    landmarks_list = []

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]

        if draw:
            mp_drawing = mp.solutions.drawing_utils
            drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec
            )

        xs, ys, zs = [], [], []
        for lm in face_landmarks.landmark:
            xs.append(lm.x)
            ys.append(lm.y)
            zs.append(lm.z)

        for i in range(len(xs)):
            landmarks_list.append(xs[i] - min(xs))
            landmarks_list.append(ys[i] - min(ys))
            landmarks_list.append(zs[i] - min(zs))

    return landmarks_list
