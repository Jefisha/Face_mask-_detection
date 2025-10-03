import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("face_mask_model.h5")

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) > 0:
        # Pick the largest face (focus on you, ignore background false detections)
        faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
        (x, y, w, h) = faces[0]  

        # Draw rectangle only on largest face
        face_roi = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face_roi, (128, 128))  # match your model input size
        face_normalized = face_resized / 255.0
        face_input = np.expand_dims(face_normalized, axis=0)

        # Predict mask / no mask
        pred = model.predict(face_input)[0]
        label = "Mask" if pred[0] > 0.5 else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    cv2.imshow("Mask Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
