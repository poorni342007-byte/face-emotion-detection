import cv2
from deepface import DeepFace
from collections import deque

emotion_window = deque(maxlen=10)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera not reading frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        try:
            result = DeepFace.analyze(
                face,
                actions=['emotion'],
                enforce_detection=False
            )

            emotion = result[0]['dominant_emotion']
            emotion_window.append(emotion)

            stable_emotion = max(
                set(emotion_window),
                key=emotion_window.count
            )

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, stable_emotion, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        except:
            pass

    cv2.imshow("Face + Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
