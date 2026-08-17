import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model

model = load_model("../models/mask_detector.keras")
face_detector = YOLO("../models/yolov8n-face-lindevs.pt")

cap = cv2.VideoCapture(0)
cv2.namedWindow(
    "Face Mask Detection",
    cv2.WINDOW_NORMAL
)
cv2.setWindowProperty(
    "Face Mask Detection",
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = face_detector(frame, verbose=False)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            x1 = max(0, x1)
            y1 = max(0, y1)

            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            face = frame[y1:y2, x1:x2]

            if face.size == 0:
                continue
            
            img = cv2.cvtColor(
                face,
                cv2.COLOR_BGR2RGB
            )

            img = cv2.resize(
                img,
                (224, 224)
            )

            img = img.astype(
                np.float32
            )

            img = np.expand_dims(
                img,
                axis=0
            )
            
            pred = model.predict(
                img,
                verbose=0
            )[0][0]

            print("Prediction:", pred)

            if pred > 0.5:
                label = "Mask"
                color = (0, 255, 0)
            else:
                label = "No Mask"
                color = (0, 0, 255)
                
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                3
            )

            cv2.putText(
                frame,
                f"{label} {pred:.3f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

    cv2.imshow(
        "Face Mask Detection",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()