import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="Face Mask Detection",
    layout="wide"
)

st.title("😷 Face Mask Detection")
st.markdown("Real-time face mask detection using YOLOv8 and TensorFlow")

@st.cache_resource
def load_models():
    mask_model = load_model("../models/mask_detector.keras")
    face_detector = YOLO("../models/yolov8n-face-lindevs.pt")
    return mask_model, face_detector


mask_model, face_detector = load_models()


class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        results = face_detector(img, verbose=False)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(img.shape[1], x2)
                y2 = min(img.shape[0], y2)

                face = img[y1:y2, x1:x2]

                if face.size == 0:
                    continue

                face_rgb = cv2.cvtColor(
                    face,
                    cv2.COLOR_BGR2RGB
                )

                face_rgb = cv2.resize(
                    face_rgb,
                    (224, 224)
                )

                face_rgb = face_rgb.astype(np.float32)

                face_rgb = np.expand_dims(
                    face_rgb,
                    axis=0
                )

                pred = mask_model.predict(
                    face_rgb,
                    verbose=0
                )[0][0]

                if pred > 0.5:
                    label = "Mask"
                    color = (0, 255, 0)
                else:
                    label = "No Mask"
                    color = (0, 0, 255)

                cv2.rectangle(
                    img,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                cv2.putText(
                    img,
                    f"{label} {pred:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )


st.set_page_config(layout="wide")

webrtc_streamer(
    key="mask-detection",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True
)