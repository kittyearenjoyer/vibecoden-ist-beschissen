import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

st.title("AI Security Camera")

# Modell laden
model = YOLO("yolov8n.pt")

uploaded_file = st.file_uploader("Upload image", type=["jpg","png","jpeg"])

if uploaded_file:

    image = Image.open(uploaded_file)
    frame = np.array(image)

    results = model(frame)

    person_detected = False

    for r in results:
        boxes = r.boxes

        for box in boxes:

            cls = int(box.cls[0])

            if cls == 0:
                person_detected = True

            x1,y1,x2,y2 = map(int,box.xyxy[0])

            cv2.rectangle(
                frame,
                (x1,y1),
                (x2,y2),
                (0,255,0),
                2
            )

    if person_detected:
        st.warning("PERSON DETECTED")

    st.image(frame)
