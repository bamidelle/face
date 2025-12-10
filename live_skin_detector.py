import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import mediapipe as mp
import numpy as np

st.title("Live Skin Problem Detector 💻📸")

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(max_num_faces=1)

# Function to process each video frame
def process_frame(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    # Simple red patch detection (acne/sunburn)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
    frame[mask_red > 0] = [0, 0, 255]  # red overlay

    # Simple dark patch detection (hyperpigmentation)
    lower_dark = np.array([0, 0, 0])
    upper_dark = np.array([179, 255, 60])
    mask_dark = cv2.inRange(hsv, lower_dark, upper_dark)
    frame[mask_dark > 0] = [0, 255, 255]  # yellow overlay

    return frame

# Callback for the live webcam
def video_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = process_frame(img)
    return img

# Streamlit WebRTC streamer
webrtc_streamer(key="skin-detector", video_frame_callback=video_callback)
