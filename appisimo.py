import matplotlib
matplotlib.use('Agg')  # Ensures Matplotlib uses a backend compatible with Streamlit

import streamlit as st
import tempfile
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import os
import sys

# Initialize Mediapipe for pose detection
mp_pose = mp.solutions.pose  # Pose model for detecting body landmarks
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities for visualizing landmarks

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    ab = np.array(a) - np.array(b)  # Vector from b to a
    bc = np.array(c) - np.array(b)  # Vector from b to c
    dot_product = np.dot(ab, bc)  # Dot product of vectors
    magnitude_ab = np.linalg.norm(ab)  # Magnitude of vector ab
    magnitude_bc = np.linalg.norm(bc)  # Magnitude of vector bc
    angle = np.arccos(dot_product / (magnitude_ab * magnitude_bc))  # Angle in radians
    return np.degrees(angle)  # Convert angle to degrees

# Function to calculate pose angles (specifically knee angle)
def calculate_pose_angles(landmarks, width, height):
    # Extract landmarks for key joints
    left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
    # Convert to pixel coordinates
    left_knee_coords = (left_knee.x * width, left_knee.y * height)
    left_hip_coords = (left_hip.x * width, left_hip.y * height)
    left_ankle_coords = (left_ankle.x * width, left_ankle.y * height)
    return calculate_angle(left_hip_coords, left_knee_coords, left_ankle_coords)  # Return angle

# Function to calculate vertical displacement of joints
def calculate_vertical_displacement(joint_positions):
    return [position[1] for position in joint_positions]  # Extract y-coordinates (vertical displacement)

# Function to calculate stride frequency
def calculate_stride_frequency(vertical_displacement_ankle, time_points, threshold=0.02):
    contact_times = []  # List to track foot contact times
    previous_state = False  # Foot off the ground initially
    for i, displacement in enumerate(vertical_displacement_ankle):
        if displacement < threshold and not previous_state:  # Detect when foot contacts the ground
            contact_times.append(time_points[i])
            previous_state = True
        elif displacement > threshold and previous_state:  # Detect when foot lifts off the ground
            previous_state = False
    stride_count = len(contact_times) // 2  # Count strides (2 contact events per stride)
    return stride_count / (time_points[-1] - time_points[0])  # Strides per second

# Function to calculate vertical displacement of the body
def calculate_vertical_displacement_of_body(joint_positions):
    return [
        (joint_positions["left_hip"][i][1] + joint_positions["right_hip"][i][1]) / 2  # Average y-displacement of hips
        for i in range(len(joint_positions["left_hip"]))
    ]

# Function to calculate stride length (distance between consecutive heel strikes)
def calculate_stride_length(ankle_positions):
    stride_lengths = []
    for i in range(1, len(ankle_positions)):
        prev_ankle = ankle_positions[i-1]
        curr_ankle = ankle_positions[i]
        distance = np.linalg.norm(np.array(curr_ankle) - np.array(prev_ankle))  # Euclidean distance
        stride_lengths.append(distance)  # Append stride length
    return stride_lengths

# Function to analyze stride performance (displacement, frequency, and stride length)
def analyze_stride_performance(joint_positions, time_points):
    left_ankle_displacement = calculate_vertical_displacement(joint_positions["left_ankle"])
    right_ankle_displacement = calculate_vertical_displacement(joint_positions["right_ankle"])
    stride_frequency = calculate_stride_frequency(left_ankle_displacement + right_ankle_displacement, time_points)

    # Plot vertical displacement of ankles
    plt.figure(figsize=(6, 4))
    plt.plot(time_points, left_ankle_displacement, label="Left Ankle")
    plt.plot(time_points, right_ankle_displacement, label="Right Ankle")
    plt.xlabel('Time (s)')
    plt.ylabel('Vertical Displacement (m)')
    plt.title('Vertical Displacement of Ankles')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)  # Display plot in Streamlit

    st.write(f"Stride Frequency: {stride_frequency:.2f} strides per second")  # Display stride frequency

    # Plot vertical body displacement (average of hips)
    vertical_displacement_body = calculate_vertical_displacement_of_body(joint_positions)
    plt.figure(figsize=(6, 4))
    plt.plot(time_points, vertical_displacement_body, label="Body Vertical Displacement", color="g")
    plt.xlabel('Time (s)')
    plt.ylabel('Vertical Displacement (m)')
    plt.title('Vertical Displacement of Body')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)  # Display plot in Streamlit

    # Calculate stride length and plot it
    left_ankle_positions = joint_positions["left_ankle"]
    right_ankle_positions = joint_positions["right_ankle"]
    stride_lengths = calculate_stride_length(left_ankle_positions + right_ankle_positions)
    
    plt.figure(figsize=(6, 4))
    plt.plot(range(len(stride_lengths)), stride_lengths, label="Stride Length")
    plt.xlabel('Stride Number')
    plt.ylabel('Stride Length (m)')
    plt.title('Stride Length Over Time')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

    return stride_frequency  # Return calculated stride frequency

# Function to classify running style based on stride frequency and vertical oscillation
def classify_running_style(stride_frequency, vertical_oscillation):
    if stride_frequency > 2.5 and vertical_oscillation > 0.08:
        return "Power Racer"
    elif stride_frequency > 2.0 and vertical_oscillation <= 0.08:
        return "Eco Sprinter"
    elif stride_frequency < 1.5:
        return "Easy Strider"
    else:
        return "Constant Glider"

# Streamlit UI
st.title("Stride Analysis and Pose Estimation")

col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("Upload up to 3 running videos:")
    st.markdown("""
    - **Suggest lenght of 10 seconds to 1 minutes**
    - **100 MB max**
    - Formats: **mp4, mov, avi, mkv**
    """)
with col2:
    st.image("recording_instructions.jpg", caption="Recording Instructions", width=250)

joint_color = st.color_picker("Pick a color for joints", "#FF0000")  # Color picker for joint visualization
selected_color = tuple(int(joint_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))  # Convert color to RGB

# Upload video files
uploaded_file_L = st.file_uploader("Upload Left Angle Video", type=["mp4", "mov", "avi", "mkv"])
uploaded_file_C = st.file_uploader("Upload Center Angle Video", type=["mp4", "mov", "avi", "mkv"])
uploaded_file_R = st.file_uploader("Upload Right Angle Video", type=["mp4", "mov", "avi", "mkv"])

uploaded_files = [uploaded_file_L, uploaded_file_C, uploaded_file_R]

if any(uploaded_files):  # If any video is uploaded
    stride_frequencies = []  # List to store stride frequencies for each video
    for index, uploaded_file in enumerate(uploaded_files):
        if uploaded_file:
            st.markdown(f"### Processing Video {index + 1}")
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")  # Temporary file for video
            temp_file.write(uploaded_file.read())
            video_path = temp_file.name

            cap = cv2.VideoCapture(video_path)  # Open video with OpenCV
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Get video frame width
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Get video frame height
            fps = int(cap.get(cv2.CAP_PROP_FPS))  # Get frames per second

            time_points = []  # List to store time points
            joint_positions = {"left_ankle": [], "right_ankle": [], "left_hip": [], "right_hip": []}  # Joint positions dictionary
            stframe = st.empty()  # Empty placeholder for displaying frames

            with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                while cap.isOpened():
                    ret, frame = cap.read()  # Read frame from video
                    if not ret:
                        break
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for Mediapipe
                    results = pose.process(image)  # Process image to get pose landmarks
                    annotated_image = image.copy()  # Copy image for annotation

                    if results.pose_landmarks:  # If pose landmarks are found
                        # Draw landmarks on the image
                        mp_drawing.draw_landmarks(
                            annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=selected_color, thickness=2, circle_radius=3)
                        )

                        # Extract specific joint positions and store them
                        left_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
                        right_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
                        left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
                        right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]

                        joint_positions["left_ankle"].append((left_ankle.x, left_ankle.y))
                        joint_positions["right_ankle"].append((right_ankle.x, right_ankle.y))
                        joint_positions["left_hip"].append((left_hip.x, left_hip.y))
                        joint_positions["right_hip"].append((right_hip.x, right_hip.y))

                        time_points.append(cap.get(cv2.CAP_PROP_POS_FRAMES) / fps)  # Store timestamp for frame

                    annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)  # Convert back to BGR for display
                    stframe.image(annotated_image_bgr, channels="BGR", use_container_width=True)  # Display frame in Streamlit

            cap.release()  # Release video capture
            stride_frequency = analyze_stride_performance(joint_positions, time_points)  # Analyze stride performance
            stride_frequencies.append(stride_frequency)  # Store stride frequency for this video

    # Classify running style based on stride frequency
    vertical_oscillation = 0.1  # Placeholder value for vertical oscillation (to be calculated)
    running_style = classify_running_style(stride_frequencies[-1], vertical_oscillation)

    st.markdown(f"### Running Style Classification: {running_style}")  # Display classification result

    # Combined visualization for all videos
    st.markdown("## Combined Classification Overview")
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = [f"Video {i+1}" for i in range(len(stride_frequencies))]

    ax.bar(labels, stride_frequencies, color=['red', 'green', 'blue'])  # Plot stride frequencies for each video
    ax.set_ylabel("Stride Frequency (strides/second)")
    ax.set_title("Stride Classification across Videos")
    ax.grid(True)

    st.pyplot(fig)  # Display combined plot in Streamlit
    # Display the stride and classification visuals
    st.markdown("### Stride and Classification Visuals")
    st.image("stride.png", caption="Stride Analysis Overview", use_container_width=200)
    st.image("classification.png", caption="Running Style Classification", use_container_width=200)
