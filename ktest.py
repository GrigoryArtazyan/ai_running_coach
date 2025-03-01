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
from streamlit_vertical_slider import vertical_slider
from streamlit_option_menu import option_menu

import menu_bar


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

# Function to calculate horizontal displacement of joints
def calculate_horizontal_displacement(joint_positions):
    return [position[0] for position in joint_positions]  # Extract x-coordinates (horizontal displacement)

# Function to calculate speed of joints
def calculate_joint_speed(joint_positions, time_points):
    speeds = []
    for i in range(1, len(joint_positions)):
        distance = np.linalg.norm(np.array(joint_positions[i]) - np.array(joint_positions[i-1]))
        time_diff = time_points[i] - time_points[i-1]
        speeds.append(distance / time_diff)
    return speeds

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

# Function to analyze stride performance (displacement, frequency, stride length, and speed)
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
    st.pyplot(plt)
    st.markdown("The vertical displacement of the ankles shows the up and down movement of the feet during running. "
                "Higher peaks indicate more vertical movement, which can affect running efficiency and injury risk.")

    if time_points and joint_positions["left_ankle"]:
        stride_frequency = calculate_stride_frequency(left_ankle_displacement + right_ankle_displacement, time_points)
        st.write(f"Stride Frequency: {stride_frequency:.2f} strides per second")  # Display stride frequency
    else:
        st.write("Stride Frequency: Not enough data to calculate stride frequency")  # Display message if not enough data

    # Plot vertical body displacement (average of hips)
    vertical_displacement_body = calculate_vertical_displacement_of_body(joint_positions)
    plt.figure(figsize=(6, 4))
    plt.plot(time_points, vertical_displacement_body, label="Body Vertical Displacement", color="g")
    plt.xlabel('Time (s)')
    plt.ylabel('Vertical Displacement (m)')
    plt.title('Vertical Displacement of Body')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)
    st.markdown("The vertical displacement of the body shows the overall up and down movement of the runner's center of mass. "
                "Consistent vertical displacement indicates a stable running form, while large variations may suggest inefficiencies.")

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
    st.markdown("Stride length measures the distance covered in one stride. Longer stride lengths can indicate more powerful running, "
                "but may also increase the risk of overstriding and injury.")

    # Calculate joint speed and plot it
    left_ankle_speed = calculate_joint_speed(left_ankle_positions, time_points)
    right_ankle_speed = calculate_joint_speed(right_ankle_positions, time_points)
    
    plt.figure(figsize=(6, 4))
    plt.plot(time_points[1:], left_ankle_speed, label="Left Ankle Speed")
    plt.plot(time_points[1:], right_ankle_speed, label="Right Ankle Speed")
    plt.xlabel('Time (s)')
    plt.ylabel('Speed (m/s)')
    plt.title('Joint Speed Over Time')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)
    st.markdown("Joint speed shows the velocity of the ankles during running. Higher speeds indicate faster running, "
                "but may also increase the risk of injury if not controlled properly.")

    return stride_frequency  # Return calculated stride frequency

# Function to classify running style based on stride frequency and vertical oscillation
def classify_running_style(stride_frequency, vertical_oscillation):
    if stride_frequency > 2.5 and vertical_oscillation > 0.08:
        return "Power Racer", "Profile with highest top speed, very common among middle-distance elite runners.\n" \
                              "Long, powerful strides with short ground contact.\n" \
                              "Stiff, springy legs yielding great elastic bounce.\n" \
                              "Large vertical peak force acting on the joints.\n" \
                              "Highest injury risk, tough on lower leg and calf due to extreme ankle joint power generation."
    elif stride_frequency > 2.0 and vertical_oscillation <= 0.08:
        return "Eco Sprinter", "Active compact stride with great elastic bounce.\n" \
                               "Best compromise between speed capacity, running economy, and injury risk.\n" \
                               "Very common profile among elite 5-42k runners.\n" \
                               "Next lowest injury risk after Quick Steppers."
    elif stride_frequency < 1.5:
        return "Easy Strider", "Easygoing style, follows the law of least resistance.\n" \
                               "Most common profile among all six.\n" \
                               "Characterized by overstride and a tendency towards soft knee during support.\n" \
                               "Sensitive to different footwear.\n" \
                               "Low to average speed capacity and running economy at medium injury risk."
    else:
        return "Constant Glider", "Short strides with nearly constant ground contact.\n" \
                                  "Characterized by low vertical oscillation and high stride frequency.\n" \
                                  "Efficient running style with moderate speed capacity and low injury risk.\n" \
                                  "Common among long-distance runners."

# Streamlit UI
st.title("Welcome To Your AI Running Coach")

st.image("images/welcome_illustrations.jpg", caption="Stride Analysis and Pose Estimation", use_container_width=True)

st.markdown("### Recording Instructions:")
col1, col2 = st.columns([1, 2])
with col1:
    st.image("images/recording_instructions.jpg", caption="Recording Instructions", use_container_width=True)
with col2:
    st.markdown("""
    - **Set up your phone and let it lean against an object (e.g. bench/tree) so that your full body is visible.**
    - **If possible, step several distance away from the camera and run in front of it (or run towards the camera) .**
    - **Keep the camera steady and at a consistent height.**
    - **Check out ways to record yourself on the phone here [Sample Video](https://www.tiktok.com/@drvizuals/video/7380753416466861317?is_from_webapp=1&sender_device=pc&web_id=7446040220871689761).**
    """)



col1, col2 = st.columns([1, 2])
with col1:
    joint_color = st.color_picker("Pick a color for joints", "#FF0000")  # Color picker for joint visualization

    # Vertical slider for user to reflect on their average running pace

    average_pace = vertical_slider(
        label="Reflect on your average running pace (min/km)",
        key="vert_01",
        height=300,
        thumb_shape="square",
        step=0.5,
        default_value=7,
        min_value=2,
        max_value=12,
        track_color="blue",
        slider_color=('red', 'blue'),
        thumb_color="orange",
        value_always_visible=True,
    )
    st.write(f"Average running pace: {average_pace} min/km")
    selected_color = tuple(int(joint_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))  # Convert color to RGB

with col2:
    st.markdown("### Upload your running videos:")
    st.markdown("""
    - **Suggest length of 10-20 seconds**
    - **Suggest size 5-10 MB preferably**
    - Formats: **MP4, MOV, AVI, MKV, MPEG4**
    """)

    # Upload video files
    uploaded_file_C = st.file_uploader("Upload Center Angle Video", type=["mp4", "mov", "avi", "mkv"])
    uploaded_file_LR = st.file_uploader("Upload Left/Right Angle Video", type=["mp4", "mov", "avi", "mkv"])

    uploaded_files = [uploaded_file_C, uploaded_file_LR]    



if any(uploaded_files):  # If any video is uploaded
    stride_frequencies = []  # List to store stride frequencies for each video
    for index, uploaded_file in enumerate(uploaded_files):
        if uploaded_file:
            st.markdown(f"### Processing Your Video /// Please Wait...")  # Display processing message
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
    running_style, running_style_description = classify_running_style(stride_frequencies[-1], vertical_oscillation)

    st.markdown(f"### Running Style Classification: {running_style}")  # Display classification result
    st.markdown(running_style_description)  # Display classification description
    st.image("images/big_six.png", caption="Running Style Classification", width=600)  # Display classification image
