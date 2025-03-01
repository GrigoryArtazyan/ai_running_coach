import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import tempfile

# Initialize Mediapipe Pose Model
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Function to calculate angle between three points (for joint angles like knees)
def calculate_angle(a, b, c):
    ab = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    dot_product = np.dot(ab, bc)
    magnitude_ab = np.linalg.norm(ab)
    magnitude_bc = np.linalg.norm(bc)
    angle = np.arccos(dot_product / (magnitude_ab * magnitude_bc))
    return np.degrees(angle)

# Function to calculate upper body tilt angle
def calculate_upper_body_tilt(shoulder, hip):
    # Calculates torso alignment
    delta_y = shoulder[1] - hip[1]
    delta_x = shoulder[0] - hip[0]
    angle = np.degrees(np.arctan2(delta_y, delta_x))
    return angle

# Streamlit UI
st.title("Real-Time Running Posture Analysis")

# File uploader
uploaded_file = st.file_uploader("Upload a Running Video", type=["mp4", "mov", "avi", "mkv"])

if uploaded_file:
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_file.write(uploaded_file.read())
    video_path = temp_file.name

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    stframe = st.empty()

    stride_lengths = []
    step_times = []
    acceleration = []
    velocities = []
    tilts = []

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        prev_ankle_y = None
        prev_time = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            annotated_image = image.copy()

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                )

                left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
                left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
                left_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
                left_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]

                left_shoulder_coords = (left_shoulder.x * width, left_shoulder.y * height)
                left_hip_coords = (left_hip.x * width, left_hip.y * height)
                left_knee_coords = (left_knee.x * width, left_knee.y * height)
                left_ankle_coords = (left_ankle.x * width, left_ankle.y * height)

                knee_angle = calculate_angle(left_hip_coords, left_knee_coords, left_ankle_coords)
                tilt_angle = calculate_upper_body_tilt(left_shoulder_coords, left_hip_coords)
                tilts.append(tilt_angle)

                if prev_ankle_y is not None and prev_time is not None:
                    stride_length = abs(left_ankle_coords[1] - prev_ankle_y)
                    stride_lengths.append(stride_length)

                    time_diff = 1 / fps
                    velocity = (stride_length * 0.7) / time_diff  # Assuming 0.7 meters per pixel
                    velocities.append(velocity)
                    if len(velocities) > 1:
                        acc = (velocities[-1] - velocities[-2]) / time_diff
                        acceleration.append(acc)
                    velocities.append(velocity)

                prev_ankle_y = left_ankle_coords[1]
                prev_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000

                cv2.putText(annotated_image, f'Knee Angle: {knee_angle:.2f} degrees', (50, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(annotated_image, f'Tilt: {tilt_angle:.2f} degrees', (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
                stframe.image(annotated_image_bgr, channels="BGR", use_container_width=True)

    cap.release()

    # Data Visualization
    st.subheader("Data Visualizations")

    # Classification Logic
    avg_stride = np.mean(stride_lengths) if stride_lengths else 0
    avg_velocity = np.mean(velocities) if velocities else 0
    avg_acceleration = np.mean(acceleration) if acceleration else 0
    avg_tilt = np.mean(tilts) if tilts else 0

    # Classification
    if avg_velocity > 6 and avg_stride > 2.0 and avg_acceleration > 2.5:
        category = "Power Racer"
        description = "Profile with highest top speed, very common among middle-distance elite runners.\n" \
                      "Long, powerful strides with short ground contact.\n" \
                      "Stiff, springy legs yielding great elastic bounce.\n" \
                      "Large vertical peak force acting on the joints.\n" \
                      "Highest injury risk, tough on lower leg and calf due to extreme ankle joint power generation."
    elif avg_velocity > 5 and avg_stride > 1.8:
        category = "Eco Sprinter"
        description = "Active compact stride with great elastic bounce.\n" \
                      "Best compromise between speed capacity, running economy, and injury risk.\n" \
                      "Very common profile among elite 5-42k runners.\n" \
                      "Next lowest injury risk after Quick Steppers."
    elif avg_velocity > 4 and avg_tilt < 10:
        category = "Constant Glider"
        description = "Short strides with nearly constant ground contact.\n" \
                      "Mild joint loading but limited bounce that prevents fast running.\n" \
                      "Common among runners with limited flexibility elasticity or strength.\n" \
                      "Many elderly runners in this category.\n" \
                      "Slowest category but among the least injured."
    elif avg_velocity > 3.5 and avg_acceleration < 1.5:
        category = "Easy Strider"
        description = "Easygoing style, follows the law of least resistance.\n" \
                      "Most common profile among all six.\n" \
                      "Characterized by overstride and a tendency towards soft knee during support.\n" \
                      "Sensitive to different footwear.\n" \
                      "Low to average speed capacity and running economy at medium injury risk."
    elif avg_velocity < 3 and avg_stride < 1.5:
        category = "Long Strider"
        description = "Long leaping strides powered by large muscular force, decent sprint capabilities but poor economy."
    else:
        category = "Quick Stepper"
        description = "Rapid footwork with good elastic bounce, good economy up to 16 km/h, gentle loading with small joint angles."

    st.subheader(f"Classification: {category}")
    st.write(description)
    fig, axs = plt.subplots(4, 1, figsize=(10, 20))

    axs[0].plot(range(len(stride_lengths)), stride_lengths, label='Stride Length')
    axs[0].set_title('Stride Length per Step')
    axs[0].set_ylabel('Distance (pixels)')
    axs[0].legend()

    axs[1].plot(range(len(velocities)), velocities, label='Velocity', color='orange')
    axs[1].set_title('Velocity per Step')
    axs[1].set_ylabel('Velocity (pixels/sec)')
    axs[1].legend()

    axs[2].plot(tilts, label='Upper Body Tilt', color='green')
    axs[2].set_title('Upper Body Tilt Over Time')
    axs[2].set_ylabel('Tilt (degrees)')
    axs[2].legend()

    axs[3].plot(range(len(acceleration)), acceleration, label='Acceleration', color='red')
    axs[3].set_title('Acceleration per Step')
    axs[3].set_ylabel('Acceleration (m/s²)')
    axs[3].legend()

    st.pyplot(fig)
