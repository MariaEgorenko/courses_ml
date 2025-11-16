import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

def get_coordinates(landmarks, landmark_name):
    return [
        landmarks[mp_pose.PoseLandmark[landmark_name].value].x,
        landmarks[mp_pose.PoseLandmark[landmark_name].value].y
    ]

def good_posture_side_view(landmarks):
    ear_left = get_coordinates(landmarks, 'LEFT_EAR')
    shoulder_left = get_coordinates(landmarks, 'LEFT_SHOULDER')
    hip_left = get_coordinates(landmarks, 'LEFT_HIP')

    ear_right = get_coordinates(landmarks, 'RIGHT_EAR')
    shoulder_gight = get_coordinates(landmarks, 'RIGHT_SHOULDER')
    hip_right = get_coordinates(landmarks, 'RIGHT_HIP')

    back_angle_left = calculate_angle(ear_left, shoulder_left, hip_left)
    back_angle_right = calculate_angle(ear_right, shoulder_gight, hip_right)

    return back_angle_left > 160 and back_angle_right > 160

def good_posture_front_view(landmarks):
    left_shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
    right_shoulder_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
    left_hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
    right_hip_y = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y

    shoulder_diff = abs(left_shoulder_y - right_shoulder_y)
    hip_diff = abs(left_hip_y - right_hip_y)

    threshold = 0.035

    return not (shoulder_diff > threshold or hip_diff > threshold)

cap = cv2.VideoCapture(0)

current_view = 'front'

if not cap.isOpened():
    print('\033[31mНе удалось получить доступ к камере...\033[0m')
    exit()

while True:
    ret, img = cap.read()

    if not ret:
        print('\033[31mНе удалось прочитать кадр...\033[0m')
        break

    height, width, _ = img.shape

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    res = pose.process(img_rgb)

    if res.pose_landmarks:
        mp_drawing.draw_landmarks(
            img, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        landmarks = res.pose_landmarks.landmark

        if current_view == 'side':  # side
            status = good_posture_side_view(landmarks)
        else: # front
            status = good_posture_front_view(landmarks)

        posture_status = 'good' if status else 'bad'

        cv2.putText(img, f'View Mode: {current_view}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        color = (0, 255, 0) if status else (0, 0, 255)
        cv2.putText(img, f"status: {posture_status}",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    cv2.imshow('video', img)

    key = cv2.waitKey(30) & 0xff
    if key == 27:
        break
    elif key == ord('s'):
        current_view = 'side'
    elif key == ord('f'):
        current_view = 'front'

cap.release()
cv2.destroyAllWindows()