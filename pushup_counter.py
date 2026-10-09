import cv2
import mediapipe as mp
import numpy as np
import math

# Initialize mediapipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

# Angle calculation function
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
              np.arctan2(a[1]-b[1], a[0]-b[0])

    angle = abs(radians * 180.0 / math.pi)
    if angle > 180:
        angle = 360 - angle

    return angle

# Video capture
cap = cv2.VideoCapture(0)

counter = 0
stage = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(img_rgb)

    if result.pose_landmarks:
        lm = result.pose_landmarks.landmark

        shoulder = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                    lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y]

        elbow = [lm[mp_pose.PoseLandmark.LEFT_ELBOW].x,
                 lm[mp_pose.PoseLandmark.LEFT_ELBOW].y]

        wrist = [lm[mp_pose.PoseLandmark.LEFT_WRIST].x,
                 lm[mp_pose.PoseLandmark.LEFT_WRIST].y]

        angle = calculate_angle(shoulder, elbow, wrist)

        # Push-up logic
        if angle > 160:
            stage = "UP"
        if angle < 90 and stage == "UP":
            stage = "DOWN"
            counter += 1

        # Draw angle
        h, w, _ = frame.shape
        elbow_pixel = tuple(np.multiply(elbow, [w, h]).astype(int))
        cv2.putText(frame, str(int(angle)),
                    elbow_pixel,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

        mp_draw.draw_landmarks(
            frame,
            result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    # Display counter
    cv2.rectangle(frame, (0, 0), (300, 80), (0, 0, 0), -1)
    cv2.putText(frame, f'Push Ups: {counter}',
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 255, 0), 2)

    cv2.putText(frame, f'Stage: {stage}',
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (255, 255, 255), 2)

    cv2.imshow("AI Push-Up Counter", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
