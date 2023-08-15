import cv2
import dlib
from scipy.spatial import distance as dist
from imutils import face_utils

# Load the face detector and facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

EAR_THRESHOLD = 0.25  # Adjust this threshold as needed
CONSECUTIVE_FRAMES = 20

# Initialize frame counters and state
frames_counter = 0
drowsy_frames = 0
alarm_on = False

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray)

    for face in faces:
        # Get facial landmarks
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        # Extract eye coordinates
        left_eye = shape[42:48]
        right_eye = shape[36:42]

        # Calculate EAR for both eyes
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0

        # Draw eye contours and calculate EAR threshold
        # ...

        # Check for drowsiness
        if avg_ear < EAR_THRESHOLD:
            drowsy_frames += 1
            if drowsy_frames >= CONSECUTIVE_FRAMES:
                if not alarm_on:
                    alarm_on = True
                    # Start an alarm or notification (e.g., sound, message)
        else:
            drowsy_frames = 0
            alarm_on = False

    # Display the frame
    cv2.imshow("Drowsiness Detector", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
