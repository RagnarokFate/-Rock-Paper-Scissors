import cv2
import mediapipe as mp
import numpy as np

def detect_hand_gesture():
    # Initialize mediapipe hands module
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()

    # Initialize video capture
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with Mediapipe hands
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract hand landmarks
                landmarks = [(int(point.x * frame.shape[1]), int(point.y * frame.shape[0])) for point in
                             hand_landmarks.landmark]

                # Draw a rectangle around the hand
                x, y, w, h = cv2.boundingRect(np.array(landmarks))
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Detect rock, paper, scissors
                gesture = detect_gesture(landmarks)
                if gesture:
                    cv2.putText(frame, f"Gesture: {gesture}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                                cv2.LINE_AA)

                # Calculate accuracy factor (based on detection confidence)
                accuracy_factor = results.multi_handedness[0].classification[0].score
                accuracy_text = f"Accuracy: {int(accuracy_factor * 100)}%"
                cv2.putText(frame, accuracy_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)


        cv2.imshow("Hand Gesture Detection", frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()


def detect_gesture(landmarks):
    if landmarks[4][1] < landmarks[8][1] and landmarks[4][1] < landmarks[12][1]:
        return "Rock"
    elif landmarks[8][1] < landmarks[6][1] and landmarks[8][1] < landmarks[10][1]:
        return "Paper"
    elif landmarks[6][1] < landmarks[8][1] and landmarks[10][1] < landmarks[8][1]:
        return "Scissors"
    else:
        return None


if __name__ == "__main__":
    detect_hand_gesture()

