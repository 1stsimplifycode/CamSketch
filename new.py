import cv2 as cv
import mediapipe as mp

class HandDetection:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mphands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mphands.Hands(min_detection_confidence=min_detection_confidence, min_tracking_confidence=min_tracking_confidence)
        self.points = []  # List to store points of the index finger tip
        self.erase_trigger_distance = 100  # Distance threshold to activate eraser (can be adjusted)

    def detect_and_draw(self, frame):
        # Convert BGR image to RGB for MediaPipe processing
        RGBimg = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        result = self.hands.process(RGBimg)

        # If hands are detected
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Get the index finger tip (landmark 8)
                index_tip = hand_landmarks.landmark[self.mphands.HandLandmark.INDEX_FINGER_TIP]
                x, y = int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0])

                # Check if the palm area is detected (using palm base landmark 0)
                palm_base = hand_landmarks.landmark[self.mphands.HandLandmark.WRIST]
                palm_x, palm_y = int(palm_base.x * frame.shape[1]), int(palm_base.y * frame.shape[0])

                # Calculate the distance between the palm and index finger tip
                distance = ((x - palm_x) ** 2 + (y - palm_y) ** 2) ** 0.5

                # If the distance is below a threshold, treat the hand as an "eraser" and clear the drawing
                if distance < self.erase_trigger_distance:
                    self.points = []  # Clear points if palm is detected
                    cv.putText(frame, "Erasing...", (100, 100), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

                # Add the current index finger tip position to the points list
                else:
                    self.points.append((x, y))

                # Draw lines connecting all the previous points
                if len(self.points) > 1:
                    for i in range(1, len(self.points)):
                        cv.line(frame, self.points[i-1], self.points[i], (0, 0, 255), 5)  # Red line for drawing

        return frame

def main():
    cap = cv.VideoCapture(0)  # Capture video from webcam (index 0)
    if not cap.isOpened():
        print("Error: Couldn't access the camera.")
        return

    detector = HandDetection()

    while True:
        success, frame = cap.read()
        if not success:
            print("Error: Failed to capture frame.")
            break
        
        # Detect hands and draw based on index finger tip movement
        frame = detector.detect_and_draw(frame)

        # Display the processed frame
        cv.imshow("Tracking Index Finger Tip - Continuous Drawing", frame)

        # Exit the loop when 'd' is pressed
        if cv.waitKey(1) & 0xFF == ord('d'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()

