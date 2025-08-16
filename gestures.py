import cv2
import mediapipe as mp
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Initialize Video Capture
cap = cv2.VideoCapture(0)

# Initialize Metrics
y_true = []  # True labels
y_pred = []  # Predicted labels
frame_times = []  # List to store the processing time of each frame

# Placeholder function to get true labels
def get_true_label():
    # Implement your logic to get true labels
    return 'class1'  # Example placeholder value

# Placeholder function to get predicted labels
def get_predicted_label(results):
    # Implement your logic to get predicted labels based on results
    if results.multi_hand_landmarks:
        return 'class1'  # Example placeholder value
    else:
        return 'class0'  # Example placeholder value

# Variable to store the previous gesture
prev_gesture = None

# Capture and Process Each Frame
while cap.isOpened():
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Get true and predicted labels
    true_label = get_true_label()
    predicted_label = get_predicted_label(results)

    # Check for gesture change
    if prev_gesture is None or prev_gesture != predicted_label:
        prev_gesture = predicted_label
        y_true.append(true_label)
        y_pred.append(predicted_label)
        # Calculate Metrics with zero_division parameter
        if frame_times:
            average_frame_time = sum(frame_times) / len(frame_times)
        else:
            average_frame_time = 0

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1-Score: {f1}")
        print(f"Average Frame Processing Time: {average_frame_time} seconds")

        # Comparison with State-of-the-Art (Placeholder for actual comparison)
        state_of_the_art_metrics = {
            'Accuracy': 0.95,
            'Precision': 0.94,
            'Recall': 0.93,
            'F1-Score': 0.94,
        }

        print("Comparison with State-of-the-Art:")
        for metric, value in state_of_the_art_metrics.items():
            print(f"{metric}: {value}")
    
    end_time = time.time()
    frame_times.append(end_time - start_time)

    # Display the Frame
    cv2.imshow('Hand Gesture Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release Resources
cap.release()
cv2.destroyAllWindows()
