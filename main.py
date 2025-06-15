import cv2
from deepface import DeepFace
import csv
from datetime import datetime

# Emoji mapping
emotion_emojis = {
    "angry": "😠",
    "disgust": "🤢",
    "fear": "😨",
    "happy": "😄",
    "sad": "😢",
    "surprise": "😲",
    "neutral": "😐"
}

# Open webcam
cap = cv2.VideoCapture(0)
print("Starting camera... Press 'q' to quit.")

# Create CSV file to log emotions
csv_file = open("emotion_log.csv", mode="w", newline="", encoding="utf-8")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["timestamp", "angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        # Analyze emotions
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotions = result[0]['emotion']
        dominant_emotion = result[0]['dominant_emotion']
        emoji = emotion_emojis.get(dominant_emotion, "")

        # Save to CSV
        csv_writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            round(emotions["angry"], 2),
            round(emotions["disgust"], 2),
            round(emotions["fear"], 2),
            round(emotions["happy"], 2),
            round(emotions["sad"], 2),
            round(emotions["surprise"], 2),
            round(emotions["neutral"], 2)
        ])

        # Display only dominant emotion + emoji
        cv2.putText(frame, f'{dominant_emotion} {emoji}', (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)

    except:
        cv2.putText(frame, "No face detected", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Show webcam feed
    cv2.imshow('Real-Time Emotion Detector', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean-up
cap.release()
csv_file.close()
cv2.destroyAllWindows()
