import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from PIL import Image
import time
import pandas as pd
import altair as alt

# -----------------------------------------------------------------------------
# 1. Configuration & Setup
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Advanced Emotion Detector", layout="wide")

# Custom CSS for "Advanced" look
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
    }
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    h1, h2, h3 {
        color: #00d4ff !important;
        font-family: 'Segoe UI', sans-serif;
    }
    .stButton>button {
        background-color: #00d4ff;
        color: #000000;
        border-radius: 8px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #00a3cc;
        color: #ffffff;
    }
    /* Metrics styling */
    div[data-testid="metric-container"] {
        background-color: #1f2937;
        padding: 10px;
        border-radius: 8px;
        border: 1px solid #374151;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Model & Face Detector Loading
# -----------------------------------------------------------------------------

@st.cache_resource
def load_face_cascade():
    # Load Haar Cascade for face detection
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    return cv2.CascadeClassifier(cascade_path)

@st.cache_resource
def load_emotion_model():
    # Reconstruct the model architecture from the notebook
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.22))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))
    
    # Load weights
    try:
        model.load_weights('emotion_detection_model.h5')
        return model
    except Exception as e:
        st.error(f"Failed to load model weights: {e}")
        return None

face_cascade = load_face_cascade()
model = load_emotion_model()

# Emotion labels (FER2013 standard)
EMOTIONS = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}

EMOTION_EMOJIS = {
    "Angry": "😠",
    "Disgust": "🤢",
    "Fear": "😨",
    "Happy": "😄",
    "Sad": "😢",
    "Surprise": "😲",
    "Neutral": "😐"
}

# -----------------------------------------------------------------------------
# 3. UI Layout
# -----------------------------------------------------------------------------

st.title("🧠 Advanced Emotion Recognition")
st.markdown("Powered by a custom CNN model trained on FER2013.")

col1, col2 = st.columns([2, 1])

with col2:
    st.subheader("⚙️ Controls & Analytics")
    run_detection = st.checkbox("Start Camera", value=False)
    show_probs = st.checkbox("Show Probability Distribution", value=True)
    show_history = st.checkbox("Show Emotion History", value=True)
    
    st.markdown("---")
    current_emotion_placeholder = st.empty()
    prob_chart_placeholder = st.empty()
    
    st.markdown("### Session Stats")
    stats_placeholder = st.empty()

with col1:
    st.subheader("📷 Live Feed")
    video_placeholder = st.empty()

# -----------------------------------------------------------------------------
# 4. Main Loop
# -----------------------------------------------------------------------------

if run_detection:
    cap = cv2.VideoCapture(0)
    
    # History buffers
    emotion_history = []
    MAX_HISTORY = 100
    
    while run_detection:
        ret, frame = cap.read()
        if not ret:
            st.warning("Could not read from webcam.")
            break
            
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        dominant_emotion = "Neutral" # Default
        probs = np.zeros(7)
        
        for (x, y, w, h) in faces:
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 212, 255), 2)
            
            # Preprocess face for model
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi_pixels = roi_gray.astype('float32') / 255.0
            roi_pixels = np.expand_dims(roi_pixels, axis=0)
            roi_pixels = np.expand_dims(roi_pixels, axis=-1)
            
            # Predict
            if model:
                predictions = model.predict(roi_pixels, verbose=0)
                probs = predictions[0]
                max_index = np.argmax(probs)
                dominant_emotion = EMOTIONS[max_index]
                
                # Overlay text
                label = f"{dominant_emotion} ({probs[max_index]*100:.1f}%)"
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 212, 255), 2)
        
        # Update UI Components
        
        # 1. Video Feed
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
        
        # 2. Current Emotion Metric
        emoji = EMOTION_EMOJIS.get(dominant_emotion, "")
        current_emotion_placeholder.metric("Dominant Emotion", f"{dominant_emotion} {emoji}")
        
        # 3. Probability Chart
        if show_probs:
            df_probs = pd.DataFrame({
                'Emotion': list(EMOTIONS.values()),
                'Probability': probs
            })
            
            chart = alt.Chart(df_probs).mark_bar().encode(
                x=alt.X('Probability', scale=alt.Scale(domain=[0, 1])),
                y=alt.Y('Emotion', sort=None),
                color=alt.condition(
                    alt.datum.Probability == probs.max(),
                    alt.value('#00d4ff'),
                    alt.value('#374151')
                ),
                tooltip=['Emotion', alt.Tooltip('Probability', format='.1%')]
            ).properties(height=250)
            
            prob_chart_placeholder.altair_chart(chart, use_container_width=True)
            
        # 4. History (Optional - simplified for performance)
        if show_history:
            emotion_history.append(dominant_emotion)
            if len(emotion_history) > MAX_HISTORY:
                emotion_history.pop(0)
            
            # Count occurrences in history
            counts = pd.Series(emotion_history).value_counts()
            stats_placeholder.bar_chart(counts)

        # Stop button logic (handled by checkbox state, but good to check)
        if not run_detection:
            break
            
        # Small sleep to reduce CPU usage
        # time.sleep(0.01) 
        
    cap.release()
else:
    st.info("Check 'Start Camera' to begin.")
