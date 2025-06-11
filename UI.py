import tensorflow as tf
from tensorflow import keras
import streamlit as st
from PIL import Image
import cv2
import numpy as np

# Load model
model = keras.models.load_model('final_model.keras')
classes = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

def processing_image(image):
    # Convert PIL Image to numpy array and convert RGB to BGR
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) == 0:
        raise ValueError("No faces detected in the image")
    
    # Use only the first detected face
    x, y, w, h = faces[0]
    face_roi = image[y:y+h, x:x+w]
    
    final_image = cv2.resize(face_roi, (224, 224))
    final_image = np.expand_dims(final_image, axis=0)
    final_image = final_image / 255.0

    return final_image

def predict_emotion(image):
    try:
        processed = processing_image(image)
        prediction = model.predict(processed)
        return classes[np.argmax(prediction)]
    except Exception as e:
        st.error(f"Error in emotion detection: {str(e)}")
        return None

# Streamlit UI
st.title("🎭 Emotion Detection App")
tab1, tab2 = st.tabs(["📁 Upload Image", "📷 Live Camera"])

# Upload image tab
with tab1:
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            label = predict_emotion(image)
            if label:
                st.success(f"Predicted Emotion: **{label}**")
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

# Webcam tab
with tab2:
    run = st.checkbox("Start Camera", key="camera_checkbox")
    FRAME_WINDOW = st.empty()  # Placeholder for dynamic updates

    if run:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap = cv2.VideoCapture(1)  # Try secondary camera
        if not cap.isOpened():
            st.error("❌ Could not access any camera")
            run = False

        if run:
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            
            stop_button = st.button("🛑 Stop Camera", key="stop_camera_button")
            
            while run and not stop_button:
                ret, frame = cap.read()
                if not ret:
                    st.warning("⚠️ Failed to capture frame")
                    break

                # Face detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)

                for (x, y, w, h) in faces:
                    try:
                        # Process and predict
                        face_roi = frame[y:y+h, x:x+w]
                        face_roi = cv2.resize(face_roi, (224, 224))
                        face_input = np.expand_dims(face_roi / 255.0, axis=0)
                        pred_label = classes[np.argmax(model.predict(face_input))]
                        
                        # Draw UI
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        (text_width, text_height) = cv2.getTextSize(
                            pred_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                        )[0]
                        cv2.rectangle(
                            frame,
                            (x, y - text_height - 10),
                            (x + text_width + 10, y),
                            (255, 255, 255),
                            cv2.FILLED
                        )
                        cv2.putText(
                            frame, pred_label,
                            (x + 5, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 255), 2
                        )
                    except Exception as e:
                        st.warning(f"Face processing error: {str(e)}")

                # Display frame
                FRAME_WINDOW.image(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    use_container_width=True
                )

                if stop_button:
                    run = False

            cap.release()
            if not run:
                st.success("✅ Camera stopped")
    else:
        st.info("👆 Enable 'Start Camera' to begin")