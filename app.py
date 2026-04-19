import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import speech_recognition as sr
from streamlit_mic_recorder import mic_recorder
import speech_recognition as sr
import io
# ---------------- LOAD MODELS ----------------
emotion_model = pickle.load(open("model/emotion_model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Mental Health Detection", layout="wide")

st.title("🧠 Mental Health & Emotion Detection System")

# ---------------- SIDEBAR ----------------
menu = st.sidebar.selectbox("Menu", ["Home", "EDA & Analytics", "Prediction"])

# ---------------- HOME ----------------
if menu == "Home":
    st.subheader("Project Overview")
    st.write("""
    This is an end-to-end Data Science project that includes:
    - Data preprocessing
    - Outlier handling
    - Model training (3 algorithms)
    - Hyperparameter tuning
    - Model comparison
    - Visualization
    - Deployment using Streamlit
    """)

# ---------------- EDA ----------------
elif menu == "EDA & Analytics":

    st.subheader("Dataset Analysis")

    try:
        df = pd.read_csv("data/emotion.csv")

        st.write("### Dataset Preview")
        st.write(df.head())

        st.write("### Descriptive Statistics")
        st.write(df.describe())

        st.write("### Emotion Distribution")
        st.bar_chart(df['label'].value_counts())

        # Text length visualization
        df['length'] = df['text'].apply(len)

        st.write("### Text Length Distribution")
        fig, ax = plt.subplots()
        ax.hist(df['length'], bins=20)
        st.pyplot(fig)

    except:
        st.error("Dataset not found. Please place emotion.csv inside data folder.")

# ---------------- PREDICTION ----------------
elif menu == "Prediction":

    st.subheader("Text-Based Emotion Detection")

    text = st.text_area("Enter your text")

    if st.button("Predict"):

        if text.strip() == "":
            st.warning("Please enter some text")
        else:
            # Transform input
            vec = vectorizer.transform([text])

            # Emotion prediction
            emotion = emotion_model.predict(vec)[0]

            # Rule-based depression mapping
            depression_map = {
                "joy": "Low",
                "love": "Low",
                "sadness": "High",
                "anger": "Medium",
                "fear": "High",
                "surprise": "Medium"
            }

            depression = depression_map.get(emotion, "Unknown")

            # Display results
            st.success(f"Emotion: {emotion}")
            st.warning(f"Depression Level: {depression}")

    # ---------------- VOICE INPUT ----------------
    st.markdown("---")
    st.subheader("🎤 Voice-Based Emotion Detection")

    audio_file = st.file_uploader("Upload a WAV audio file", type=["wav"])

    if audio_file is not None:
        import speech_recognition as sr

        recognizer = sr.Recognizer()

        # Save uploaded file temporarily
        with open("temp.wav", "wb") as f:
            f.write(audio_file.read())

        try:
            with sr.AudioFile("temp.wav") as source:
                audio = recognizer.record(source)
                voice_text = recognizer.recognize_google(audio)

                st.success(f"Recognized Text: {voice_text}")

                # Transform input
                vec = vectorizer.transform([voice_text])

                # Emotion prediction
                emotion = emotion_model.predict(vec)[0]

                # Same depression mapping
                depression_map = {
                    "joy": "Low",
                    "love": "Low",
                    "sadness": "High",
                    "anger": "Medium",
                    "fear": "High",
                    "surprise": "Medium"
                }

                depression = depression_map.get(emotion, "Unknown")

                # Display results
                st.success(f"Emotion: {emotion}")
                st.warning(f"Depression Level: {depression}")

        except Exception as e:
            st.error("Could not process audio. Please try a clear WAV file.")
    # ---------------- LIVE MIC INPUT ----------------
    st.markdown("---")
    st.subheader("🎤 Live Microphone Input")

    audio = mic_recorder(start_prompt="Start Recording", stop_prompt="Stop Recording")

    if audio:
        st.audio(audio["bytes"])

        recognizer = sr.Recognizer()

        try:
            # Convert bytes to AudioFile
            audio_bytes = io.BytesIO(audio["bytes"])

            with sr.AudioFile(audio_bytes) as source:
                audio_data = recognizer.record(source)
                voice_text = recognizer.recognize_google(audio_data)

                st.success(f"Recognized Text: {voice_text}")

                # ---- PREDICTION ----
                vec = vectorizer.transform([voice_text])
                emotion = emotion_model.predict(vec)[0]

                depression_map = {
                    "joy": "Low",
                    "love": "Low",
                    "sadness": "High",
                    "anger": "Medium",
                    "fear": "High",
                    "surprise": "Medium"
                }

                depression = depression_map.get(emotion, "Unknown")

                st.success(f"Emotion: {emotion}")
                st.warning(f"Depression Level: {depression}")

        except Exception as e:
            st.error("Speech recognition failed. Try speaking clearly.")
# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("Developed as Mini Project | Data Science Pipeline")