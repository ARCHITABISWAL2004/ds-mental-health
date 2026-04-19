import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import speech_recognition as sr
from streamlit_mic_recorder import mic_recorder
import io
import requests

# ---------------- API CONFIG ----------------
API_URL = "https://cerevra.onrender.com/predict"

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
    - Model training
    - API deployment using Flask (Render)
    - Frontend using Streamlit
    - Emotion & Depression Detection
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

    st.subheader("📝 Text-Based Emotion Detection")

    text = st.text_area("Enter your text")

    if st.button("Predict Text"):

        if text.strip() == "":
            st.warning("Please enter some text")
        else:
            try:
                with st.spinner("Analyzing..."):
                    response = requests.post(API_URL, json={"text": text})
                    result = response.json()

                    emotion = result["prediction"]

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

            except:
                st.error("API request failed. Please check backend.")

    # ---------------- VOICE INPUT (UPLOAD) ----------------
    st.markdown("---")
    st.subheader("🎤 Voice-Based Emotion Detection (Upload)")

    audio_file = st.file_uploader("Upload a WAV audio file", type=["wav"])

    if audio_file is not None:
        recognizer = sr.Recognizer()

        with open("temp.wav", "wb") as f:
            f.write(audio_file.read())

        try:
            with sr.AudioFile("temp.wav") as source:
                audio = recognizer.record(source)
                voice_text = recognizer.recognize_google(audio)

                st.success(f"Recognized Text: {voice_text}")

                with st.spinner("Analyzing..."):
                    response = requests.post(API_URL, json={"text": voice_text})
                    result = response.json()

                    emotion = result["prediction"]

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

        except:
            st.error("Could not process audio. Try a clear WAV file.")

    # ---------------- LIVE MIC INPUT ----------------
    st.markdown("---")
    st.subheader("🎤 Live Microphone Input")

    audio = mic_recorder(start_prompt="Start Recording", stop_prompt="Stop Recording")

    if audio:
        st.audio(audio["bytes"])

        recognizer = sr.Recognizer()

        try:
            audio_bytes = io.BytesIO(audio["bytes"])

            with sr.AudioFile(audio_bytes) as source:
                audio_data = recognizer.record(source)
                voice_text = recognizer.recognize_google(audio_data)

                st.success(f"Recognized Text: {voice_text}")

                with st.spinner("Analyzing..."):
                    response = requests.post(API_URL, json={"text": voice_text})
                    result = response.json()

                    emotion = result["prediction"]

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

        except:
            st.error("Speech recognition failed. Try speaking clearly.")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("Developed as Mini Project | Data Science Pipeline")