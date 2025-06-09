import streamlit as st
import requests
import io

API_url="http://localhost:8000/analyze/"

st.title("Multimodal Emotion Detection")

audio_bytes=st.audio_input("Record your audio")

if audio_bytes is not None:
    st.audio(audio_bytes, format="audio/wav")

    if st.button("Analyze Recorded Audio"):
        files={"file": ("recorded_audio.wav", audio_bytes, "audio/wav")}
        with st.spinner("Analyzing"):
            response=requests.post(API_url, files=files)
        if response.status_code==200:
            data=response.json()
            st.subheader("Transcription")
            st.write(data["transcription"])
            st.subheader("Predictions")
            st.write(f"Audio model prediction: {data['audio predicted emotion']}")
            st.write(f"Text model prediction: {data['text predicted emotion']}")
            st.write(f"Fused final prediction: {data['final prediction']}")
        else:
            st.error(f"API Error: {response.status_code}-{response.text}")

uploaded_file=st.file_uploader("Or upload an audio file", type=["wav"])
if uploaded_file is not None:
    files={"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
    if st.button("Analyze Uploaded Audio"):
        with st.spinner("Analyzing"):
            response=requests.post(API_url, files=files)
        if response.status_code==200:
            data=response.json()
            st.subheader("Transcription")
            st.write(data["transcription"])
            st.subheader("Predictions")
            st.write(f"Audio model prediction: {data['audio predicted emotion']}")
            st.write(f"Text model prediction: {data['text predicted emotion']}")
            st.write(f"Fused final prediction: {data['final prediction']}")
        else:
            st.error(f"API Error: {response.status_code}-{response.text}")
    