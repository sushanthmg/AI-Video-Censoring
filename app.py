


import streamlit as st
import subprocess
import librosa
import noisereduce as nr
import soundfile as sf
import torch
import whisper
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from pydub import AudioSegment
from scipy.signal import butter, lfilter
import os
from tempfile import NamedTemporaryFile
import re

# Set page configuration
st.set_page_config(
    page_title="Alkimi AdCensor: AI for Adulterate Content Detection & Compliance",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Display logos in columns
col1, col2, col3 = st.columns([1, 6, 1])
with col1:
    st.image("alkimilogo.png", use_container_width=True)
with col3:
    st.image("kensaltensilogo.png", use_container_width=True)

# Title
st.title("Alkimi AdCensor: AI for Adulterate Content Detection & Compliance")

# Instruction below title
st.markdown("""
This tool extracts speech from video advertisements and detects explicit 18+ content.
It also classifies sentences as positive or negative based on sentiment analysis.
""")

# File uploader
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    with NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_video_path = temp_file.name

    # Process the video
    st.write("Processing video...")

    # =============== 1. Convert Video to Audio (MP3) ===============
    def convert_video_to_mp3(input_file, output_file="audio.mp3"):
        ffmpeg_cmd = [
            "ffmpeg",
            "-i", input_file,
            "-vn",
            "-acodec", "libmp3lame",
            "-ab", "192k",
            "-ar", "44100",
            "-y",
            output_file
        ]
        try:
            subprocess.run(ffmpeg_cmd, check=True)
            return output_file
        except subprocess.CalledProcessError:
            st.error("❌ Conversion failed!")
            return None

    # =============== 2. Audio Preprocessing (Denoising & Filtering) ===============
    def load_audio(file_path):
        y, sr = librosa.load(file_path, sr=None)
        return y, sr

    def noise_reduction(y, sr):
        noise_sample = y[:sr]  # First 1 second as noise profile
        reduced_noise = nr.reduce_noise(y=y, sr=sr, y_noise=noise_sample, prop_decrease=0.9)
        return reduced_noise

    def bandpass_filter(y, sr, lowcut=300, highcut=3400, order=5):
        nyquist = 0.5 * sr
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        filtered_audio = lfilter(b, a, y)
        return filtered_audio

    def amplify_audio(y):
        return y * 1.8  # Amplify audio by 1.8x

    def preprocess_audio(file_path, output_path="processed_audio.wav"):
        y, sr = load_audio(file_path)
        y_denoised = noise_reduction(y, sr)
        y_filtered = bandpass_filter(y_denoised, sr)
        y_amplified = amplify_audio(y_filtered)
        sf.write(output_path, y_amplified, sr)
        return output_path

    # =============== 3. Speech-to-Text Transcription ===============
    def transcribe_audio(audio_file):
        model = whisper.load_model("medium")
        result = model.transcribe(audio_file)
        return result["text"]

    # =============== 4. 18+ Content Detection & Sentiment Analysis ===============
    offensive_model_name = "cardiffnlp/twitter-roberta-base-offensive"
    offensive_tokenizer = AutoTokenizer.from_pretrained(offensive_model_name)
    offensive_model = AutoModelForSequenceClassification.from_pretrained(offensive_model_name)

    sentiment_pipeline = pipeline("sentiment-analysis")

    explicit_words = {"swearword1", "swearword2", "swearword3"}  # Add explicit words here

    def highlight_explicit_words(sentence):
        words = sentence.split()
        highlighted_sentence = " ".join([f'<span style="color:red; font-weight:bold;">{word}</span>' if word.lower() in explicit_words else word for word in words])
        return highlighted_sentence

    def classify_text(text):
        inputs = offensive_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = offensive_model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        labels = ["Non-Offensive", "Offensive"]
        pred_label = labels[torch.argmax(probs).item()]
        return pred_label, probs[0].tolist()

    def analyze_text(text):
        sentences = text.split('.')
        analyzed_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                offensive_label, offensive_confidence = classify_text(sentence)
                sentiment = sentiment_pipeline(sentence)[0]
                highlighted_sentence = highlight_explicit_words(sentence)
                analyzed_sentences.append({
                    "sentence": highlighted_sentence,
                    "offensive_label": offensive_label,
                    "offensive_confidence": offensive_confidence,
                    "sentiment": sentiment["label"]
                })
        return analyzed_sentences














