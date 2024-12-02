import yt_dlp
import whisper
import streamlit as st
import groq  # Import Groq library
from google.oauth2 import service_account
import librosa
import numpy as np
import soundfile as sf
import os
import json
import subprocess

# Set up Groq API key
GROQ_API_KEY = st.secrets["groq_api"]["GROQ_API_KEY"]  # Replace with your actual Groq API key
client = groq.Client(api_key=GROQ_API_KEY)

# Set up Google Cloud credentials using Streamlit secrets for Google Drive
gdrive_credentials = st.secrets["gdrive_credentials"]


FFMPEG_PATH = "/usr/bin/ffmpeg"

def download_youtube_audio(url):
    """Downloads audio from a YouTube video."""
    options = {
        'format': 'bestaudio',
        'outtmpl': 'downloads/%(title)s.%(ext)s',
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav', 'preferredquality': '192'}],
        'ffmpeg_location': FFMPEG_PATH
    }
    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=True)
            return ydl.prepare_filename(info).replace('.webm', '.wav').replace('.m4a', '.wav')
    except Exception as e:
        st.error(f"Error downloading audio: {e}")
        return None

def sample_audio_segments(audio_path, segment_duration=10, skip_duration=20):
    """Samples audio by taking segments every skip_duration."""
    audio, sr = librosa.load(audio_path, sr=None)
    total_duration = librosa.get_duration(y=audio, sr=sr)
    
    samples = []
    for start in np.arange(0, total_duration, skip_duration):
        start_sample = int(start * sr)
        end_sample = int(min((start + segment_duration) * sr, len(audio)))
        samples.append(audio[start_sample:end_sample])
    
    sample_path = f"{audio_path}_sampled.wav"
    sf.write(sample_path, np.concatenate(samples), sr)
    return sample_path

def transcribe_audio_whisper(audio_path):
    """Transcribes sampled audio to text using Whisper tiny model."""
    model = whisper.load_model("tiny")  # Load Whisper tiny model for speed
    result = model.transcribe(audio_path)
    return result["text"]

def summarize_with_groq(text):
    """Summarize the transcribed text using Groq API."""
    prompt = f"""
    Please provide a detailed and structured summary of the following YouTube video transcription. Organize the summary into the following sections: 

    1. **Introduction**: Briefly describe the main topic and purpose of the video.
    2. **Key Points**: Outline the major points discussed, including any important arguments, examples, or data presented.
    3. **Conclusion**: Summarize the final thoughts or conclusions drawn in the video.
    4. **Takeaways**: Highlight any actionable insights or lessons learned from the video.

    Transcription:
    {text}
    """
    
    # Construct the Groq query
    query = f"""
    match {{
      "text": {{
        "contains": {{
          "query_string": "{text}",
          "field": "content"
        }}
      }}
    }}
    """
    
    try:
        # Execute the Groq query using Groq API
        response = client.execute(query)

        # Process the response from Groq to generate a summary
        summary = []
        for result in response:
            summary.append(result.get("text", ""))

        # Join the relevant content to form a summary
        summary_text = "\n".join(summary)
        return summary_text if summary_text else "No summary available."
    except Exception as e:
        st.error(f"Error generating summary: {e}")
        return "Error generating summary."

# Streamlit Interface
st.title("YouTube Video Summarizer")
url = st.text_input("Enter YouTube Video URL")

if st.button("Summarize Video"):
    if url:
        st.write("Downloading audio from YouTube...")
        audio_file_path = download_youtube_audio(url)

        if audio_file_path:
            st.write("Sampling audio for faster processing...")
            sampled_audio_path = sample_audio_segments(audio_file_path)

            st.write("Transcribing sampled audio to text...")
            transcribed_text = transcribe_audio_whisper(sampled_audio_path)

            if transcribed_text:
                st.write("Summarizing the text...")
                summary = summarize_with_groq(transcribed_text)

                if summary:
                    st.markdown("<h3>Summary:</h3><p>{}</p>".format(summary), unsafe_allow_html=True)

                    # Option to download the summary
                    st.download_button("Download Summary", summary, file_name="summary.txt")
    else:
        st.error("Please enter a valid YouTube URL.")
