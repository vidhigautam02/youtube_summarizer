import yt_dlp
import whisper
import streamlit as st
import google.generativeai as genai
from google.oauth2 import service_account
import librosa
import numpy as np
import soundfile as sf
import os
import json

# Configure genai with Google API key for Gemini model using Streamlit secrets
GOOGLE_API_KEY = st.secrets["google_api"]["GOOGLE_API_KEY"]
genai.configure(api_key=GOOGLE_API_KEY)

# Set up generation configuration for Google Gemini
generation_config = {
    "temperature": 0.1,
    "max_output_tokens": 1200,
}

# Set up Google Cloud credentials using Streamlit secrets
gdrive_credentials = st.secrets["gdrive_credentials"]

    # Extract values from the secrets
    credentials_info = {
        "type": gdrive_credentials["type"],
        "project_id": gdrive_credentials["project_id"],
        "private_key_id": gdrive_credentials["private_key_id"],
        "private_key": gdrive_credentials["private_key"].replace("\\n", "\n"),  # Replace \n with actual new line
        "client_email": gdrive_credentials["client_email"],
        "client_id": gdrive_credentials["client_id"],
        "auth_uri": gdrive_credentials["auth_uri"],
        "token_uri": gdrive_credentials["token_uri"],
        "auth_provider_x509_cert_url": gdrive_credentials["auth_provider_x509_cert_url"],
        "client_x509_cert_url": gdrive_credentials["client_x509_cert_url"],
    }

    # Authenticate with Google Drive API using the credentials
    credentials = service_account.Credentials.from_service_account_info(credentials_info)

# Define the path for FFmpeg from Streamlit secrets
FFMPEG_PATH = 'ffmpeg'  # No path needed for Streamlit Cloud or Linux servers


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

def summarize_with_gemini(text):
    prompt = f"""
    Please provide a detailed and structured summary of the following YouTube video transcription. Organize the summary into the following sections: 

    1. **Introduction**: Briefly describe the main topic and purpose of the video.
    2. **Key Points**: Outline the major points discussed, including any important arguments, examples, or data presented.
    3. **Conclusion**: Summarize the final thoughts or conclusions drawn in the video.
    4. **Takeaways**: Highlight any actionable insights or lessons learned from the video.

    Transcription:
{text}
    """
    try:
        model = genai.GenerativeModel(model_name="gemini-1.5-flash", generation_config=generation_config)
        response = model.generate_content(prompt)
        
        # Extract and return the summary
        if hasattr(response, 'candidates') and len(response.candidates) > 0:
            return response.candidates[0].content.parts[0].text
        else:
            return "No summary available."
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
                summary = summarize_with_gemini(transcribed_text)

                if summary:
                    st.markdown("<h3>Summary:</h3><p>{}</p>".format(summary), unsafe_allow_html=True)

                    # Option to download the summary
                    st.download_button("Download Summary", summary, file_name="summary.txt")
    else:
        st.error("Please enter a valid YouTube URL.")
