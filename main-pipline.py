import os
import requests
import numpy as np
import joblib
import librosa
from urllib.parse import urlparse, parse_qs
from moviepy.editor import VideoFileClip
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import yt_dlp
import whisper
from transformers import pipeline
import tkinter as tk
from tkinter import scrolledtext, messagebox

# === Helper Functions ===

def is_youtube_url(url):
    parsed_url = urlparse(url)
    return any(domain in parsed_url.netloc for domain in ["youtube.com", "youtu.be"])

def generate_folder_name(url):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if is_youtube_url(url):
        video_id = ""
        if "youtu.be" in url:
            video_id = url.split("/")[-1].split("?")[0]
        elif "youtube.com" in url:
            qs = parse_qs(urlparse(url).query)
            video_id = qs.get("v", ["unknown"])[0]
        return f"downloads/youtube_{video_id}_{timestamp}"
    else:
        filename = os.path.basename(urlparse(url).path)
        name_without_ext = os.path.splitext(filename)[0]
        return f"downloads/direct_{name_without_ext}_{timestamp}"

def download_video_from_youtube(url, output_path):
    try:
        ydl_opts = {
            'format': 'best[ext=mp4]/best',
            'outtmpl': output_path,
            'quiet': True,
            'noplaylist': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return output_path
    except Exception as e:
        raise RuntimeError(f"Error downloading video from YouTube: {e}")

def download_video_from_direct_url(url, output_path):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return output_path
    except Exception as e:
        raise RuntimeError(f"Error downloading MP4 from direct URL: {e}")

def extract_audio_from_video(video_path, audio_output_path):
    try:
        clip = VideoFileClip(video_path)
        audio = clip.audio
        if audio:
            audio.write_audiofile(audio_output_path, fps=16000)
            return audio_output_path
        else:
            raise RuntimeError("No audio found in video.")
    except Exception as e:
        raise RuntimeError(f"Error extracting audio: {e}")

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=16000)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    mfcc_features = np.mean(np.vstack([mfcc, delta_mfcc, delta2_mfcc]).T, axis=0)

    duration = librosa.get_duration(y=y, sr=sr)
    f0 = librosa.yin(y, fmin=50, fmax=300, sr=sr)
    f0 = f0[f0 > 0]
    pitch_median = np.median(f0) if len(f0) > 0 else 0
    rmse = librosa.feature.rms(y=y)
    intensity_mean = np.mean(rmse)

    prosodic_features = np.array([duration, pitch_median, intensity_mean])
    return np.concatenate([mfcc_features, prosodic_features])

def transcribe_audio(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result["text"]

def summarize_text(text):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(text, max_length=1024, min_length=10, do_sample=True)
    return summary[0]['summary_text']

# === Main Pipeline with UI Feedback ===

def run_pipeline(url, output_widget):
    try:
        # Show loading message and refresh UI
        output_widget.delete(1.0, tk.END)
        output_widget.insert(tk.END, "⏳ Loading... Please wait while we process the video.\n")
        output_widget.update()

        folder_name = generate_folder_name(url)
        os.makedirs(folder_name, exist_ok=True)
        video_path = os.path.join(folder_name, "video.mp4")
        audio_path = os.path.join(folder_name, "audio.wav")

        # Download video
        if is_youtube_url(url):
            video_file = download_video_from_youtube(url, video_path)
        else:
            video_file = download_video_from_direct_url(url, video_path)

        # Extract audio
        audio_file = extract_audio_from_video(video_file, audio_path)

        # Feature extraction
        features = extract_features(audio_file)
        features = features.reshape(1, -1)

        # Load model
        scaler = joblib.load("D:\\Charbel_LEBBOUS\\AI Projects\\accent-detector\\model\\scaler.pkl")
        model = joblib.load("D:\\Charbel_LEBBOUS\\AI Projects\\accent-detector\\model\\random_forest_accent_classifier_tuned.pkl")
        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        confidence = np.max(probabilities) * 100

        label_prob = dict(zip(model.classes_, probabilities))
        prob_str = "\n".join([f"   - {label}: {prob*100:.2f}%" for label, prob in sorted(label_prob.items(), key=lambda x: -x[1])])

        # Transcription & Summary
        transcription = transcribe_audio(audio_file)
        summary = summarize_text(transcription)

        # Display in UI
        output_widget.delete(1.0, tk.END)
        output_widget.insert(tk.END, f"🗣️ Predicted Accent: {prediction}\n")
        output_widget.insert(tk.END, f"🔍 Confidence Score: {confidence:.2f}%\n\n")
        output_widget.insert(tk.END, "📊 Probabilities:\n" + prob_str + "\n\n")
        output_widget.insert(tk.END, "📝 Transcript:\n" + transcription + "\n\n")
        output_widget.insert(tk.END, "🧠 Summary:\n" + summary + "\n")

    except Exception as e:
        messagebox.showerror("Error", str(e))

# === GUI Setup ===

def start_ui():
    window = tk.Tk()
    window.title("Accent Detector & Speech Summarizer")
    window.geometry("800x700")

    tk.Label(window, text="📽️ Enter YouTube video or Youtube Shorts or MP4 URL\n\nThe smaller the better for time efficency (to not wait long for processing):", font=("Arial", 12)).pack(pady=10)
    url_entry = tk.Entry(window, width=90)
    url_entry.pack(pady=5)

    output_box = scrolledtext.ScrolledText(window, wrap=tk.WORD, width=95, height=30)
    output_box.pack(pady=10)

    def on_run():
        url = url_entry.get().strip()
        if url:
            run_pipeline(url, output_box)
        else:
            messagebox.showwarning("Input Needed", "Please enter a video URL.")

    tk.Button(window, text="Analyze Video", command=on_run, font=("Arial", 12), bg="blue", fg="white").pack(pady=10)

    window.mainloop()

if __name__ == "__main__":
    start_ui()
