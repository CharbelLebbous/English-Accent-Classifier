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

# === Helper functions ===

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
        print(f"✅ YouTube video downloaded to: {output_path}")
        return output_path
    except Exception as e:
        print(f"❌ Error downloading video from YouTube: {e}")
        return None

def download_video_from_direct_url(url, output_path):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✅ Direct MP4 video downloaded to: {output_path}")
        return output_path
    except Exception as e:
        print(f"❌ Error downloading MP4 from direct URL: {e}")
        return None

def extract_audio_from_video(video_path, audio_output_path):
    try:
        clip = VideoFileClip(video_path)
        audio = clip.audio
        if audio:
            audio.write_audiofile(audio_output_path, fps=16000)
            print(f"🎵 Audio extracted to: {audio_output_path}")
            return audio_output_path
        else:
            print("❌ No audio found in the video.")
            return None
    except Exception as e:
        print(f"❌ Error extracting audio: {e}")
        return None

# === Feature Extraction ===

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=16000)

    # MFCC + deltas
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    mfcc_features = np.mean(np.vstack([mfcc, delta_mfcc, delta2_mfcc]).T, axis=0)

    # Prosodic Features
    duration = librosa.get_duration(y=y, sr=sr)
    f0 = librosa.yin(y, fmin=50, fmax=300, sr=sr)
    f0 = f0[f0 > 0]
    pitch_median = np.median(f0) if len(f0) > 0 else 0
    rmse = librosa.feature.rms(y=y)
    intensity_mean = np.mean(rmse)

    prosodic_features = np.array([duration, pitch_median, intensity_mean])

    return np.concatenate([mfcc_features, prosodic_features])

# === Transcription and Summarization ===

def transcribe_audio(audio_path):
    print("📝 Transcribing audio using Whisper...")
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result["text"]

def summarize_text(text):
    print("📚 Summarizing transcript...")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(text, max_length=1024, min_length=10, do_sample=True)
    return summary[0]['summary_text']

# === Main pipeline ===

def main():
    url = input("📽️ Enter a public video URL (YouTube or direct MP4 link): ").strip()
    folder_name = generate_folder_name(url)
    os.makedirs(folder_name, exist_ok=True)

    video_path = os.path.join(folder_name, "video.mp4")
    audio_path = os.path.join(folder_name, "audio.wav")

    # Step 1: Download Video
    if is_youtube_url(url):
        video_file = download_video_from_youtube(url, video_path)
    else:
        video_file = download_video_from_direct_url(url, video_path)

    if not video_file or not os.path.exists(video_file):
        print("⚠️ Video download failed.")
        return

    # Step 2: Extract Audio
    audio_file = extract_audio_from_video(video_file, audio_path)
    if not audio_file:
        print("⚠️ Audio extraction failed.")
        return

        # Step 3: Extract Features
    features = extract_features(audio_file)

    if features is None or len(features) == 0:
        print("❌ Feature extraction failed or returned empty.")
        return

    features = features.reshape(1, -1)
    print(f"✅ Extracted feature shape: {features.shape}")

    # Step 4: Load model and scaler
    scaler_path = "D:\\Charbel_LEBBOUS\\AI Projects\\accent-detector\\model\\scaler.pkl"
    model_path = "D:\\Charbel_LEBBOUS\\AI Projects\\accent-detector\\model\\random_forest_accent_classifier_tuned.pkl"

    if not os.path.exists(scaler_path) or not os.path.exists(model_path):
        print("❌ Missing model or scaler files.")
        return

    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)

    try:
        features_scaled = scaler.transform(features)

        # Step 5: Predict Accent
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]

        if len(probabilities) == 0 or np.max(probabilities) == 0:
            print("❌ Prediction failed: empty or invalid probability distribution.")
            return

        confidence = np.max(probabilities) * 100

        print(f"\n🗣️ **Predicted Accent:** {prediction}")
        print(f"🔍 **Confidence Score:** {confidence:.2f}%")
    # Optional: Show full class probabilities
        label_prob = dict(zip(model.classes_, probabilities))
        print("📊 Probabilities:")
        
        for accent, prob in sorted(label_prob.items(), key=lambda x: -x[1]):
            print(f"   - {accent}: {prob*100:.2f}%")
            
    except IndexError as ie:
        print(f"❌ IndexError during prediction: {ie}")
        print(f"💡 Check if model expects a specific feature size.")
        print(f"Expected input shape: {model.n_features_in_} (if available), got: {features.shape[1]}")
        return

    except Exception as e:
        print(f"❌ Unexpected error during prediction: {e}")
        return

    # Step 6: Transcribe Speech
    transcription = transcribe_audio(audio_file)
    print("\n📝 Full Transcript:")
    print(transcription)

    # Step 7: Summarize Speech
    summary = summarize_text(transcription)
    print("\n🧠 Summary of Spoken Content:")
    print(summary)

if __name__ == "__main__":
    main()
