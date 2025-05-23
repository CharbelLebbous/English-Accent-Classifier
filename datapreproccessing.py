import os
import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
import joblib
from collections import Counter

# === Paths ===
metadata_file = "D:\\Charbel_LEBBOUS\\AI Projects\\accent-detector\\cv-corpus-10.0-delta-2022-07-04\\en\\validated.tsv"
audio_dir = "D:\\Charbel_LEBBOUS\\AI Projects\\accent-detector\\cv-corpus-10.0-delta-2022-07-04\\en\\clips"

# === Load and Clean Data ===
df = pd.read_csv(metadata_file, sep="\t")
df = df[df['accents'].notna()]
df = df[df['accents'].str.strip() != ""]

# === Accent Mapping ===
label_map = {
    "United States English": "US English",
    "United States English,Midwestern,Low,Demure": "US English",
    "United States English,Southwestern United States English": "US English",
    "United States English,southern United States,New Orleans dialect": "US English",
    "United States English,England English": "US English",
    "England English": "British English",
    "Northern Irish": "British English",
    "Canadian English": "North American English",
    "Australian English": "Australian English",
    "New Zealand English": "Australian English",
    "India and South Asia (India, Pakistan, Sri Lanka)": "Indian English",
    "German English,Non native speaker": "German English",
    "Filipino": "Asian English",
    "Thai": "Asian English"
}

df = df[df['accents'].isin(label_map.keys())]
df['label'] = df['accents'].map(label_map)

print(f"✅ Valid samples after mapping: {len(df)}")

# === Undersample for Class Balance ===
max_per_class = {
    "Asian English": 30,
    "Indian English": 33,
    "Australian English": 43,
    "North American English": 77,
    "German English": 126,
    "British English": 150,
    "US English": 150
}

balanced_df = df.groupby('label').apply(
    lambda x: x.sample(min(len(x), max_per_class[x.name]), random_state=42)
).reset_index(drop=True)

print("📊 Sample counts after undersampling:")
print(balanced_df['label'].value_counts())

# === Feature Extraction ===
features = []
labels = []

for _, row in tqdm(balanced_df.iterrows(), total=len(balanced_df)):
    file_path = os.path.join(audio_dir, row['path'])

    if not os.path.exists(file_path):
        continue

    try:
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

        # Combine
        final_features = np.concatenate([mfcc_features, prosodic_features])
        features.append(final_features)
        labels.append(row['label'])

    except Exception as e:
        print(f"⚠️ Error processing {file_path}: {e}")
        continue

X = np.array(features)
y = np.array(labels)

print(f"\n✅ Feature extraction with prosody complete. Shape: {X.shape}")
print("📈 Pre-oversampling class distribution:")
unique, counts = np.unique(y, return_counts=True)
for cls, cnt in zip(unique, counts):
    print(f"{cls}: {cnt}")

# === Oversampling ===
target_class_sizes = {
    "Asian English": 80,
    "Indian English": 80,
    "Australian English": 80,
    "North American English": 100,
    "German English": 126,
    "British English": 150,
    "US English": 150
}

sampling_strategy = {
    cls: target_class_sizes[cls] for cls in np.unique(y)
    if sum(y == cls) < target_class_sizes[cls]
}

ros = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

print("\n✅ Oversampling complete.")
print("📊 Final class distribution:")
unique, counts = np.unique(y_resampled, return_counts=True)
for cls, cnt in zip(unique, counts):
    print(f"{cls}: {cnt}")

# === Normalize Features ===
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X_resampled)

# Save scaler
joblib.dump(scaler, 'D:\\Charbel_LEBBOUS\\AI Projects\\accent-detector\\model\\scaler.pkl')
print("🧠 Saved feature scaler to scaler.pkl")

# === Save Datasets ===
np.save("features_balanced_normalized.npy", X_normalized)
np.save("labels_balanced.npy", y_resampled)

print(f"\n💾 Saved {len(X_normalized)} normalized samples to features_balanced_normalized.npy / labels_balanced.npy")
