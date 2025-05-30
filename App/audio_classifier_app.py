import tkinter as tk
from tkinter import messagebox, filedialog
import librosa
import numpy as np
import torch
import tensorflow_hub as hub
import tensorflow as tf
import pandas as pd
import wave
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pygame
import pyaudio
from pydub import AudioSegment
from pydub.utils import which
import noisereduce as nr
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ideally, 50S+
import torch
import torch.nn as nn
import torch.nn.functional as F

class YAMNet1DCNN_Improved(nn.Module):
    def __init__(self, num_classes=57):
        super().__init__()
        self.conv1 = nn.Conv1d(1024, 512, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(512)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(512, 256, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(256)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(256, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.65)  # Slightly higher for better regularization
        self.fc = nn.Linear(128, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, 100, 1024) → (B, 1024, 100)
        x = self.pool1(F.gelu(self.bn1(self.conv1(x))))
        x = self.pool2(F.gelu(self.bn2(self.conv2(x))))
        x = F.gelu(self.bn3(self.conv3(x)))
        x = self.global_pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)


pygame.init()
script_dir = os.path.dirname(os.path.abspath(__file__))
recorded_path = os.path.join(script_dir, "recorded.wav")
AudioSegment.converter = which("ffmpeg")

df = pd.read_csv(r"C:\Users\nasrr\Desktop\CNN_Projects\AudioClassifier\final_merged_with_class_ids.csv")
label_dict = df[['class_id', 'category']].drop_duplicates().set_index('class_id')['category'].to_dict()

yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YAMNet1DCNN_Improved().to(device)
model_path = r"C:\Users\nasrr\Desktop\CNN_Projects\AudioClassifier\Model_91_50_Epoch77.pt"
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

pyaudio_instance = pyaudio.PyAudio()
audio_frames = []
recording_stream = None
recording_wavefile = recorded_path

def predict_my_class(wav_path):
    try:
        waveform, sr = librosa.load(wav_path, sr=16000, mono=True)

        # Trim silence
        waveform, _ = librosa.effects.trim(waveform)

        # Normalize
        if np.max(np.abs(waveform)) > 0:
            waveform = waveform / np.max(np.abs(waveform))

        # Denoise
        waveform = nr.reduce_noise(y=waveform, sr=sr)

        # Convert to tensor
        waveform = tf.convert_to_tensor(waveform, dtype=tf.float32)
        _, embeddings, _ = yamnet_model(waveform)

        emb_np = embeddings.numpy()

        # Ensure at least 100 frames
        if emb_np.shape[0] < 100:
            emb_np = np.pad(emb_np, ((0, 100 - emb_np.shape[0]), (0, 0)), mode='constant')
        else:
            emb_np = emb_np[:100]

        input_tensor = torch.tensor(emb_np, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 100, 1024)

        with torch.no_grad():
            logits = model(input_tensor)
            pred_class = logits.argmax(1).item()
            return label_dict.get(pred_class, f"Class ID: {pred_class}")
    except Exception as e:
        return f"❌ Error: {e}"

def start_recording():
    global recording_stream, audio_frames
    audio_frames = []
    recording_stream = pyaudio_instance.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
    result_label.config(text="🎙️ Recording...")
    record_button.config(state="disabled")
    stop_button.config(state="normal")

    def callback():
        try:
            data = recording_stream.read(1024, exception_on_overflow=False)
            audio_frames.append(data)
            app.after(1, callback)
        except Exception as e:
            result_label.config(text=f"❌ Recording error: {e}")

    callback()

def stop_recording():
    global recording_stream
    if recording_stream:
        recording_stream.stop_stream()
        recording_stream.close()

    with wave.open(recording_wavefile, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(pyaudio_instance.get_sample_size(pyaudio.paInt16))
        wf.setframerate(16000)
        wf.writeframes(b''.join(audio_frames))

    record_button.config(state="normal")
    stop_button.config(state="disabled")
    result_label.config(text="✅ Recording saved")

def play_audio():
    try:
        wf = wave.open(recording_wavefile, 'rb')
        stream = pyaudio_instance.open(format=pyaudio_instance.get_format_from_width(wf.getsampwidth()),
                                       channels=wf.getnchannels(),
                                       rate=wf.getframerate(),
                                       output=True)
        data = wf.readframes(1024)
        while data:
            stream.write(data)
            data = wf.readframes(1024)
        stream.stop_stream()
        stream.close()
        result_label.config(text="🔊 Finished playing")
    except Exception as e:
        result_label.config(text=f"❌ Playback error: {e}")

def predict_recorded():
    result_label.config(text="🔄 Predicting...")
    app.update_idletasks()
    result = predict_my_class(recorded_path)
    result_label.config(text=f"🎧 Predicted: {result}")

def upload_and_predict():
    path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
    if path:
        result_label.config(text="🔄 Predicting...")
        app.update_idletasks()
        result = predict_my_class(path)
        result_label.config(text=f"📂 Predicted: {result}")

def convert_mp3_to_wav(mp3_path):
    wav_path = os.path.splitext(mp3_path)[0] + ".wav"
    sound = AudioSegment.from_mp3(mp3_path)
    sound = sound.set_frame_rate(16000).set_channels(1)
    sound.export(wav_path, format="wav")
    return wav_path

def convert_and_predict():
    path = filedialog.askopenfilename(filetypes=[("MP3 files", "*.mp3")])
    if not path:
        return
    result_label.config(text="🔄 Converting and predicting...")
    app.update_idletasks()
    try:
        wav_path = convert_mp3_to_wav(path)
        result = predict_my_class(wav_path)
        result_label.config(text=f"🎼 MP3 Prediction: {result}")
    except Exception as e:
        result_label.config(text=f"❌ Error: {e}")

def styled_button(text, command):
    return tk.Button(app, text=text, font=("Segoe UI", 12, "bold"), fg="white", bg="#007acc",
                     activebackground="#005f9e", activeforeground="white", bd=0, relief="flat",
                     padx=20, pady=8, cursor="hand2", command=command)

app = tk.Tk()
app.configure(bg="#1e1e2f")
app.title("🎙️ Audio Classifier")
app.geometry("800x250")

header = tk.Label(app, text="🎧 Audio Classifier", font=("Segoe UI Semibold", 20), fg="#61dafb", bg="#1e1e2f")
header.pack(pady=10)

#styled_button("🔊 Hear", play_audio).pack(pady=5)
#styled_button("🔍 Predict", predict_recorded).pack(pady=5)

styled_button("📂 Upload & Predict", upload_and_predict).pack(pady=5)
styled_button("🎼 Convert MP3 & Predict", convert_and_predict).pack(pady=5)

#record_button = styled_button("🎙️ Start Recording", start_recording)
#record_button.pack(pady=5)

#stop_button = styled_button("⏹️ Stop Recording", stop_recording)
#stop_button.pack(pady=5)

result_label = tk.Label(app, text="🔎 Result will appear here", font=("Segoe UI", 12), fg="#c6c6c6", bg="#1e1e2f",
                        wraplength=400, justify="center")
result_label.pack(pady=20)

def on_enter(e):
    e.widget['bg'] = "#005f9e"

def on_leave(e):
    e.widget['bg'] = "#007acc"

for widget in app.winfo_children():
    if isinstance(widget, tk.Button):
        widget.bind("<Enter>", on_enter)
        widget.bind("<Leave>", on_leave)

app.mainloop()
