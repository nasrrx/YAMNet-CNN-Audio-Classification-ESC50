import tkinter as tk
from tkinter import messagebox, filedialog
import sounddevice as sd
import librosa
import numpy as np
import torch
import tensorflow_hub as hub
import tensorflow as tf
import pandas as pd
import tempfile
import scipy.io.wavfile
from pydub import AudioSegment
import simpleaudio as sa 
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pygame
import soundfile as sf
import tempfile
import os
import sounddevice as sd

pygame.init()

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Set the path to save the recorded audio
recorded_path = os.path.join(script_dir, "recorded.wav")

from pydub.utils import which
AudioSegment.converter = which("ffmpeg")

# Load CSV label mapping
df = pd.read_csv(r"C:\Users\nasrr\Desktop\CNN_Projects\AudioClassifier\final_merged_with_class_ids.csv")

label_dict = df[['class_id', 'category']].drop_duplicates().set_index('class_id')['category'].to_dict()

# Load YAMNet TF model
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

# Load PyTorch model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        self.dropout = nn.Dropout(0.75)  # Slightly higher for better regularization
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

model = YAMNet1DCNN_Improved().to(device)

model_path = r"C:\Users\nasrr\Desktop\CNN_Projects\AudioClassifier\Best_Model.pt"

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

def predict_my_class(wav_path):
    try:
        waveform, sr = librosa.load(wav_path, sr=16000, mono=True)
        waveform = tf.convert_to_tensor(waveform, dtype=tf.float32)
        _, embeddings, _ = yamnet_model(waveform)
        emb_np = embeddings.numpy()
        emb_np = np.pad(emb_np, ((0, max(0, 100 - emb_np.shape[0])), (0, 0)), mode='constant')[:100]
        input_tensor = torch.tensor(emb_np, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(input_tensor)
            pred_class = logits.argmax(1).item()
            return label_dict.get(pred_class, f"Class ID: {pred_class}")
    except Exception as e:
        return f"❌ Error: {e}"

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

    # 🌀 Frame 1: Start conversion
    result_label.config(text="🔄 Converting MP3 (step 1)...")
    app.update_idletasks()

    # Delay to simulate frame 1
    def step_2():
        result_label.config(text="🔄 Converting MP3 (step 2)...")
        app.update_idletasks()

        # Delay before actual prediction
        def final_step():
            try:
                wav_path = convert_mp3_to_wav(path)
                result_label.config(text="🔄 Predicting...")
                app.update_idletasks()
                result = predict_my_class(wav_path)
                result_label.config(text=f"🎼 MP3 Prediction: {result}")
            except Exception as e:
                result_label.config(text=f"❌ Error: {e}")

        app.after(100, final_step)  # 🕓 0.1s pause between frame 2 and prediction

    app.after(100, step_2)  # 🕓 0.1s pause between frame 1 and frame 2


def start_recording():
    global stream, live_buffer
    live_buffer = []

    def audio_callback(indata, frames, time, status):
        if status:
            print(status)
        live_buffer.append(indata[:, 0])

    stream = sd.InputStream(callback=audio_callback, samplerate=samplerate, channels=1)
    stream.start()
    result_label.config(text="🎙️ Recording...")
    record_button.config(state="disabled")
    stop_button.config(state="normal")

def stop_recording():
    global stream
    if stream:
        stream.stop()
        stream.close()

    pygame.mixer.music.stop()
    pygame.mixer.quit()
    pygame.mixer.init()

    audio = np.concatenate(live_buffer)

    # 🔧 Normalize and clip to avoid distortion
    audio = np.clip(audio, -1.0, 1.0)

    # ✅ Convert float [-1.0, 1.0] to int16 [-32767, 32767]
    int_audio = (audio * 32767).astype(np.int16)

    try:
        import scipy.io.wavfile
        scipy.io.wavfile.write(recorded_path, 16000, int_audio)
        result_label.config(text="✅ Recording saved")
    except Exception as e:
        result_label.config(text=f"❌ Save error: {e}")

    record_button.config(state="normal")
    stop_button.config(state="disabled")



    try:
        pygame.mixer.music.load(recorded_path)
        pygame.mixer.music.play()
        result_label.config(text="🔊 Playing...")
    except Exception as e:
        result_label.config(text=f"❌ Can't play: {e}")

    try:
        wave_obj = sa.WaveObject.from_wave_file(recorded_path)
        wave_obj.play()
    except Exception as e:
        result_label.config(text=f"❌ Can't play: {e}")

def play_audio():
    try:
        data, fs = sf.read(recorded_path, dtype='float32')
        sd.play(data, fs)
        result_label.config(text="🔊 Playing (via sounddevice)...")
    except Exception as e:
        result_label.config(text=f"❌ Can't play: {e}")



def predict_recorded():
    result_label.config(text="🔄 Predicting...")
    app.update_idletasks()  # Forces GUI to update immediately
    result = predict_my_class(recorded_path)
    result_label.config(text=f"🎧 Predicted: {result}")
        # 🧹 Release any audio lock
    pygame.mixer.music.stop()
    pygame.mixer.quit()
    pygame.mixer.init()

def upload_and_predict():
    path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
    if path:
        result_label.config(text="🔄 Predicting...")
        app.update_idletasks()
        result = predict_my_class(path)
        result_label.config(text=f"📂 Predicted: {result}")
        
def update_plot():
    if live_buffer:
        data = np.concatenate(live_buffer)[-1024:]
        if len(data) < 1024:
            data = np.pad(data, (0, 1024 - len(data)))
        line.set_ydata(data)
        line.set_xdata(np.arange(len(data)))
        canvas.draw()
    app.after(2, update_plot)




# 🖼️ GUI
def styled_button(text, command):
    return tk.Button(
        app,
        text=text,
        font=("Segoe UI", 12, "bold"),
        fg="white",
        bg="#007acc",
        activebackground="#005f9e",
        activeforeground="white",
        bd=0,
        relief="flat",
        padx=20,
        pady=8,
        cursor="hand2",
        command=command
    )

app = tk.Tk()
app.configure(bg="#1e1e2f")  # Dark background

header = tk.Label(
    app,
    text="🎧 Audio Classifier",
    font=("Segoe UI Semibold", 20),
    fg="#61dafb",
    bg="#1e1e2f"
)
header.pack(pady=10)


fig, ax = plt.subplots(figsize=(5, 1.5))
line, = ax.plot([], [], lw=2)
ax.set_ylim(-1, 1)
ax.set_xlim(0, 1024)
ax.axis("off")

canvas = FigureCanvasTkAgg(fig, master=app)
canvas.get_tk_widget().pack()

live_buffer = []
stream = None
samplerate = 16000

app.title("🎙️ Audio Classifier")
app.geometry("800x800")


styled_button("🔊 Hear", play_audio).pack(pady=5)
styled_button("🔍 Predict", predict_recorded).pack(pady=5)
styled_button("📂 Upload & Predict", upload_and_predict).pack(pady=5)
styled_button("🎼 Convert MP3 & Predict", convert_and_predict).pack(pady=5)


record_button = styled_button("🎙️ Start Recording", start_recording)
record_button.pack(pady=5)

stop_button = styled_button("⏹️ Stop Recording", stop_recording)
stop_button.pack(pady=5)


result_label = tk.Label(
    app,
    text="🔎 Result will appear here",
    font=("Segoe UI", 12),
    fg="#c6c6c6",
    bg="#1e1e2f",
    wraplength=400,
    justify="center"
)
result_label.pack(pady=20)


def on_enter(e):
    e.widget['bg'] = "#005f9e"

def on_leave(e):
    e.widget['bg'] = "#007acc"

for widget in app.winfo_children():
    if isinstance(widget, tk.Button):
        widget.bind("<Enter>", on_enter)
        widget.bind("<Leave>", on_leave)


update_plot()
app.mainloop()
