import os
import torch
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from pydub.utils import which
import librosa
import tensorflow_hub as hub
import tensorflow as tf
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.filechooser import FileChooserIconView
from kivy.core.window import Window
import pandas as pd
import torch


# Load YAMNet model once globally
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🧠 Load TorchScript model
base_dir = os.path.dirname(os.path.abspath(__file__))
# Load CSV and map ID to label
label_df = pd.read_csv(os.path.join(base_dir, "final_merged_with_class_ids.csv"))
label_dict = label_df[['class_id', 'category']].drop_duplicates().set_index('class_id')['category'].to_dict()

model_path = os.path.join(base_dir, "Mobile_Audio_Classifier.pt")
model = torch.jit.load(model_path)
model.eval()

# ⚙ Ensure pydub uses ffmpeg
AudioSegment.converter = which("ffmpeg")

# 🎨 App Window size
Window.size = (500, 720)

class MobileAudioClassifierApp(App):
    def build(self):
        self.root = BoxLayout(orientation='vertical', padding=20, spacing=20)

        # 📂 File chooser, defaults to script directory
        self.file_chooser = FileChooserIconView(path=base_dir, filters=["*.wav", "*.mp3"])
        self.root.add_widget(self.file_chooser)

        # 🧠 Result label
        self.result_label = Label(text="🎧 Upload a WAV/MP3 file to classify", font_size=18, halign='center')
        self.root.add_widget(self.result_label)

        # 🔘 Buttons
        self.predict_btn = self._make_button("🎧 Predict Audio File", self.predict_with_yamnet_cnn)
        self.convert_btn = self._make_button("🎼 Convert MP3 to WAV", self.convert_mp3)

        self.root.add_widget(self.predict_btn)
        self.root.add_widget(self.convert_btn)

        return self.root

    def _make_button(self, text, func):
        btn = Button(
            text=text,
            font_size=18,
            size_hint=(1, None),
            height=60,
            background_color=(0.1, 0.5, 0.8, 1),
            color=(1, 1, 1, 1)
        )
        btn.bind(on_press=func)
        return btn

    def get_selected_file(self):
        if self.file_chooser.selection:
            return self.file_chooser.selection[0]
        self.result_label.text = "❌ Please select a file."
        return None

    def convert_mp3(self, instance):
        path = self.get_selected_file()
        if not path or not path.lower().endswith(".mp3"):
            self.result_label.text = "❌ Please select an MP3 file."
            return

        try:
            audio = AudioSegment.from_mp3(path)
            audio = audio.set_frame_rate(16000).set_channels(1)
            out_path = os.path.splitext(path)[0] + "_converted.wav"
            audio.export(out_path, format="wav")
            self.result_label.text = f"✅ Converted:\n{os.path.basename(out_path)}"
        except Exception as e:
            self.result_label.text = f"❌ MP3 conversion error: {e}"

    def predict_with_yamnet_cnn(self, instance):
        path = self.get_selected_file()
        if not path or not path.lower().endswith((".wav", ".mp3")):
            self.result_label.text = "❌ Please select a WAV or MP3 file."
            return

        try:
            waveform, sr = librosa.load(path, sr=16000, mono=True)
            waveform = tf.convert_to_tensor(waveform, dtype=tf.float32)

            _, embeddings, _ = yamnet_model(waveform)
            emb_np = embeddings.numpy()

            emb_np = np.pad(emb_np, ((0, max(0, 100 - emb_np.shape[0])), (0, 0)), mode='constant')[:100]
            input_tensor = torch.tensor(emb_np, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(input_tensor)
                pred_class = logits.argmax(1).item()

            label = label_dict.get(pred_class, f"Class ID: {pred_class}")
            self.result_label.text = f"✅ Predicted: {label} (ID: {pred_class})"

        except Exception as e:
            self.result_label.text = f"❌ Error: {e}"

            return f"❌ Error: {e}"

if __name__ == "__main__":
    MobileAudioClassifierApp().run()
