import tkinter as tk
from tkinter import ttk
import threading
import pyaudio
import wave
import numpy as np
from transformers import MarianMTModel, MarianTokenizer
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import noisereduce as nr

# Set up the audio recording parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 2048
FILENAME = "recorded_audio.wav"
MAX_FRAMES = int(RATE / CHUNK * 5)  # For 5 seconds of recording

class AudioProcessing:
    def __init__(self):
        self.processor = None
        self.transcription_model = None
        self.translator_tokenizer = None
        self.translator_model = None
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.frames = []
        self.is_recording = False

    def start_recording(self):
        """Begin audio recording in a new thread."""
        self.stream = self.audio.open(format=FORMAT,
                                      channels=CHANNELS,
                                      rate=RATE,
                                      input=True,
                                      frames_per_buffer=CHUNK)
        self.frames = []
        self.is_recording = True

        # Change the button color to red
        button.config(bg="red", text="Recording...")
        root.update()  # Update the UI

        # Start recording in a separate thread
        threading.Thread(target=self.record).start()

    def record(self):
        """Record audio until stopped."""
        while self.is_recording:
            data = self.stream.read(CHUNK)
            self.frames.append(data)

    def stop_recording(self):
        """Stop recording and save the file."""
        self.is_recording = False
        self.stream.stop_stream()
        self.stream.close()

        # Save the recorded audio to a .wav file
        wf = wave.open(FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(self.frames))
        wf.close()

        # Revert the button color
        button.config(bg="SystemButtonFace", text="Record and Process")
        root.update()  # Update the UI

        # Proceed to transcription and translation
        self.transcribe_and_translate()

    def load_models(self, transcription_lang, translation_lang):
        """Load the transcription and translation models based on the selected languages."""
        if transcription_lang == "English":
            transcription_model = "facebook/wav2vec2-large-960h"
        elif transcription_lang == "Spanish":
            transcription_model = "facebook/wav2vec2-large-xlsr-53-spanish"
        else:
            print(f"Unsupported transcription language: {transcription_lang}")
            return

        self.processor = Wav2Vec2Processor.from_pretrained(transcription_model)
        self.transcription_model = Wav2Vec2ForCTC.from_pretrained(transcription_model)

        if translation_lang == "English":
            translator_model = 'Helsinki-NLP/opus-mt-es-en'
        elif translation_lang == "Spanish":
            translator_model = 'Helsinki-NLP/opus-mt-en-es'
        elif translation_lang == "French":
            translator_model = 'Helsinki-NLP/opus-mt-es-fr'
        else:
            print(f"Unsupported translation language: {translation_lang}")
            return

        self.translator_tokenizer = MarianTokenizer.from_pretrained(translator_model)
        self.translator_model = MarianMTModel.from_pretrained(translator_model)

    def transcribe_and_translate(self):
        """Perform transcription and translation after recording."""
        # Load the recorded audio for transcription
        wf = wave.open(FILENAME, 'rb')
        audio_data = wf.readframes(wf.getnframes())
        audio_data = np.frombuffer(audio_data, dtype=np.int16)

        # Reduce noise
        audio_data = nr.reduce_noise(y=audio_data, sr=RATE)

        # Transcribe the audio
        input_values = self.processor(audio_data, sampling_rate=16000, return_tensors="pt", padding="longest").input_values
        input_values = input_values.to(torch.float32)

        with torch.no_grad():
            logits = self.transcription_model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0].strip()

        # Translate the transcription
        tokens = self.translator_tokenizer(transcription, return_tensors='pt')
        translation = self.translator_model.generate(**tokens)
        translated_text = self.translator_tokenizer.decode(translation[0], skip_special_tokens=True)

        # Display the results
        display_text = f"Transcribed: {transcription}\nTranslated: {translated_text}"
        text_label.config(text=display_text)
        text_label.grid(row=5, column=0, columnspan=2, pady=10)

    def close_app(self):
        """Ensure graceful shutdown of the application."""
        if self.is_recording:
            self.stop_recording()  # Stop recording if still active
        self.audio.terminate()  # Terminate the audio interface
        root.destroy()  # Close the application



def main():
    global root, button, text_label, stop_button, close_button, combobox_1, combobox_2

    # Initialize the audio processing class
    processing = AudioProcessing()

    # Set up the main window
    root = tk.Tk()
    root.title("Speech to Text Translator")
    root.geometry("400x400")  # Adjusted height for better visibility

    # Language selection comboboxes
    combobox_1 = ttk.Combobox(root, values=["English", "Spanish", "French"])
    combobox_1.set("English")  # Default value
    combobox_1.grid(row=0, column=0, padx=10, pady=10, sticky="we")

    combobox_2 = ttk.Combobox(root, values=["English", "Spanish", "French"])
    combobox_2.set("Spanish")  # Default value
    combobox_2.grid(row=0, column=1, padx=10, pady=10, sticky="we")

    # Record button
    button = tk.Button(root, text="Record and Process", background="white", command=lambda: processing.load_models(combobox_1.get(), combobox_2.get()) or processing.start_recording())
    button.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="we")

    # Stop button
    stop_button = tk.Button(root, text="Stop Recording", command=processing.stop_recording)
    stop_button.grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky="we")

    # Text label to display transcriptions and translations
    text_label = tk.Label(root, text="", font=("Helvetica", 16))
    text_label.grid(row=5, column=0, columnspan=2, pady=10, sticky="we")

    # Close app button
    close_button = tk.Button(root, text="Close App", command=processing.close_app)
    close_button.grid(row=3, column=0, columnspan=2, padx=10, pady=20, sticky="we")

    # Configure the grid to ensure everything is centered
    root.grid_columnconfigure(0, weight=1)
    root.grid_columnconfigure(1, weight=1)

    # Start the Tkinter main loop
    root.mainloop()

if __name__ == "__main__":
    main()