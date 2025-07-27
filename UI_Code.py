import sys
import os
import numpy as np
import soundfile as sf
import sounddevice as sd
import warnings
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvas
from PyQt5.QtWidgets import (
    QMainWindow, QApplication, QLabel, QTextEdit, QListWidget,
    QVBoxLayout, QHBoxLayout, QWidget, QScrollArea, QSplitter,
    QPushButton, QFileDialog, QMessageBox, QProgressBar
)
from PyQt5.QtCore import Qt, QTimer, QUrl
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtGui import QPixmap

# Import the inference module
from inference import inference

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class AudioCaptionInterface(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Caption Interface")
        self.setGeometry(300, 300, 1200, 900)
        self.setStyleSheet(self.get_stylesheet())

        self.audio_path = None
        self.recording_path = "temp_audio.wav"  # Temporary file for recording
        self.num_predictions = 3
        self.recording = False
        self.recording_duration = 0
        self.timer = QTimer()
        self.audio_data_real_time = np.array([])  # For real-time audio data

        self.audio_path_label = QLabel("Audio Path:")
        self.audio_path_text = QTextEdit()
        self.audio_path_text.setReadOnly(True)

        self.image_label = QLabel("Image:")
        self.image_display = QLabel()
        self.image_display.setFixedSize(400, 300)

        self.ground_truth_label = QLabel("Live Environment Sound Captions:")
        self.ground_truth_list = QListWidget()
        
        self.prediction_label = QLabel("Model Predictions:")
        self.prediction_list = QListWidget()

        self.record_button = QPushButton("Record Audio")
        self.record_button.clicked.connect(self.record_audio)

        self.num_pred_label = QLabel("Number of Predictions:")
        self.num_pred_text = QTextEdit()
        self.num_pred_text.setPlainText("3")

        self.figure = Figure(figsize=(12, 6))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

        self.media_player = QMediaPlayer()
        self.play_button = QPushButton("Play Audio")
        self.play_button.clicked.connect(self.play_audio)

        self.progress_bar = QProgressBar()
        self.media_player.positionChanged.connect(self.update_progress_bar)
        self.media_player.durationChanged.connect(self.update_duration)

        self.create_layout()
        self.load_initial_image()  # Load the initial image

    def get_stylesheet(self):
        return """
            QLabel { font-size: 20px; }
            QTextEdit { border: 2px solid #3498db; }
            QListWidget { border: 2px solid #3498db; }
            QPushButton { font-size: 20px; }
        """

    def create_layout(self):
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.audio_path_label)
        left_layout.addWidget(self.audio_path_text)
        left_layout.addWidget(self.image_label)
        left_layout.addWidget(self.image_display)
        left_layout.addWidget(self.record_button)
        left_layout.addWidget(self.num_pred_label)
        left_layout.addWidget(self.num_pred_text)
        left_layout.addWidget(self.play_button)
        left_layout.addWidget(self.progress_bar)

        left_widget = QWidget()
        left_widget.setLayout(left_layout)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.prediction_label)
        right_layout.addWidget(self.prediction_list)
        right_layout.addWidget(self.ground_truth_label)
        right_layout.addWidget(self.ground_truth_list)

        # Buttons for predictions
        self.run_model_button = QPushButton("Run Model Prediction")
        self.run_model_button.clicked.connect(self.run_model_prediction)
        right_layout.addWidget(self.run_model_button)

        self.run_live_button = QPushButton("Run Live Prediction")
        self.run_live_button.clicked.connect(self.run_live_prediction)
        right_layout.addWidget(self.run_live_button)

        right_layout.addWidget(self.canvas)

        right_widget = QWidget()
        right_widget.setLayout(right_layout)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)

        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        main_layout.addWidget(splitter)
        self.setCentralWidget(central_widget)

    def load_initial_image(self):
        image_path = r'D:\Y3ACCsurf\UI.jpg'  # Change to your image path
        pixmap = QPixmap(image_path).scaled(self.image_display.size(), Qt.KeepAspectRatio)
        self.image_display.setPixmap(pixmap)

    def record_audio(self):
        if not self.recording:
            self.recording = True
            self.recording_duration = 0
            self.timer.timeout.connect(self.update_recording_duration)
            self.timer.start(1000)  # Update every second
            self.record_button.setText("Stop Recording")
            QMessageBox.information(self, "Recording", "Recording started.")

            # Start recording
            self.audio_data = []
            self.recording_thread = sd.InputStream(callback=self.audio_callback)
            self.recording_thread.start()
        else:
            self.recording = False
            self.timer.stop()
            self.record_button.setText("Record Audio")
            self.recording_thread.stop()
            self.save_recording()
            QMessageBox.information(self, "Recording", f"Recording stopped. Duration: {self.recording_duration} seconds.\nSaved to: {self.recording_path}")

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(status)
        # Append new audio data for real-time plotting
        self.audio_data.append(indata.copy())
        self.audio_data_real_time = np.append(self.audio_data_real_time, indata.copy())
        self.update_waveform_plot()

    def save_recording(self):
        audio_data = np.concatenate(self.audio_data, axis=0)
        sf.write(self.recording_path, audio_data, 44100)

    def update_recording_duration(self):
        self.recording_duration += 1

    def update_waveform_plot(self):
        self.ax.clear()
        time = np.arange(0, len(self.audio_data_real_time)) / 44100  # Assuming a sample rate of 44100 Hz
        self.ax.plot(time, self.audio_data_real_time, color='#3498db', linewidth=2)
        self.ax.set_xlabel("Time (seconds)")
        self.ax.set_ylabel("Amplitude")
        self.ax.set_title("Audio Waveform")
        self.ax.grid(True)
        self.figure.tight_layout()
        self.canvas.draw()

    def run_inference(self):
        if self.audio_path is None and self.recording_path is None:
            QMessageBox.warning(self, "Warning", "Please select an audio file or record audio first.")
            return

        audio_path_to_use = self.recording_path if self.recording_path else self.audio_path

        try:
            self.num_predictions = int(self.num_pred_text.toPlainText())
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter a valid number for predictions.")
            return

        try:
            predictions = inference(audio_path_to_use, num_samples=self.num_predictions)
            if self.recording_path:  # If audio is from recording, show predictions in ground truth
                self.ground_truth_list.clear()
                for pred in predictions:
                    self.ground_truth_list.addItem(pred)
            else:  # If audio is from file, show predictions in model predictions
                self.prediction_list.clear()
                for pred in predictions:
                    self.prediction_list.addItem(pred)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to run inference: {str(e)}")

    def run_model_prediction(self):
        if self.audio_path is None:
            QMessageBox.warning(self, "Warning", "Please select an audio file first.")
            return

        try:
            predictions = inference(self.audio_path, num_samples=self.num_predictions)
            self.prediction_list.clear()
            for pred in predictions:
                self.prediction_list.addItem(pred)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to run model prediction: {str(e)}")

    def run_live_prediction(self):
        if self.recording_path is None:
            QMessageBox.warning(self, "Warning", "No recorded audio available for live prediction.")
            return

        try:
            predictions = inference(self.recording_path, num_samples=self.num_predictions)
            self.ground_truth_list.clear()
            for pred in predictions:
                self.ground_truth_list.addItem(pred)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to run live prediction: {str(e)}")

    def load_audio(self, audio_path):
        if not os.path.exists(audio_path):
            QMessageBox.critical(self, "Error", f"Audio file not found: {audio_path}")
            return

        audio, sr = sf.read(audio_path)
        time = np.arange(0, len(audio)) / sr
        self.ax.clear()
        self.ax.plot(time, audio, color='#3498db', linewidth=2)
        self.ax.set_xlabel("Time (seconds)")
        self.ax.set_ylabel("Amplitude")
        self.ax.set_title("Audio Waveform")
        self.ax.grid(True)
        self.figure.tight_layout()
        self.canvas.draw()

        self.audio_path = audio_path
        self.audio_path_text.setPlainText(audio_path)

        self.media_player.setMedia(QMediaContent(QUrl.fromLocalFile(audio_path)))

    def play_audio(self):
        if self.media_player.mediaStatus() == QMediaPlayer.NoMedia:
            QMessageBox.warning(self, "Warning", "No audio file loaded.")
            return
        if self.media_player.state() == QMediaPlayer.PlayingState:
            self.media_player.pause()
        else:
            self.media_player.play()

    def update_progress_bar(self, position):
        self.progress_bar.setValue(position)

    def update_duration(self, duration):
        self.progress_bar.setRange(0, duration)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls and urls[0].isLocalFile():
            file_path = urls[0].toLocalFile()
            if file_path.lower().endswith(('.wav', '.mp3')):
                self.load_audio(file_path)
            else:
                QMessageBox.warning(self, "Warning", "Unsupported file format. Please drop a WAV or MP3 file.")

def main():
    app = QApplication(sys.argv)
    window = AudioCaptionInterface()

    # File dialog to select audio file
    options = QFileDialog.Options()
    file_path, _ = QFileDialog.getOpenFileName(window, "Select Audio File", "", "Audio Files (*.wav *.mp3);;All Files (*)", options=options)
    if file_path:
        window.load_audio(file_path)

    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()