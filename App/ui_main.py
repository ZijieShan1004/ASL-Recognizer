import cv2

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
  QHBoxLayout,
  QLabel,
  QMainWindow,
  QPushButton,
  QVBoxLayout,
  QWidget,
)

from shared_config import DISPLAY_WIDTH, DISPLAY_HEIGHT


class MainWindow(QMainWindow):
  # Build the main GUI window
  def __init__(self, reset_handler):
    super().__init__()
    self.reset_handler = reset_handler
    self.setWindowTitle("ASL Realtime Demo")
    self._build_layout()

  # Create all UI widgets and layout
  def _build_layout(self):
    root = QWidget()
    self.setCentralWidget(root)

    top_layout = QHBoxLayout()
    main_layout = QVBoxLayout(root)

    self.recognized_label = QLabel("Recognized: ")
    self.recognized_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    self.recognized_label.setStyleSheet("font-size: 22px; font-weight: 600; padding: 8px;")

    self.reset_button = QPushButton("Reset")
    self.reset_button.setFixedWidth(120)
    self.reset_button.setStyleSheet("font-size: 18px; padding: 8px;")
    self.reset_button.clicked.connect(self._reset_clicked)

    top_layout.addWidget(self.recognized_label, 1)
    top_layout.addWidget(self.reset_button, 0)

    self.video_label = QLabel()
    self.video_label.setAlignment(Qt.AlignCenter)
    self.video_label.setFixedSize(DISPLAY_WIDTH, DISPLAY_HEIGHT)
    self.video_label.setStyleSheet("background-color: black;")

    self.current_label = QLabel("Current: ")
    self.current_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    self.current_label.setStyleSheet("font-size: 18px; padding: 6px;")

    self.state_label = QLabel("State: NO_HANDS")
    self.state_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    self.state_label.setStyleSheet("font-size: 18px; padding: 6px;")

    self.sentence_label = QLabel("Sentence: ")
    self.sentence_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    self.sentence_label.setWordWrap(True)
    self.sentence_label.setStyleSheet("font-size: 20px; padding: 8px; color: #00aaee;")

    main_layout.addLayout(top_layout)
    main_layout.addWidget(self.video_label, 0, Qt.AlignCenter)
    main_layout.addWidget(self.current_label)
    main_layout.addWidget(self.state_label)
    main_layout.addWidget(self.sentence_label)

  # Forward reset action to controller
  def _reset_clicked(self):
    self.reset_handler()

  # Update the recognized words line
  def set_recognized_text(self, text):
    self.recognized_label.setText(f"Recognized: {text}")

  # Update the natural sentence line
  def set_sentence_text(self, text):
    self.sentence_label.setText(f"Sentence: {text}")

  # Update current prediction and runtime state
  def set_runtime_status(self, current_word, current_conf, state):
    if current_word:
      self.current_label.setText(f"Current: {current_word} ({current_conf:.2f})")
    else:
      self.current_label.setText("Current: ")
    self.state_label.setText(f"State: {state}")

  # Show one BGR frame in the UI
  def set_video_frame(self, frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, c = frame_rgb.shape
    bytes_per_line = c * w
    image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
    pixmap = QPixmap.fromImage(image)
    scaled = pixmap.scaled(
      self.video_label.width(),
      self.video_label.height(),
      Qt.KeepAspectRatio,
      Qt.SmoothTransformation
    )
    self.video_label.setPixmap(scaled)