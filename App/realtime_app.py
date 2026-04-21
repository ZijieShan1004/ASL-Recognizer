import sys
import cv2

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from shared_config import CAMERA_INDEX, TIMER_INTERVAL_MS
from model_runtime import ASLModelRuntime
from mediapipe_live import LiveHandExtractor
from realtime_recognizer import RealtimeRecognizer
from ui_main import MainWindow
from llm_rewriter import LocalLLMRewriter


class RealtimeASLApp:
  # Initialize camera, model, extractor, recognizer, LLM, and UI
  def __init__(self):
    self.qt_app = QApplication(sys.argv)
    self.model_runtime = ASLModelRuntime()
    self.hand_extractor = LiveHandExtractor()
    self.llm_rewriter = LocalLLMRewriter()
    self.recognizer = RealtimeRecognizer(self.model_runtime, self.llm_rewriter)
    self.window = MainWindow(self._on_reset)

    self.cap = cv2.VideoCapture(CAMERA_INDEX)
    self.timer = QTimer()
    self.timer.timeout.connect(self._on_timer)

  # Reset current recognized and generated output
  def _on_reset(self):
    self.recognizer.reset()
    self.window.set_recognized_text("")
    self.window.set_sentence_text("")
    self.window.set_runtime_status("", 0.0, "NO_HANDS")

  # Process one camera frame
  def _on_timer(self):
    ok, frame = self.cap.read()
    if not ok:
      return

    frame = cv2.flip(frame, 1)
    frame_feature, has_hands, annotated = self.hand_extractor.process_frame(frame)
    result = self.recognizer.update(frame_feature, has_hands)

    if result["current_word"]:
      cv2.putText(
        annotated,
        f'{result["current_word"]} ({result["current_conf"]:.2f})',
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
        cv2.LINE_AA
      )

    cv2.putText(
      annotated,
      f'State: {result["state"]}',
      (20, 80),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.9,
      (255, 255, 255),
      2,
      cv2.LINE_AA
    )

    self.window.set_video_frame(annotated)
    self.window.set_recognized_text(result["recognized_text"])
    self.window.set_sentence_text(result["llm_text"])
    self.window.set_runtime_status(
      result["current_word"],
      result["current_conf"],
      result["state"]
    )

  # Start the desktop application loop
  def run(self):
    self.window.show()
    self.timer.start(TIMER_INTERVAL_MS)
    exit_code = self.qt_app.exec()
    self.timer.stop()
    self.cap.release()
    self.hand_extractor.close()
    sys.exit(exit_code)


# Launch the realtime GUI app
def main():
  app = RealtimeASLApp()
  app.run()


if __name__ == "__main__":
  main()