from collections import Counter, deque
import numpy as np
import time

from shared_config import (
  WINDOW_SIZE,
  PREDICT_EVERY_N_FRAMES,
  VOTE_WINDOW,
  MIN_VOTE_COUNT,
  MIN_AVG_CONF,
  COOLDOWN_FRAMES,
  LLM_IDLE_SECONDS,
)


class RealtimeRecognizer:
  # Initialize runtime smoothing and LLM trigger state
  def __init__(self, model_runtime, llm_rewriter):
    self.model_runtime = model_runtime
    self.llm_rewriter = llm_rewriter
    self.frame_buffer = deque(maxlen=WINDOW_SIZE)
    self.pred_history = deque(maxlen=VOTE_WINDOW)
    self.confirmed_words = []
    self.last_confirmed_word = None
    self.cooldown = 0
    self.frame_counter = 0
    self.current_word = ""
    self.current_conf = 0.0
    self.state = "NO_HANDS"
    self.last_confirmed_time = None
    self.llm_generated_text = ""
    self.llm_dirty = False

  # Reset all runtime and LLM state
  def reset(self):
    self.frame_buffer.clear()
    self.pred_history.clear()
    self.confirmed_words = []
    self.last_confirmed_word = None
    self.cooldown = 0
    self.frame_counter = 0
    self.current_word = ""
    self.current_conf = 0.0
    self.state = "NO_HANDS"
    self.last_confirmed_time = None
    self.llm_generated_text = ""
    self.llm_dirty = False

  # Decrease cooldown timer
  def _tick_cooldown(self):
    if self.cooldown > 0:
      self.cooldown -= 1

  # Build output payload for UI
  def _build_payload(self, confirmed_word=None):
    return {
      "state": self.state,
      "current_word": self.current_word,
      "current_conf": self.current_conf,
      "confirmed_word": confirmed_word,
      "recognized_text": " ".join(self.confirmed_words),
      "llm_text": self.llm_generated_text,
    }

  # Trigger LLM rewrite after idle time
  def _maybe_run_llm(self):
    if not self.confirmed_words:
      return

    if not self.llm_dirty:
      return

    if self.last_confirmed_time is None:
      return

    if time.time() - self.last_confirmed_time < LLM_IDLE_SECONDS:
      return

    self.llm_generated_text = self.llm_rewriter.rewrite(self.confirmed_words)
    self.llm_dirty = False

  # Run one online update step
  def update(self, frame_feature, has_hands):
    self.frame_counter += 1
    self._tick_cooldown()
    self._maybe_run_llm()

    if not has_hands:
      self.state = "NO_HANDS"
      self.current_word = ""
      self.current_conf = 0.0
      self.frame_buffer.clear()
      self.pred_history.clear()
      return self._build_payload()

    self.frame_buffer.append(frame_feature.astype(np.float32))

    if len(self.frame_buffer) < WINDOW_SIZE:
      self.state = "FILLING"
      return self._build_payload()

    if self.frame_counter % PREDICT_EVERY_N_FRAMES != 0:
      self.state = "PREDICTING"
      return self._build_payload()

    raw_window = np.stack(list(self.frame_buffer), axis=0)
    pred_word, pred_conf, _ = self.model_runtime.predict_window(raw_window)

    self.current_word = pred_word
    self.current_conf = pred_conf
    self.pred_history.append((pred_word, pred_conf))

    confirmed_word = self._try_confirm_word()

    if confirmed_word is not None:
      self.state = "COOLDOWN"
      return self._build_payload(confirmed_word=confirmed_word)

    self.state = "PREDICTING"
    return self._build_payload()

  # Confirm a stable word from recent predictions
  def _try_confirm_word(self):
    if self.cooldown > 0:
      return None

    if len(self.pred_history) < VOTE_WINDOW:
      return None

    words = [word for word, _ in self.pred_history]
    counts = Counter(words)
    best_word, best_count = counts.most_common(1)[0]

    if best_count < MIN_VOTE_COUNT:
      return None

    confs = [conf for word, conf in self.pred_history if word == best_word]
    avg_conf = float(sum(confs) / len(confs))

    if avg_conf < MIN_AVG_CONF:
      return None

    if best_word == self.last_confirmed_word:
      return None

    self.confirmed_words.append(best_word)
    self.last_confirmed_word = best_word
    self.cooldown = COOLDOWN_FRAMES
    self.pred_history.clear()
    self.last_confirmed_time = time.time()
    self.llm_dirty = True
    self.llm_generated_text = ""

    return best_word