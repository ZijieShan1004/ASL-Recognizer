import os


APP_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.join(APP_DIR, ".."))

MODEL_PATH = os.path.join(ROOT_DIR, "best_model_v2.pt")
LABEL_MAP_PATH = os.path.join(ROOT_DIR, "label_mapping_v2.json")
CLASSES_PATH = os.path.join(ROOT_DIR, "MSASL_classes.json")
HAND_MODEL_PATH = os.path.join(ROOT_DIR, "models", "hand_landmarker.task")

WINDOW_SIZE = 64
RAW_FRAME_DIM = 126
INPUT_DIM = 252
HIDDEN = 256
NUM_LAYERS = 2
DROPOUT = 0.3

NUM_HANDS = 2
MIN_HAND_DETECTION_CONFIDENCE = 0.3
MIN_HAND_PRESENCE_CONFIDENCE = 0.3
MIN_TRACKING_CONFIDENCE = 0.3

PREDICT_EVERY_N_FRAMES = 2
VOTE_WINDOW = 4
MIN_VOTE_COUNT = 2
MIN_AVG_CONF = 0.8
COOLDOWN_FRAMES = 10

CAMERA_INDEX = 0
TIMER_INTERVAL_MS = 30
DISPLAY_WIDTH = 960
DISPLAY_HEIGHT = 720

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:1.5b"
LLM_IDLE_SECONDS = 1.8
LLM_TIMEOUT_SECONDS = 20