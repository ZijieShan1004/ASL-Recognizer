## Requirements

- Python 3.10 or newer recommended
- Windows, macOS, or Linux
- Webcam for realtime mode
- Ollama installed for local LLM rewriting
- Local model files available

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/ZijieShan1004/ASL-Recognizer.git
cd ASL-Recognizer
```

### 2. Install Python dependencies
```bash
pip install torch torchvision torchaudio
pip install mediapipe opencv-python numpy pandas scikit-learn tqdm
pip install PySide6
```
## Confirm required local model files exist
Make sure the following files are present in the project:

- models/hand_landmarker.task
- best_model_v2.pt
- label_mapping_v2.json
- MSASL_classes.json

If any of these files are stored in different locations, update the paths in the corresponding scripts or in app/shared_config.py.

## Local LLM Setup

### 1. Install Ollama
Download Ollama from the official website:

https://ollama.com/download

After installation, verify that it works:
```bash
ollama --version
```

### 2. Pull a local Qwen model
Same model: 
```bash
ollama pull qwen2.5:1.5b
```

Please remember to change the model name in shared_config.py line 36 if you want to use other LLMs.

### 3. Test the local LLM
```bash
ollama run qwen2.5:1.5b
```
Try:
```bash
Convert ASL-like words into a natural English sentence. Output only the sentence.
Words: how you
```
Expected style of output:
```bash
How are you?
```
## Local Deployment
This repository supports two local deployment modes.

## Local Deployment 1: Video Recognition
This mode runs ASL recognition locally on a prerecorded video file.

### What this mode does
-Loads a video file from disk
-Extracts MediaPipe hand landmarks
-Builds the same normalized landmark + velocity feature format used during training
-Runs the BiGRU recognition model
-Prints the predicted ASL word and confidence

### Steps
1. Open the video inference script:infer_sign_video.py
2. Check these path variables in the script and adjust them if necessary:
- model checkpoint path
- label mapping path
- MS-ASL class list path
- hand landmarker path
- input video path (Make sure to change the video name to your video!)
3. Run local video recognition from the project root:
```bash
python scripts/infer_sign_video.py
```
4. Expected output:
The script will print results such as:
- predicted label
- predicted word
- confidence
- top candidate predictions

### Use case
Use this mode when you want to:
- test a single video clip
- validate that the model checkpoint is working
- evaluate recognition results before using the realtime GUI

## Local Deployment 2: Realtime Webcam Recognition
This mode launches the local desktop application for live ASL recognition with webcam input.

### What this mode does
- Opens the webcam
- Detects hand landmarks with MediaPipe
- Draws the hand skeleton overlay on the live video
- Buffers frames into a sliding window
- Runs the BiGRU recognition model online
- Confirms stable sign words with smoothing logic
- Sends recognized word sequences to a local Qwen model through Ollama
- Displays the rewritten sentence in the GUI

### Steps
1. Confirm required files exist:
- models/hand_landmarker.task
- best_model_v2.pt
- label_mapping_v2.json
- MSASL_classes.json
2. Launch the local realtime application from the project root:
```bash
cd app
python realtime_app.py
```
3. Expected GUI behavior:
- The webcam feed opens
- MediaPipe hand landmarks are drawn on detected hands
- The top bar shows confirmed ASL words
- The bottom sentence field shows the local LLM-refined English output
- The reset button clears current recognition results

### Example flow
Recognized words: how you
Rewritten sentence: How are you?

### Use case
Use this mode when you want to:
- demonstrate the full local system
- test realtime sign recognition
- showcase the local LLM rewriting step
- record a live project demo

### Realtime Tuning
If realtime recognition feels too conservative or too slow, adjust these values in app/shared_config.py:

- PREDICT_EVERY_N_FRAMES
- VOTE_WINDOW
- MIN_VOTE_COUNT
- MIN_AVG_CONF
- COOLDOWN_FRAMES

A more responsive example:
```bash
PREDICT_EVERY_N_FRAMES = 2
VOTE_WINDOW = 6
MIN_VOTE_COUNT = 4
MIN_AVG_CONF = 0.60
COOLDOWN_FRAMES = 15
```

A more aggressive example:
```bash
PREDICT_EVERY_N_FRAMES = 1
VOTE_WINDOW = 5
MIN_VOTE_COUNT = 3
MIN_AVG_CONF = 0.55
COOLDOWN_FRAMES = 12
```

## Notes
- The sign recognizer was originally trained as a clip-level classifier, so realtime deployment depends heavily on buffering, smoothing, and stable execution of each sign.
- The local LLM is only used for sentence refinement. It is not used for visual sign recognition.
- For the best realtime demo results, perform one sign at a time and pause briefly between words.

## Acknowledgments
- Microsoft MS-ASL dataset
- MediaPipe Hand Landmarker
- Ollama
- Qwen open-source language models
