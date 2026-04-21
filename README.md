# ASL-Recognizer

A local American Sign Language recognition system built with MediaPipe, PyTorch, and a local open-source LLM.

This project supports two deployment modes:

1. **Video Recognition**
   - Run ASL recognition on a prerecorded video file

2. **Realtime Webcam Recognition**
   - Run realtime ASL recognition with live webcam input, hand landmark visualization, and local LLM sentence rewriting

The system is fully local. It does not require any paid API.

Please check Deploy Instructions.md for more instructions.
---

## Features

- ASL word recognition from prerecorded videos
- Realtime ASL word recognition from a webcam
- MediaPipe hand landmark extraction
- Live hand skeleton overlay
- BiGRU + Attention sequence model for sign classification
- Local LLM sentence refinement with Ollama + Qwen
- Desktop GUI with recognized words, reset button, and rewritten sentence output
- Fully local deployment for both offline and realtime inference

---

## Project Overview

This project is designed as a local end-to-end ASL recognition pipeline.

### Stage 1: Sign Recognition
- Input frames are processed with MediaPipe Hand Landmarker
- Left and right hand landmarks are extracted
- Landmark coordinates are normalized
- Velocity features are added
- A BiGRU-based sequence model predicts ASL words

### Stage 2: Language Refinement
- Recognized ASL-like word sequences are buffered
- After a short idle period, the sequence is sent to a local LLM
- The local LLM rewrites the output into smoother English

Example:

- Recognized words: `how you`
- Rewritten sentence: `How are you?`

---

## Tech Stack

- Python
- PyTorch
- MediaPipe Tasks
- OpenCV
- PySide6
- Ollama
- Qwen 2.5 Instruct

---

## Repository Structure

```text
ASL-Recognizer/
├── app/
│   ├── shared_config.py
│   ├── model_runtime.py
│   ├── mediapipe_live.py
│   ├── realtime_recognizer.py
│   ├── ui_main.py
│   ├── llm_rewriter.py
│   └── realtime_app.py
├── scripts/
│   ├── infer_sign_video.py
│   ├── train_final_bigru.py
│   ├── extract_msasl1000_mediapipe.py
│   └── ...
├── models/
│   └── hand_landmarker.task
├── best_model_v2.pt
├── label_mapping_v2.json
├── MSASL_classes.json
└── README.md
