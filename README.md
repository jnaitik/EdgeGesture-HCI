---
title: EdgeGesture-HCI
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: gradio
app_file: src/app.py
pinned: false
---

# 🚀 EdgeGesture-HCI

### Real-Time Contactless Hand Gesture Recognition for Spacecraft Control

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/TensorFlow-2.15+-orange?logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/MediaPipe-0.10+-green?logo=google&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-1.58+-red?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Accuracy-91.72%25-brightgreen" />
</p>

<p align="center">
  A deep learning-based gesture recognition system designed for contactless spacecraft navigation in microgravity environments. Built with a hybrid <b>CNN-GRU</b> architecture, achieving <b>91.72% test accuracy</b> across 11 gesture classes.
</p>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Gesture Classes](#gesture-classes)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Adversarial Robustness](#adversarial-robustness)
- [Tech Stack](#tech-stack)
- [Team](#team)
- [References](#references)

---

## Overview

Traditional input devices (keyboards, mice, touchscreens) are impractical in microgravity environments aboard spacecraft, where physical contact with surfaces can cause unintended body movement. **EdgeGesture-HCI** provides a contactless alternative — astronauts can control spacecraft systems using natural hand gestures captured through a standard webcam.

The system uses **MediaPipe** for real-time 21-point 3D hand landmark detection and a **CNN-GRU** deep learning model for gesture classification. It is trained with space-specific data augmentation (microgravity jitter, scale variation, EVA glove occlusion) to ensure robustness in orbital environments.

**Developed as an ISRO-affiliated research project at MANIT Bhopal.**

---

## Features

- 🎯 **Real-time gesture detection** at ~30 FPS using webcam
- 🧠 **Hybrid CNN-GRU model** — 91.72% test accuracy across 11 gesture classes
- 🛡️ **Adversarial robustness testing** — 5 simulated space conditions
- 🚀 **Spacecraft cockpit simulation** — live navigation and attitude control panels
- 📊 **Metrics dashboard** — real-time FPS, confidence tracking, gesture distribution
- 🔄 **Dual classification modes** — deep learning model with rule-based fallback
- 📷 **Multi-camera support** — built-in and USB webcam selection
- 🧤 **Space-robust training** — augmented with jitter, scaling, and fingertip occlusion

---

## System Architecture

```
┌──────────────┐    ┌─────────────────────┐    ┌──────────────────┐    ┌──────────────┐
│   Webcam     │───▶│  MediaPipe Hand     │───▶│  CNN-GRU Model   │───▶│  Spacecraft   │
│   Input      │    │  Landmarker (21pts) │    │  (11 classes)    │    │  Command      │
└──────────────┘    └─────────────────────┘    └──────────────────┘    └──────────────┘
                              │                         │
                              ▼                         ▼
                    ┌─────────────────────┐    ┌──────────────────┐
                    │  63-dim Feature     │    │  Adversarial     │
                    │  Vector (21×3)      │    │  Engine (5 tests)│
                    └─────────────────────┘    └──────────────────┘
```

---

## Gesture Classes

| Gesture | Spacecraft Command | Type |
|---------|-------------------|------|
| ☝️ Up | Translate Up | Translation |
| 👇 Down | Translate Down | Translation |
| 👈 Left | Translate Left | Translation |
| 👉 Right | Translate Right | Translation |
| 🤚 Forward | Thrust Forward | Translation |
| ✊ Backward | Thrust Backward | Translation |
| 🔼 Pitch Up | Pitch Up (nose up) | Rotation |
| 🔽 Pitch Down | Pitch Down (nose down) | Rotation |
| ↩️ Yaw Left | Yaw Left (turn left) | Rotation |
| ↪️ Yaw Right | Yaw Right (turn right) | Rotation |
| 🖐️ Background | No Command (idle) | Idle |

---

## Project Structure

```
EdgeGesture-HCI/
│
├── src/
│   ├── __init__.py              # Package init
│   ├── app.py                   # Main Streamlit application
│   ├── config.py                # Gesture labels, commands, colors, settings
│   ├── hand_tracker.py          # MediaPipe hand landmark detection (Tasks API)
│   ├── gesture_classifier.py    # CNN-GRU model + rule-based fallback
│   ├── adversarial_engine.py    # Space condition simulation (5 conditions)
│   ├── spacecraft_ui.py         # Cockpit navigation & attitude panels
│   ├── metrics_dashboard.py     # Real-time performance tracking
│   └── create_subset.py         # Dataset utility
│
├── models/
│   ├── gesture_model_robust.h5  # Trained CNN-GRU model (~4MB)
│   └── label_encoder.pkl        # Sklearn label encoder
│
├── paper/
│   ├── main.tex                 # IEEE research paper (LaTeX)
│   ├── generate_figures.py      # Script to generate paper figures
│   └── figures/                 # Generated charts and diagrams
│
├── hand_gesture_preprocessing.ipynb  # Data preprocessing + model training notebook
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

---

## Installation

### Prerequisites
- Python 3.10+
- Webcam (built-in or USB)

### Setup

```bash
# Clone the repository
git clone https://github.com/labhanshgoyal/EdgeGesture-HCI.git
cd EdgeGesture-HCI

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Run the Application

```bash
cd EdgeGesture-HCI
venv\Scripts\python.exe -m streamlit run src/app.py
```

Opens at **https://huggingface.co/spaces/jnaitik/EdgeGesture-HCI**

### How to Use

1. Select your camera source in the sidebar (0 = Built-in, 1 = USB)
2. Adjust the confidence threshold if needed (default: 60%)
3. Click **▶️ Start Camera**
4. Perform gestures in front of the camera
5. See real-time gesture detection and spacecraft commands

### Train Your Own Model

Open `hand_gesture_preprocessing.ipynb` in Google Colab or Jupyter:

```bash
jupyter notebook hand_gesture_preprocessing.ipynb
```

The notebook handles:
- Frame extraction from gesture videos
- MediaPipe landmark detection
- Data augmentation (jitter, scale, occlusion)
- CNN-GRU model training
- Evaluation metrics and confusion matrix

---

## Model Architecture

```
Input (63,) ──▶ Reshape (21, 3)
                    │
            ┌───────▼────────┐
            │  Conv1D (64)   │──▶ BatchNorm
            │  Conv1D (128)  │──▶ BatchNorm
            │  Conv1D (256)  │──▶ BatchNorm
            └───────┬────────┘
                    │
            ┌───────▼────────┐
            │   GRU (128)    │
            └───────┬────────┘
                    │
            ┌───────▼────────┐
            │ Dense(256)+Drop│──▶ Dropout(0.5)
            │ Dense(128)+Drop│──▶ Dropout(0.3)
            │ Dense(11)      │──▶ Softmax
            └────────────────┘
                    │
              11 Gesture Classes
```

**Training:** Adam optimizer, sparse categorical cross-entropy, early stopping (patience=15), LR reduction on plateau.

---

## Results

### Classification Performance

| Metric | Score |
|--------|-------|
| **Test Accuracy** | **91.72%** |
| Macro F1-Score | 0.89 |
| Weighted F1-Score | 0.90 |
| Macro Precision | 0.89 |
| Macro Recall | 0.89 |

### Per-Class Performance

| Gesture | Precision | Recall | F1-Score |
|---------|-----------|--------|----------|
| Up | 0.97 | 1.00 | 0.99 |
| Pitch Down | 1.00 | 0.97 | 0.98 |
| Down | 0.97 | 0.97 | 0.97 |
| Pitch Up | 0.94 | 0.97 | 0.95 |
| Right | 0.92 | 0.97 | 0.94 |
| Background | 1.00 | 0.85 | 0.92 |
| Forward | 1.00 | 0.86 | 0.92 |
| Backward | 0.91 | 0.91 | 0.91 |
| Left | 0.87 | 0.93 | 0.90 |
| Yaw Left | 0.67 | 0.69 | 0.68 |
| Yaw Right | 0.57 | 0.63 | 0.60 |

---

## Adversarial Robustness

The model is tested under 5 simulated space conditions:

| Condition | Purpose | Best Accuracy (under stress) |
|-----------|---------|------------------------------|
| 🌊 Microgravity Jitter | Hand trembling in zero-G | 64.24% (σ=0.01) |
| 📡 Sensor Noise | Radiation-induced interference | 88.48% (σ=0.005) |
| 🧤 Glove Occlusion | EVA pressurized gloves | 92.42% (1 fingertip) |
| 💡 Lighting Variation | Orbital sun/shadow transitions | Tested via HSV scaling |
| ⏱️ Communication Latency | Processing delay simulation | Frame buffering test |

**Key finding:** Model maintains >88% accuracy under moderate sensor noise and >77% with 3 fingertips occluded.

---

## Tech Stack

| Technology | Purpose |
|------------|---------|
| **TensorFlow/Keras** | CNN-GRU model training and inference |
| **MediaPipe** | 21-point 3D hand landmark detection |
| **OpenCV** | Webcam capture and frame processing |
| **Streamlit** | Web application UI |
| **NumPy** | Numerical computation |
| **Scikit-learn** | Label encoding, metrics |
| **Matplotlib** | Research paper figure generation |

---

## Team

<!-- UPDATE: Add your actual team member names -->

| Member | Role |
|--------|------|
| **Member 1** | Preprocessing & Model Training |
| **Member 2** | Adversarial Engine & Robustness Testing |
| **Member 3** | Spacecraft UI & Metrics Dashboard |
| **Member 4** | Research Paper & Documentation |

**Institution:** Maulana Azad National Institute of Technology (MANIT), Bhopal
**Affiliated with:** Indian Space Research Organisation (ISRO)

---

## References

1. "Hierarchical Attention-Based Astronaut Gesture Recognition: A Dataset and CNN Model" — Reference dataset paper
2. [MediaPipe Hands](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) — Google's hand landmark detection
3. [TensorFlow](https://www.tensorflow.org/) — Deep learning framework
4. [Streamlit](https://streamlit.io/) — Web application framework

---

## License

This project is developed for academic and research purposes under ISRO-MANIT collaboration.

---

<p align="center">
  <b>Built with ❤️ at MANIT Bhopal for ISRO Space Missions</b>
</p>
