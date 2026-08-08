# Machine Learning-Based Pre-Departure Fatigue Detection for Bus Drivers

This repository contains the source code developed for the thesis:

**Machine Learning-Based Pre-Departure Fatigue Detection for Bus Drivers**

The study proposes a visual fatigue detection system for bus drivers using facial landmarks, traditional machine learning, dual-branch 2D convolutional neural networks, geometric mouth filtering, and real-time duration-based fatigue judgement.

## Main Components

### Eye-State CNN
`train_2d_cnn_eyes.py`

Trains the two-dimensional convolutional neural network for binary classification of open and closed eye states.

### Mouth-State CNN
`train_2d_cnn_mouth.py`

Trains the two-dimensional convolutional neural network for classification of normal mouth and yawning states.

### Facial Feature Extraction
`extract_features.py`

Extracts facial geometric and temporal features used by the traditional machine-learning baseline.

### Traditional Machine-Learning Baseline
`train_baseline_models.py`

Evaluates traditional machine-learning classifiers using extracted facial features.

### Yawning Detection Evaluation
`evaluate_yawdd_performance.py`

Evaluates the performance of the mouth-state and yawning detection components.

### Video-Level System Evaluation
`test_videos_with_real_system.py`

Evaluates the complete fatigue detection system on labelled video data.

### Real-Time Fatigue Detection Prototype
`realtime_demo_2dcnn.py`

Implements the real-time visual fatigue detection prototype using eye-state and mouth-state CNN models.

## System Overview

The proposed system uses:

- RGB video input
- MediaPipe Face Mesh
- Eye and mouth Regions of Interest (ROIs)
- Dual-branch 2D-CNN models
- Mouth geometric filtering
- Real-time duration-based state judgement
- Fatigue alarm generation

The system focuses primarily on detecting prolonged eye closure and yawning for pre-departure fatigue screening of bus drivers.

## Research Purpose

This repository is provided as supplementary material for academic research and thesis reproducibility.

## Author

Ma Wei
