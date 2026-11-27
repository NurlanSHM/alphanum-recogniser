# CharNet32 — Handwritten Character Recogniser

A small convolutional neural network that recognises handwritten digits and letters.

## Overview

- **Model:** `CharNet32` — 3-block CNN + classifier head (feature dim 1024 → hidden 128 → 47 character classes)
- **Training data:** EMNIST Balanced (digits + letters)
- **Input:** 32×32 grayscale glyph, cropped/scaled/centred from a 280×280 drawing canvas
- **Output:** top-5 predictions with confidence bars, plus online corrections

## Files

- `main.py` — model, EMNIST training/eval pipeline, Tkinter drawing app, feedback & corrections
- `train.py` — CLI training entrypoint
- `export.py` — ONNX export for Netron
- `live_web.py` / `draw_web.py` — visualisation helpers

## Usage

- `python main.py` — train (if needed) and open the drawing app
- `python train.py --epochs 30` — CLI training
- `python export.py` — export ONNX

## Model weights

Trained checkpoints (`.pth` / `.pt`) are intentionally excluded from the repository.
