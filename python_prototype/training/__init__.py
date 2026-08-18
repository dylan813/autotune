"""Causal F0 model training (Milestone 7).

Goal: a small CNN F0 estimator that runs causally (no future lookahead),
trains on synthetic vocal-like data with known F0, and exports to ONNX
for use in the JUCE plugin.

Modules:
    model.py            : the causal CNN.
    synthetic_data.py   : on-the-fly training data generator with ground truth.
    train.py            : training loop (PyTorch).
    export_onnx.py      : exports a trained checkpoint to ONNX.
"""
