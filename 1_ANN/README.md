# 1. Artificial Neural Network (ANN)

## Model Overview
A feedforward neural network with fully connected (Dense) layers. Learns which combination of symptoms maps to a disease.

## Dataset
**Custom Fake Symptom Dataset** (created with NumPy — no download needed)
- 30 samples, 3 features, 3 classes
- Features: `fever`, `cough`, `headache` (binary: 1=Yes, 0=No)
- Labels: `0=Healthy`, `1=Cold`, `2=Flu`

## Task
Multi-class disease classification from binary symptom inputs

## Architecture
```
Input(3) → Dense(16, ReLU) → Dropout(0.2) → Dense(16, ReLU) → Dense(3, Softmax)
```

## How to Run on Google Colab
1. Upload `ann_model.ipynb` to [Google Colab](https://colab.research.google.com/)
2. Runtime → Change runtime type → **CPU** (default)
3. Runtime → **Run All**

## Expected Output
- Train/Test accuracy printed after training
- Accuracy plot over epochs
- Confusion matrix (Healthy / Cold / Flu)
- Sample predictions: `[1,1,1] → Flu`, `[0,0,0] → Healthy`
