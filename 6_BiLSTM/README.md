# 6. Bidirectional LSTM (BiLSTM)

## Model Overview
BiLSTM reads a sequence both **forward** and **backward** at the same time, capturing context from both directions. More powerful than regular LSTM for rating/sequence patterns.

## Dataset
**Custom Fake Product Review Score Dataset** (created with NumPy — no download needed)
- 40 samples (20 good, 20 bad products)
- Features: `quality`, `price_value`, `delivery`, `support` (scores out of 10)
- Labels: `0=Bad Product`, `1=Good Product`

## Task
Binary prediction: Is this product **Good** or **Bad** based on its scores?

## Architecture
```
Input(4 timesteps, 1 feature) → Bidirectional(LSTM(16)) → Dropout(0.2) → Dense(1, Sigmoid)
```

## How to Run on Google Colab
1. Upload `bilstm_model.ipynb` to [Google Colab](https://colab.research.google.com/)
2. Runtime → Change runtime type → **CPU** (default)
3. Runtime → **Run All**

## Expected Output
- Train/Test accuracy after training
- Accuracy plot over epochs
- Predictions: `[9,8,9,9] → GOOD ✅`, `[2,3,2,2] → BAD ❌`
