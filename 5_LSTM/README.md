# 5. Long Short-Term Memory (LSTM)

## Model Overview
LSTM is an advanced RNN with memory gates (input, forget, output) that help it remember long-term patterns. Much better than RNN at spotting trends over time.

## Dataset
**Custom Fake Stock Price Dataset** (created with NumPy — no download needed)
- 30 sequences of 5 consecutive daily stock prices
- Features: normalized price values (0–1)
- Labels: `0=DOWN (Don't Buy)`, `1=UP (Buy)`

## Task
Binary prediction: Based on 5 days of prices, will the stock go **UP** or **DOWN**?

## Architecture
```
Input(5 timesteps, 1 feature) → LSTM(32) → Dropout(0.2) → Dense(1, Sigmoid)
```

## How to Run on Google Colab
1. Upload `lstm_model.ipynb` to [Google Colab](https://colab.research.google.com/)
2. Runtime → Change runtime type → **CPU** (default)
3. Runtime → **Run All**

## Expected Output
- Train/Test accuracy after training
- Accuracy plot over epochs
- Predictions: `[100,103,106,109,113] → BUY ✅`, `[200,195,190,185,179] → DONT BUY ❌`
