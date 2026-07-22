# 3. Recurrent Neural Network (RNN)

## Model Overview
RNN processes data in sequence, remembering past values through a hidden state. Great for time-series and sequential patterns.

## Dataset
**Custom Fake Temperature Dataset** (created with NumPy — no download needed)
- 40 fake daily temperature readings (in Celsius)
- Sequences of 3 past days used to predict next day
- Labels: `0=Cold` (temp < 20°C), `1=Hot` (temp ≥ 20°C)

## Task
Binary prediction: Will tomorrow be **Hot** or **Cold** based on last 3 days?

## Architecture
```
Input(3 timesteps, 1 feature) → SimpleRNN(16) → SimpleRNN(8) → Dense(1, Sigmoid)
```

## How to Run on Google Colab
1. Upload `rnn_model.ipynb` to [Google Colab](https://colab.research.google.com/)
2. Runtime → Change runtime type → **CPU** (default)
3. Runtime → **Run All**

## Expected Output
- Train/Test accuracy after training
- Accuracy plot over epochs
- Predictions: `[Hot, Hot, Hot] → Hot`, `[Cold, Cold, Cold] → Cold`
