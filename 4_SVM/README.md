# 4. Support Vector Machine (SVM)

## Model Overview
SVM finds the optimal decision boundary (hyperplane) to separate classes. Uses the RBF kernel to handle non-linear patterns.

## Dataset
**Custom Fake Fruit Measurement Dataset** (created with NumPy — no download needed)
- 45 samples (15 per fruit), 3 features
- Features: `weight (g)`, `diameter (cm)`, `sweetness (1–10)`
- Labels: `0=Apple`, `1=Banana`, `2=Orange`

## Task
Classify a fruit based on its physical measurements

## Architecture
```
StandardScaler → SVC(kernel='rbf', C=1.0)
```
*(No deep learning — pure classical ML. Trains in under 1 second!)*

## How to Run on Google Colab
1. Upload `svm_model.ipynb` to [Google Colab](https://colab.research.google.com/)
2. Runtime → Change runtime type → **CPU** (default)
3. Runtime → **Run All**

## Expected Output
- Train/Test accuracy
- Classification report (precision, recall, F1)
- Confusion matrix
- Predictions: `weight=182g, diameter=7.5cm, sweetness=6 → Apple`
