
# 👚 Zalando Fashion Classifier

A simple Streamlit app that classifies Zalando-style fashion images using a CNN model trained on Fashion MNIST.

---

## 🌐 Features

- Upload a fashion image (`.png`, `.jpg`, `.jpeg`)
- Predicts one of 10 fashion categories
- Colored border based on prediction class
- Displays prediction probability and confidence bar chart
- Saves recently uploaded images
- Shows class-wise confusion matrix & accuracy chart
- Downloads prediction CSV
- GPT-based automatic model performance summary

---

## 🧠 Model

- CNN trained on Fashion MNIST (28x28 grayscale images)
- Model file: `cnn_fashion_model.keras`
- Classes: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot

---

## 🚀 How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
