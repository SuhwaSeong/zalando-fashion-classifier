# 📄 app.py

import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from PIL import Image, ImageOps
import os
import random

# 🏷️ 클래스 레이블 정의
labels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
          "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# 🎯 모델 로딩
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cnn_fashion_model.keras")

model = load_model()

# 🎨 테두리 색상 매핑
border_colors = {
    "T-shirt/top": "gray", "Trouser": "olive", "Pullover": "purple",
    "Dress": "pink", "Coat": "brown", "Sandal": "orange",
    "Shirt": "teal", "Sneaker": "blue", "Bag": "green", "Ankle boot": "red"
}

# 📌 앱 시작
st.title("👚 Zalando Fashion Classifier")
st.write("Upload a fashion image to predict its category.")

uploaded_file = st.file_uploader("📁 Upload an image", type=["png", "jpg", "jpeg"])
recent_folder = "recent_uploads"
os.makedirs(recent_folder, exist_ok=True)

if uploaded_file:
    image = Image.open(uploaded_file).convert("L")
    image = image.resize((28, 28))
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    # 🔮 예측 수행
    prediction = model.predict(img_array)[0]
    predicted_class = labels[np.argmax(prediction)]
    confidence = np.max(prediction)

    # 🎨 테두리 이미지 생성
    color = border_colors.get(predicted_class, "black")
    bordered_image = ImageOps.expand(image, border=8, fill=color)
    st.image(bordered_image, caption=f"🎯 Predicted: {predicted_class}", use_column_width=False)

    # 🔢 확률 차트
    st.subheader(f"🔍 Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence * 100:.2f}%")
    fig, ax = plt.subplots()
    ax.barh(labels, prediction)
    ax.set_xlabel("Probability")
    ax.set_title("Prediction Probabilities")
    st.pyplot(fig)

    # 저장
    recent_path = os.path.join(recent_folder, uploaded_file.name)
    bordered_image.save(recent_path)
    st.success("📥 Image saved to recent uploads.")

# 📉 Confusion Matrix 시각화
def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    st.pyplot(fig)

# 📈 클래스별 정확도 시각화
def plot_class_accuracy(y_true, y_pred, class_names):
    df = pd.DataFrame({'true': y_true, 'pred': y_pred})
    class_acc = df.groupby('true').apply(lambda x: accuracy_score(x['true'], x['pred']))
    fig, ax = plt.subplots()
    class_acc[class_names].plot(kind='bar', ax=ax)
    plt.title("Class-wise Accuracy")
    plt.ylabel("Accuracy")
    st.pyplot(fig)

st.markdown("---")
if st.checkbox("📉 Show Confusion Matrix and Class Accuracy"):
    try:
        df = pd.read_csv("fashion_predictions.csv")
        y_true = df["actual"]
        y_pred = df["predicted"]
        plot_confusion_matrix(y_true, y_pred, labels)
        plot_class_accuracy(y_true, y_pred, labels)
    except Exception as e:
        st.error(f"❌ Error loading prediction file: {e}")

# ⬇️ 예측 CSV 다운로드
if st.checkbox("⬇️ Download Prediction CSV"):
    try:
        df = pd.read_csv("fashion_predictions.csv")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, "fashion_predictions.csv", "text/csv")
    except:
        st.warning("No prediction CSV found.")

# 🖼️ 최근 업로드 이미지 갤러리
if st.checkbox("🖼️ Show Recently Uploaded Images"):
    files = os.listdir(recent_folder)
    if files:
        cols = st.columns(min(5, len(files)))
        for i, file in enumerate(files[:10]):
            img = Image.open(os.path.join(recent_folder, file))
            cols[i % len(cols)].image(img, caption=file, use_column_width=True)
    else:
        st.info("No images found in recent uploads.")

# 🧠 GPT 요약
if st.checkbox("🧠 GPT-based Model Summary"):
    try:
        import openai
        openai.api_key = st.text_input("🔑 Enter your OpenAI API key", type="password")
        if openai.api_key:
            df = pd.read_csv("fashion_predictions.csv")
            summary = df.groupby(['actual', 'predicted']).size().reset_index(name='count')
            prompt = f"Summarize this model performance:\n\n{summary.to_string(index=False)}"
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}]
            )
            st.markdown("### 🧠 GPT Summary:")
            st.write(response.choices[0].message['content'])
    except Exception as e:
        st.error(f"GPT summary failed: {e}")
# Streamlit 전체 앱 코드를 여기에 작성
