import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from PIL import Image, ImageOps
import os
from openai import OpenAI

# 클래스 레이블 정의 / Define class labels
labels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
          "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# 모델 로드 / Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cnn_fashion_model.keras")

model = load_model()

# 테두리 색상 / Set border color by predicted class
border_colors = {
    "T-shirt/top": "gray", "Trouser": "olive", "Pullover": "purple",
    "Dress": "pink", "Coat": "brown", "Sandal": "orange",
    "Shirt": "teal", "Sneaker": "blue", "Bag": "green", "Ankle boot": "red"
}

# UI 시작 / App UI start
st.title("👚 Zalando 패션 이미지 분류기 / Zalando Fashion Classifier")
st.write("Zalando 스타일의 패션 이미지를 업로드하세요.\nUpload a fashion image to classify it.")

uploaded_file = st.file_uploader("📁 이미지 업로드 / Upload image", type=["png", "jpg", "jpeg"])
recent_folder = "recent_uploads"
os.makedirs(recent_folder, exist_ok=True)

if uploaded_file:
    image = Image.open(uploaded_file).convert("L")
    image = image.resize((28, 28))
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    prediction = model.predict(img_array)[0]
    predicted_class = labels[np.argmax(prediction)]
    confidence = np.max(prediction)

    color = border_colors.get(predicted_class, "black")
    bordered_image = ImageOps.expand(image, border=8, fill=color)
    st.image(bordered_image, caption=f"🎯 예측 / Predicted: {predicted_class}", use_container_width=True)

    st.subheader(f"🔍 예측 결과 / Prediction: {predicted_class}")
    st.write(f"Confidence (신뢰도): {confidence * 100:.2f}%")
    fig, ax = plt.subplots()
    ax.barh(labels, prediction)
    ax.set_xlabel("Probability / 예측 확률")
    st.pyplot(fig)

    actual_label = st.selectbox("✅ 실제 레이블을 선택하세요 / Select the actual label", labels)

    if st.button("📏 예측 저장 / Save Prediction"):
        csv_path = "fashion_predictions.csv"
        new_data = pd.DataFrame([{
            "filename": uploaded_file.name,
            "Predicted Label": predicted_class,
            "True Label": actual_label
        }])
        if os.path.exists(csv_path):
            existing = pd.read_csv(csv_path)
            updated = pd.concat([existing, new_data], ignore_index=True)
        else:
            updated = new_data
        updated.to_csv(csv_path, index=False)
        st.success("✅ 예측이 저장되었습니다 / Prediction saved!")

    recent_path = os.path.join(recent_folder, uploaded_file.name)
    bordered_image.save(recent_path)
    st.success("📅 최근 업로드에 이미지 저장됨 / Image saved to recent uploads.")

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    ax.set_xlabel('예측 / Predicted')
    ax.set_ylabel('실제 / Actual')
    ax.set_title('Confusion Matrix / 혼동 행렬')
    st.pyplot(fig)

def plot_class_accuracy(y_true, y_pred, class_names):
    df = pd.DataFrame({'true': y_true, 'pred': y_pred})
    class_acc = df.groupby('true').apply(lambda x: accuracy_score(x['true'], x['pred']))
    fig, ax = plt.subplots()
    class_acc[class_names].plot(kind='bar', ax=ax)
    ax.set_title("클래스별 정확도 / Class-wise Accuracy")
    ax.set_ylabel("정확도 / Accuracy")
    st.pyplot(fig)

st.markdown("---")

if st.checkbox("📊 혼동 행렬 및 정확도 보기 / Show Confusion Matrix & Accuracy"):
    try:
        df = pd.read_csv("fashion_predictions.csv")
        y_true = df["True Label"]
        y_pred = df["Predicted Label"]
        col1, col2 = st.columns(2)
        with col1:
            plot_confusion_matrix(y_true, y_pred, labels)
        with col2:
            plot_class_accuracy(y_true, y_pred, labels)
    except Exception as e:
        st.error(f"❌ 오류 발생 / Error loading prediction file: {e}")

if st.checkbox("⬇️ 예측 기록 다운로드 / Download Prediction CSV"):
    try:
        df = pd.read_csv("fashion_predictions.csv")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📁 CSV 다운로드 / Download CSV", csv, "fashion_predictions.csv", "text/csv")
    except Exception as e:
        st.warning(f"다운로드 실패 / CSV download failed: {e}")

if st.checkbox("🖼️ 최근 업로드 이미지 보기 / Show Recent Uploads"):
    files = os.listdir(recent_folder)
    if files:
        cols = st.columns(min(5, len(files)))
        for i, file in enumerate(files[:10]):
            img = Image.open(os.path.join(recent_folder, file))
            cols[i % len(cols)].image(img, caption=file, use_container_width=True)
    else:
        st.info("⚠️ 업로드된 이미지가 없습니다 / No images found.")

client = OpenAI(api_key=st.secrets["openai"]["api_key"])

def generate_gpt_summary(df):
    class_counts = df["Predicted Label"].value_counts().to_dict()
    acc = (df["Predicted Label"] == df["True Label"]).mean()

    prompt = f"""
You are an AI assistant analyzing fashion image classification results.
Here is the data summary:
- Total predictions: {len(df)}
- Overall accuracy: {acc:.2%}
- Class distribution: {class_counts}

Please write a short summary (3–5 sentences) in English about the model performance, including which classes perform well or poorly.
"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that explains model performance."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=300
    )
    return response.choices[0].message.content

if st.checkbox("🧠 GPT 기반 모델 요약 보기 / Show GPT-based Model Summary"):
    try:
        df = pd.read_csv("fashion_predictions.csv")
        if len(df) < 2:
            st.warning("❗ 최소 2개의 예측 데이터가 필요합니다 / Need at least 2 records.")
        else:
            summary = generate_gpt_summary(df)
            st.markdown("### 📋 GPT 요약 결과 / GPT Summary Result")
            st.success(summary)
    except Exception as e:
        st.error(f"❌ 요약 생성 실패 / Failed to generate summary: {e}")
st.markdown("""
---
### 🧠 GPT 요약 기능 안내 / GPT Summary Guide

📌 이 기능은 저장된 예측 결과를 바탕으로 GPT가 모델의 성능을 분석 및 요약해주는 기능입니다.  
📌 This feature uses GPT to summarize your model performance based on saved predictions.

#### ✅ 사용 조건 / Requirements:
- 최소 **2개 이상의 예측 결과**가 저장되어 있어야 합니다.  
  At least **2 predictions** must be saved.
- 예측 후 **"예측 저장" 버튼**을 눌러야 데이터가 기록됩니다.  
  You must click **"Save Prediction"** after prediction to store the result.

#### 📤 출력 내용 / Output Includes:
- 전체 예측 수 / Total predictions
- 정확도 / Overall accuracy
- 클래스별 예측 분포 / Class-wise prediction distribution
- GPT가 자동 생성한 요약 (영문) / GPT-generated summary (in English)

> 🔐 GPT 요약은 OpenAI API를 통해 실행됩니다.  
> *GPT summary is powered by OpenAI API. Ensure your API key is configured properly.*

---
""")

